import time
import aiohttp
import asyncio
import logging
import os
import threading
import subprocess

from .trickle_subscriber import TrickleSubscriber
from .trickle_publisher import TricklePublisher
from .decoder import decode_av
from .encoder import encode_av

async def run_subscribe(subscribe_url: str, image_callback, put_metadata, monitoring_callback):
    # TODO add some pre-processing parameters, eg image size
    try:
        read_fd, write_fd = os.pipe()
        parse_task = asyncio.create_task(decode_in(read_fd, image_callback, put_metadata))
        subscribe_task = asyncio.create_task(subscribe(subscribe_url, await AsyncifyFdWriter(write_fd), monitoring_callback))
        await asyncio.gather(subscribe_task, parse_task)
        logging.info("run_subscribe complete")
    except Exception as e:
        logging.error(f"preprocess got error {e}", e)
        raise e
    finally:
        put_metadata(None) # in case decoder quit without writing anything

async def subscribe(subscribe_url, out_pipe, monitoring_callback):
    first_segment = True

    async with TrickleSubscriber(url=subscribe_url) as subscriber:
        logging.info(f"launching subscribe loop for {subscribe_url}")
        while True:
            segment = None
            try:
                segment = await subscriber.next()
                if not segment:
                    break # complete
                if first_segment:
                    first_segment = False
                    await monitoring_callback({
                        "type": "runner_receive_first_ingest_segment",
                        "timestamp": int(time.time() * 1000)
                    }, queue_event_type="stream_trace")
                while True:
                    chunk = await segment.read()
                    if not chunk:
                        break # end of segment
                    out_pipe.write(chunk)
                    await out_pipe.drain()
            except aiohttp.ClientError as e:
                logging.info(f"Failed to read segment - {e}")
                break # end of stream?
            except Exception as e:
                raise e
            finally:
                if segment:
                    await segment.close()
                else:
                    # stream is complete
                    out_pipe.close()
                    break

async def AsyncifyFdWriter(write_fd):
    loop = asyncio.get_event_loop()
    write_protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader(), loop=loop)
    write_transport, _ = await loop.connect_write_pipe( lambda: write_protocol, os.fdopen(write_fd, 'wb'))
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    return writer

async def decode_in(in_pipe, frame_callback, put_metadata):
    def decode_runner():
        try:
            decode_av(f"pipe:{in_pipe}", frame_callback, put_metadata)
        except Exception as e:
            logging.error(f"Decoding error {e}", exc_info=True)
        finally:
            os.close(in_pipe)
            logging.info("Decoding finished")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, decode_runner)

async def run_publish(publish_url: str, image_generator, get_metadata, monitoring_callback):
    first_segment = True

    publisher = None
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")

        loop = asyncio.get_running_loop()
        async def callback(pipe_file, pipe_name):
            nonlocal first_segment
            # trickle publish a segment with the contents of `pipe_file`
            async with await publisher.next() as segment:
                # convert pipe_fd into an asyncio friendly StreamReader
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe_file)
                while True:
                    sz = 32 * 1024 # read in chunks of 32KB
                    data = await reader.read(sz)
                    if not data:
                        break
                    await segment.write(data)
                    if first_segment:
                        first_segment = False
                        await monitoring_callback({
                            "type": "runner_send_first_processed_segment",
                            "timestamp": int(time.time() * 1000)
                        }, queue_event_type="stream_trace")
                transport.close()

        def sync_callback(pipe_file, pipe_name):
            def do_schedule():
                schedule_callback(callback(pipe_file, pipe_name), pipe_name)
            loop.call_soon_threadsafe(do_schedule)

        # hold tasks since `loop.create_task` is a weak reference that gets GC'd
        # TODO use asyncio.TaskGroup once all pipelines are on Python 3.11+
        live_tasks = set()
        live_tasks_lock = threading.Lock()

        def schedule_callback(coro, pipe_name):
            task = loop.create_task(coro)
            with live_tasks_lock:
                live_tasks.add(task)
            def task_done(t: asyncio.Task):
                try:
                    t.result()
                except Exception as e:
                    logging.error(f"Task {pipe_name} crashed: {e}")
                with live_tasks_lock:
                    live_tasks.remove(t)
            task.add_done_callback(task_done)

        encode_thread = threading.Thread(target=encode_av, args=(image_generator, sync_callback, get_metadata), kwargs={"audio_codec":"libopus"})
        encode_thread.start()
        logging.debug("run_publish: encoder thread started")

        # Wait for encode thread to complete
        def joins():
            encode_thread.join()
        await asyncio.to_thread(joins)

        # wait for IO tasks to complete
        # TODO use asyncio.TaskGroup once all pipelines are on python 3.11+
        while True:
            with live_tasks_lock:
                current_tasks = list(live_tasks)
            if not current_tasks:
                break  # nothing left to wait on
            await asyncio.wait(current_tasks, return_when=asyncio.ALL_COMPLETED)
            # loop in case another task was added while awaiting

        logging.info("run_publish complete")

    except Exception as e:
        logging.error(f"postprocess got error {e}", e)
        raise e
    finally:
        if publisher:
            await publisher.close()
