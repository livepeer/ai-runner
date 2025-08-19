import time
import aiohttp
import asyncio
import logging
import os
import threading

from .trickle_subscriber import TrickleSubscriber
from .trickle_publisher import TricklePublisher
from .decoder import decode_av
from .encoder import encode_av, reset_encoder

MAX_DECODER_RETRIES = 3
DECODER_RETRY_RESET_SECONDS = 120 # reset retry counter after 2 minutes

MAX_ENCODER_RETRIES = 3
ENCODER_RETRY_RESET_SECONDS = 120 # reset retry counter after 2 minutes

async def run_subscribe(subscribe_url: str, image_callback, put_metadata, monitoring_callback, target_width, target_height):
    # TODO add some pre-processing parameters, eg image size
    try:
        in_pipe, out_pipe = os.pipe()
        write_fd = await AsyncifyFdWriter(out_pipe)
        parse_task = asyncio.create_task(decode_in(in_pipe, image_callback, put_metadata, write_fd, target_width, target_height))
        subscribe_task = asyncio.create_task(subscribe(subscribe_url, write_fd, monitoring_callback))
        await asyncio.gather(subscribe_task, parse_task)
        logging.info("run_subscribe complete")
    except Exception as e:
        logging.exception("run_subscribe got error", stack_info=True)
    finally:
        put_metadata(None) # in case decoder quit without writing anything
        image_callback(None) # stops inference if this function exits early

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
                while True:
                    chunk = await segment.read()
                    if not chunk:
                        break # end of segment
                    out_pipe.write(chunk)
                    await out_pipe.drain()
                if first_segment:
                    first_segment = False
                    await monitoring_callback({
                        "type": "runner_receive_first_ingest_segment",
                        "timestamp": int(time.time() * 1000)
                    }, queue_event_type="stream_trace")
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

async def decode_in(in_pipe, frame_callback, put_metadata, write_fd, target_width, target_height):
    def decode_runner():
        retry_count = 0
        last_retry_time = time.time()
        while retry_count < MAX_DECODER_RETRIES:
            try:
                decode_av(f"pipe:{in_pipe}", frame_callback, put_metadata, target_width, target_height)
                break  # clean exit
            except Exception as e:
                msg = str(e)
                if f"Invalid data found when processing input: 'pipe:{in_pipe}'" in msg:
                    logging.info("Stream closed before initialization")
                    break
                current_time = time.time()
                # Reset retry counter if enough time has elapsed
                if current_time - last_retry_time > DECODER_RETRY_RESET_SECONDS:
                    logging.info("Resetting decoder retry count")
                    retry_count = 0
                retry_count += 1
                last_retry_time = current_time
                if retry_count < MAX_DECODER_RETRIES:
                    logging.exception(f"Error in decode_av, retrying {retry_count}/{MAX_DECODER_RETRIES}", stack_info=True)
                else:
                    logging.exception("Error in decode_av, maximum retries reached", stack_info=True)

        try:
            # force write end of pipe to close to terminate trickle subscriber
            write_fd.close()
        except Exception:
            # happens sometimes but ignore
            pass

        os.close(in_pipe)
        logging.info("Decoding finished")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, decode_runner)

def encode_in(task_pipes, task_lock, image_generator, sync_callback, get_metadata, **kwargs):
    # encode_av has a tendency to crash, so restart as necessary
    retryCount = 0
    last_retry_time = time.time()
    recovery_bag = {} # opaque bag of stuff to help recover from encoder crashes
    while retryCount < MAX_ENCODER_RETRIES:
        try:
            encode_av(image_generator, sync_callback, get_metadata, recovery_bag, **kwargs)
            break  # clean exit
        except Exception as exc:
            current_time = time.time()
            # Reset retry counter if enough time has elapsed
            if current_time - last_retry_time > ENCODER_RETRY_RESET_SECONDS:
                logging.info("Resetting encoder retry count")
                retryCount = 0
            retryCount += 1
            last_retry_time = current_time
            if retryCount < MAX_ENCODER_RETRIES:
                logging.exception(f"Error in encode_av, retrying {retryCount}/{MAX_ENCODER_RETRIES}", stack_info=True)
            else:
                logging.exception("Error in encode_av, maximum retries reached", stack_info=True)

            # first reset the encoder before closing pipes
            reset_encoder(recovery_bag)

            # close leftover writer ends of any pipes to prevent hanging
            pipe_count = 0
            total_pipes = 0
            with task_lock:
                pipes = list(task_pipes)
                total_pipes = len(pipes)
                for p in pipes:
                    try:
                        p.close()
                        pipe_count += 1
                    except Exception as e:
                        logging.exception("Error closing pipe on task list", stack_info=True)
            logging.info(f"Closed pipes - {pipe_count}/{total_pipes}")

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
                start_time = time.time()
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
                logging.info(f"Published seq={segment.seq()} took={time.time()-start_time}s")

        def sync_callback(pipe_reader, pipe_writer, pipe_name):
            def do_schedule():
                schedule_callback(callback(pipe_reader, pipe_name), pipe_writer, pipe_name)
            loop.call_soon_threadsafe(do_schedule)

        # hold tasks since `loop.create_task` is a weak reference that gets GC'd
        # TODO use asyncio.TaskGroup once all pipelines are on Python 3.11+
        # Also hold pipes in case we need to explicitly close them during exceptions
        live_tasks = set()
        live_pipes = set()
        live_tasks_lock = threading.Lock()

        def schedule_callback(coro, pipe_writer, pipe_name):
            task = loop.create_task(coro)
            with live_tasks_lock:
                live_tasks.add(task)
                live_pipes.add(pipe_writer)
            def task_done2(t: asyncio.Task, p):
                try:
                    t.result()
                except Exception as e:
                    logging.error(f"Task {pipe_name} crashed: {e}")
                with live_tasks_lock:
                    live_tasks.remove(t)
                    live_pipes.remove(p)
            def task_done(t2:asyncio.Task):
                return task_done2(task, pipe_writer)
            task.add_done_callback(task_done)

        encode_thread = threading.Thread(target=encode_in, args=(live_pipes, live_tasks_lock, image_generator, sync_callback, get_metadata), kwargs={"audio_codec":"libopus"})
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
