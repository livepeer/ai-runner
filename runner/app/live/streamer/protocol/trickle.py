import asyncio
import logging
import queue
import json
import time
from typing import AsyncGenerator, Optional, Any

from trickle import media, TricklePublisher, TrickleSubscriber, InputFrame, OutputFrame, AudioFrame, AudioOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT

from .protocol import StreamProtocol
from .last_value_cache import LastValueCache

class TrickleProtocol(StreamProtocol):
    def __init__(self, subscribe_url: str, publish_url: str, control_url: Optional[str] = None, events_url: Optional[str] = None, width: Optional[int] = DEFAULT_WIDTH, height: Optional[int] = DEFAULT_HEIGHT):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.subscribe_queue = queue.Queue[InputFrame | None]()
        self.publish_queue = queue.Queue[OutputFrame | None]()
        self.event_queue = asyncio.Queue[dict | None]()
        self.control_subscriber = None
        self.events_publisher = None
        self.subscribe_task = None
        self.publish_task = None
        self.event_batcher_task = None
        self.width = width
        self.height = height

    async def start(self):
        self.subscribe_queue = queue.Queue[InputFrame | None]()
        self.publish_queue = queue.Queue[OutputFrame | None]()
        self.event_queue = asyncio.Queue[dict | None]()
        metadata_cache = LastValueCache[dict]() # to pass video metadata from decoder to encoder
        self.subscribe_task = asyncio.create_task(
            media.run_subscribe(self.subscribe_url, self.subscribe_queue.put, metadata_cache.put, self.emit_monitoring_event, self.width, self.height)
        )
        self.publish_task = asyncio.create_task(
            media.run_publish(self.publish_url, self.publish_queue.get, metadata_cache.get, self.emit_monitoring_event)
        )
        if self.control_url and self.control_url.strip() != "":
            self.control_subscriber = TrickleSubscriber(self.control_url)
        if self.events_url and self.events_url.strip() != "":
            self.events_publisher = TricklePublisher(self.events_url, "application/json")
            self.event_batcher_task = asyncio.create_task(self._event_batcher_loop())

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            return # already stopped

        # send sentinel None values to stop the trickle tasks gracefully
        self.subscribe_queue.put(None)
        self.publish_queue.put(None)
        await self.event_queue.put(None)

        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None

        tasks = [self.subscribe_task, self.publish_task, self.event_batcher_task]
        try:
            await asyncio.wait(tasks, timeout=10.0)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.subscribe_task = None
        self.publish_task = None
        self.event_batcher_task = None

    async def _event_batcher_loop(self):
        """Background task that batches events and sends them as JSONL segments."""
        BATCH_TIMEOUT = 5.0  # 5 seconds timeout

        if not self.events_publisher:
            raise ValueError("Events publisher not initialized")

        while True:
            try:
                event_data = await self.event_queue.get()
                if not event_data:
                    return

                async with await self.events_publisher.next() as current_segment:
                    segment_start_time = time.time()

                    event_json = json.dumps(event_data)
                    await current_segment.write(event_json.encode() + b'\n')

                    while True:
                        timeout = BATCH_TIMEOUT - (time.time() - segment_start_time)
                        if timeout <= 0.1:
                            break

                        try:
                            next_event = await asyncio.wait_for(self.event_queue.get(), timeout=timeout)
                            if not next_event:
                                return
                        except asyncio.TimeoutError:
                            break

                        event_json = json.dumps(next_event)
                        await current_segment.write(event_json.encode() + b'\n')

                logging.debug("Sent batched events segment")

            except Exception:
                logging.error(f"Error in event batcher loop", exc_info=True)

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        subscribe_queue = self.subscribe_queue
        publish_queue = self.publish_queue
        def dequeue_frame():
            frame = subscribe_queue.get()
            if not frame:
                return None

            return frame

        while not done.is_set():
            image = await asyncio.to_thread(dequeue_frame)
            if not image:
                break
            # TEMP: Put audio immediately into the publish queue
            # TODO: Remove once there is ComfyUI audio support
            if isinstance(image, AudioFrame):
                publish_queue.put(AudioOutput([image]))
                continue
            yield image

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        publish_queue = self.publish_queue
        def enqueue_bytes(frame: OutputFrame):
            publish_queue.put(frame)

        async for frame in output_frames:
            await asyncio.to_thread(enqueue_bytes, frame)

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        if not self.events_publisher:
            return

        event_data = {"event": event, "queue_event_type": queue_event_type}
        await self.event_queue.put(event_data)

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        if not self.control_subscriber:
            logging.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logging.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        def read_event_line(line: str) -> Any:
            """Read a single event line. Returns None for JSON decode errors and keepalive messages."""
            try:
                data = json.loads(line)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    return None

                logging.info("Received control message with params: %s", data)
                return data
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON line: {line}, error: {e}")
                return None

        while not done.is_set():
            try:
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    return

                buffer = ""

                while True:
                    chunk = await segment.read()
                    if not chunk:
                        break

                    buffer += chunk.decode('utf-8')

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue

                        event = read_event_line(line)
                        if event:
                            yield event

                buffer = buffer.strip()
                if buffer:
                    event = read_event_line(buffer)
                    if event:
                        yield event

            except Exception:
                logging.error(f"Error in control loop", exc_info=True)
                continue

