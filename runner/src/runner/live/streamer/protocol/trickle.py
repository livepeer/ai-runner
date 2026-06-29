import asyncio
import logging
import os
import queue
import json
from typing import AsyncGenerator, Optional

from PIL import Image

from ...trickle import media, TricklePublisher, TrickleSubscriber, InputFrame, OutputFrame, AudioFrame, AudioOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT

from .protocol import StreamProtocol
from .last_value_cache import LastValueCache

DEFAULT_PUBLISH_QUEUE_MAXSIZE = 1


def _publish_queue_maxsize() -> int:
    raw = os.environ.get("LIVE_TRICKLE_PUBLISH_QUEUE_MAXSIZE")
    if raw is None or raw.strip() == "":
        return DEFAULT_PUBLISH_QUEUE_MAXSIZE
    try:
        value = int(raw)
    except ValueError:
        logging.warning(
            "Invalid LIVE_TRICKLE_PUBLISH_QUEUE_MAXSIZE=%r, using %s",
            raw,
            DEFAULT_PUBLISH_QUEUE_MAXSIZE,
        )
        return DEFAULT_PUBLISH_QUEUE_MAXSIZE
    return max(1, value)


def _put_drop_oldest(_queue: queue.Queue, item) -> int:
    dropped = 0
    while True:
        try:
            _queue.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                old = _queue.get_nowait()
            except queue.Empty:
                continue
            dropped += 1
            del old


class TrickleProtocol(StreamProtocol):
    def __init__(self, subscribe_url: str, publish_url: str, control_url: Optional[str] = None, events_url: Optional[str] = None, input_width: Optional[int] = DEFAULT_WIDTH, input_height: Optional[int] = DEFAULT_HEIGHT, output_width: Optional[int] = None, output_height: Optional[int] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.subscribe_queue = queue.Queue[InputFrame]()
        self.publish_queue = queue.Queue[OutputFrame](maxsize=_publish_queue_maxsize())
        self.publish_queue_dropped = 0
        self.control_subscriber = None
        self.events_publisher = None
        self.subscribe_task = None
        self.publish_task = None
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width if output_width is not None else input_width
        self.output_height = output_height if output_height is not None else input_height

    async def start(self):
        self.subscribe_queue = queue.Queue[InputFrame]()
        self.publish_queue = queue.Queue[OutputFrame](maxsize=_publish_queue_maxsize())
        self.publish_queue_dropped = 0
        metadata_cache = LastValueCache[dict]() # to pass video metadata from decoder to encoder
        self.subscribe_task = asyncio.create_task(
            media.run_subscribe(self.subscribe_url, self.subscribe_queue.put, metadata_cache.put, self.emit_monitoring_event, self.input_width, self.input_height)
        )
        self.publish_task = asyncio.create_task(
            media.run_publish(self.publish_url, self.publish_queue.get, metadata_cache.get, self.emit_monitoring_event, self.output_width, self.output_height)
        )
        if self.control_url and self.control_url.strip() != "":
            self.control_subscriber = TrickleSubscriber(self.control_url)
        if self.events_url and self.events_url.strip() != "":
            self.events_publisher = TricklePublisher(self.events_url, "application/json")

    async def stop(self):
        if not self.subscribe_task or not self.publish_task:
            return # already stopped

        # send sentinel None values to stop the trickle tasks gracefully
        self.subscribe_queue.put(None)
        _put_drop_oldest(self.publish_queue, None)

        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None

        tasks = [self.subscribe_task, self.publish_task]
        try:
            await asyncio.wait(tasks, timeout=10.0)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()

        self.subscribe_task = None
        self.publish_task = None

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
                self.publish_queue_dropped += _put_drop_oldest(
                    publish_queue,
                    AudioOutput([image]),
                )
                continue
            yield image

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        publish_queue = self.publish_queue
        def enqueue_bytes(frame: OutputFrame):
            return _put_drop_oldest(publish_queue, frame)

        async for frame in output_frames:
            try:
                dropped = await asyncio.to_thread(enqueue_bytes, frame)
                self.publish_queue_dropped += dropped
                if dropped and self.publish_queue_dropped % 100 == 1:
                    logging.warning(
                        "Dropped stale publish frames dropped_total=%s queue_maxsize=%s",
                        self.publish_queue_dropped,
                        publish_queue.maxsize,
                    )
            finally:
                del frame

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        if not self.events_publisher:
            return
        try:
            event_json = json.dumps({"event": event, "queue_event_type": queue_event_type})
            async with await self.events_publisher.next() as event:
                await event.write(event_json.encode())
        except Exception as e:
            logging.error(f"Error reporting status: {e}")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        if not self.control_subscriber:
            logging.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logging.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not done.is_set():
            try:
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    return

                params = await segment.read()
                data = json.loads(params)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    continue

                logging.info("Received control message with params: %s", data)
                yield data

            except Exception:
                logging.error(f"Error in control loop", exc_info=True)
                continue
