import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager

class TricklePublisher:
    def __init__(self, url: str, mime_type: str):
        self.url = url
        self.mime_type = mime_type
        self.idx = 0  # Start index for POSTs
        self.next_writer = None
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    def streamIdx(self):
        return f"{self.url}/{self.idx}"

    async def preconnect(self):
        """Preconnect to the server by initiating a POST request to the current index."""
        url = self.streamIdx()
        logging.debug(f"Preconnecting to URL: {url}")
        try:
            # we will be incrementally writing data into this queue
            queue = asyncio.Queue(maxsize=1)
            asyncio.create_task(self._run_post(url, queue))
            return queue
        except aiohttp.ClientError as e:
            logging.error(f"Failed to complete POST for {url}: {e}")
            return None

    async def _run_post(self, url, queue):
        try:
            resp = await self.session.post(
                url,
                headers={'Connection': 'close', 'Content-Type': self.mime_type},
                data=self._stream_data(queue)
            )
            # TODO propagate errors?
            if resp.status != 200:
                body = await resp.text()
                logging.error(f"Trickle POST failed {url}, status code: {resp.status}, msg: {body}")
        except Exception as e:
            logging.exception(f"Trickle POST exception {url}")
        return None

    async def _run_delete(self):
        try:
            await self.session.delete(self.url)
        except Exception:
            logging.error(f"Error sending trickle delete request", exc_info=True)

    async def _stream_data(self, queue):
        """Stream data from the queue for the POST request."""
        while True:
            chunk = await queue.get()
            if chunk is None:  # Stop signal
                break
            yield chunk

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        async with self.lock:
            if self.next_writer is None:
                logging.info(f"No pending connection, preconnecting {self.streamIdx()}...")
                self.next_writer = await self.preconnect()

            seq = self.idx
            writer = self.next_writer
            self.next_writer = None

            # Set up the next POST in the background
            asyncio.create_task(self._preconnect_next_segment())

        return SegmentWriter(writer, seq)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        async with self.lock:
            if self.next_writer is not None:
                return
            self.idx += 1  # Increment the index for the next POST
            next_writer = await self.preconnect()
            if next_writer:
                self.next_writer = next_writer

    async def close(self):
        """Close the session when done."""
        logging.info(f"Closing {self.url}")
        async with self.lock:
            if self.next_writer:
                s = SegmentWriter(self.next_writer)
                await s.close()
                self.next_writer = None
            if self.session:
                try:
                    await self._run_delete()
                    await self.session.close()
                except Exception:
                    logging.error(f"Error closing trickle publisher", exc_info=True)
                finally:
                    self.session = None

class SegmentWriter:
    def __init__(self, queue: asyncio.Queue, seq: int = -99):
        self.queue = queue
        self._seq = seq

    async def write(self, data):
        """Write data to the current segment."""
        await self.queue.put(data)

    async def close(self):
        """Ensure the request is properly closed when done."""
        await self.queue.put(None)  # Send None to signal end of data

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the connection."""
        await self.close()

    def seq(self) -> int:
        """Return the sequence number of this segment."""
        return self._seq
