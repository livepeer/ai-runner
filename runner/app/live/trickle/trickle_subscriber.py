import asyncio
import aiohttp
import logging
import sys

class TrickleSubscriber:
    def __init__(self, url: str, start_seq=-2, max_retries=5):
        self.base_url = url
        self.idx = start_seq
        self.pending_get: aiohttp.ClientResponse | None = None  # Pre-initialized GET request
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
        self.errored = False
        self.max_retries = max_retries

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    async def preconnect(self):
        """
            Preconnect to the server by making a GET request to fetch the next segment.
            For any non-200 responses, retries up to max_retries unless a 404 is encountered
        """
        url = f"{self.base_url}/{self.idx}"
        for attempt in range(0, self.max_retries):
            logging.info(f"Trickle sub Preconnecting attempt: {attempt} URL: {url}")
            try:

                resp = await self.session.get(url, headers={'Connection':'close'})

                if resp.status == 200:
                    # Return the response for later processing
                    return resp

                if resp.status == 404:
                    logging.info(f"Trickle sub got 404, terminating {url}")
                    resp.release()
                    self.errored = True
                    return None

                if resp.status == 470:
                    # channel exists but no data at this index, so reset
                    idx = resp.headers.get('Lp-Trickle-Latest') or '-1'
                    url = f"{self.base_url}/{idx}"
                    logging.info(f"Trickle sub resetting index to leading edge {url}")
                    resp.release()
                    # continue immediately
                    continue

                body = await resp.text()
                resp.release()
                logging.error(f"Trickle sub Failed GET {url} status code: {resp.status}, msg: {body}")

            except Exception:
                logging.exception(f"Trickle sub Failed to complete GET {url}", stack_info=True)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.5)

        # max retries hit, so bail out
        logging.error(f"Trickle sub hit max retries, exiting {url}")
        self.errored = True
        return None

    async def next(self):
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:

            if self.errored:
                logging.info(f"Trickle subscription closed or errored for {self.base_url}")
                return None

            # If we don't have a pending GET request, preconnect
            if self.pending_get is None:
                logging.info("Trickle sub No pending connection, preconnecting...")
                self.pending_get = await self.preconnect()

            # Extract the current connection to use for reading
            resp = self.pending_get
            self.pending_get = None

            # Preconnect has failed, notify caller
            if resp is None:
                return None

            # Extract and set the next index from the response headers
            segment = Segment(resp)

            if segment.eos():
                return None

            idx = segment.seq()
            if idx >= 0:
                self.idx = idx + 1

            # Set up the next connection in the background
            asyncio.create_task(self._preconnect_next_segment())

        return segment

    async def _preconnect_next_segment(self):
        """Preconnect to the next segment in the background."""
        async with self.lock:
            if self.pending_get is not None:
                return
            next_conn = await self.preconnect()
            if next_conn:
                self.pending_get = next_conn

    async def close(self):
        """Close the session when done."""
        logging.info(f"Closing {self.base_url}")
        async with self.lock:
            if self.pending_get:
                self.pending_get.close()
                self.pending_get = None
            if self.session:
                try:
                    await self.session.close()
                except Exception:
                    logging.error(f"Error closing trickle subscriber", exc_info=True)
                finally:
                    self.session = None

class Segment:
    def __init__(self, response):
        self.response = response

    def seq(self):
        """Extract the sequence number from the response headers."""
        seq_str = self.response.headers.get('Lp-Trickle-Seq')
        try:
            seq = int(seq_str)
        except (TypeError, ValueError):
            return -1
        return seq

    def eos(self):
        return self.response.headers.get('Lp-Trickle-Closed') != None

    async def read(self, chunk_size=32 * 1024):
        """Read the next chunk of the segment."""
        if not self.response:
            await self.close()
            return None
        chunk = await self.response.content.read(chunk_size)
        if not chunk:
            await self.close()
        return chunk

    async def close(self):
        """Ensure the response is properly closed when done."""
        if self.response is None:
            return
        if not self.response.closed:
            self.response.release()
            self.response.close()
