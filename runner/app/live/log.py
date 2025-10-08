import logging
import time
from contextlib import contextmanager


logger: logging.Logger | None = None
handler: logging.Handler | None = None


def config_logging(*, log_level: int = 0, request_id: str = "", manifest_id: str = "", stream_id: str = ""):
    global logger, handler

    if not logger:
        logger = logging.getLogger()  # Root logger
        for cur_handler in logger.handlers:
            if isinstance(cur_handler, logging.StreamHandler):
                handler = cur_handler
                break
    if not handler:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    if log_level:
        logger.setLevel(log_level)
        handler.setLevel(log_level)
    config_logging_fields(handler, request_id, manifest_id, stream_id)

    return logger


def config_logging_fields(handler: logging.Handler, request_id: str, manifest_id: str, stream_id: str):
    formatter = logging.Formatter(
        "timestamp=%(asctime)s level=%(levelname)s location=%(filename)s:%(lineno)d:%(funcName)s gateway_request_id=%(request_id)s manifest_id=%(manifest_id)s stream_id=%(stream_id)s message=%(message)s",
        defaults={"request_id": request_id, "manifest_id": manifest_id, "stream_id": stream_id},
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

@contextmanager
def log_timing(operation_name: str):
    start_time = time.time()
    status = "failure"
    try:
        yield
        status = "success"
    finally:
        duration = time.time() - start_time
        logging.info(f"operation={operation_name} status={status} duration_s={duration}s")
