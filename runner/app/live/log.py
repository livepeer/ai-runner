import time
import logging
from contextlib import contextmanager


logger: logging.Logger | None = None
handler: logging.Handler | None = None


def config_logging(*, log_level: int = 0, request_id: str = "", manifest_id: str = "", stream_id: str = ""):
    global logger, handler
    if logger and handler:
        if log_level:
            logger.setLevel(log_level)
            handler.setLevel(log_level)
        config_logging_fields(handler, request_id, manifest_id, stream_id)
        return logger

    handler = logging.StreamHandler()
    if log_level:
        handler.setLevel(log_level)
    config_logging_fields(handler, request_id, manifest_id, stream_id)

    logger = logging.getLogger()  # Root logger
    if log_level:
        logger.setLevel(log_level)
    logger.addHandler(handler)
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
    global logger
    start_time = time.time()
    status = "failure"
    try:
        yield
        status = "success"
    finally:
        duration = time.time() - start_time
        logger.info(f"operation={operation_name} status={status} duration_s={duration}s")
