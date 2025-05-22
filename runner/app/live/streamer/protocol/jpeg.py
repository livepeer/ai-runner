import io
from PIL import Image


def to_jpeg_bytes(frame: Image.Image):
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG")
    bytes = buffer.getvalue()
    buffer.close()
    return bytes


def from_jpeg_bytes(frame_bytes: bytes):
    image = Image.open(io.BytesIO(frame_bytes))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image
