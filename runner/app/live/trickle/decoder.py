import av
from av.video.reformatter import VideoReformatter
from av.container import InputContainer
import time
import logging
from typing import cast
import numpy as np
import torch

from .frame import InputFrame

MAX_FRAMERATE=24
ENGINE_DIMENSIONS: dict[str, int] = {
    "width": 384,
    "height": 704
} # Static dimensions for the only supported engine

def decode_av(pipe_input, frame_callback, put_metadata, output_width, output_height):
    """
    Reads from a pipe (or file-like object).

    :param pipe_input: File path, 'pipe:', sys.stdin, or another file-like object.
    :param frame_callback: A function that accepts an InputFrame object
    :param put_metadata: A function that accepts audio/video metadata
    """
    container = cast(InputContainer, av.open(pipe_input, 'r'))

    # Locate the first video and first audio stream (if they exist)
    video_stream = None
    audio_stream = None
    if container.streams.video:
        video_stream = container.streams.video[0]
    if container.streams.audio:
        audio_stream = container.streams.audio[0]

    # Prepare audio-related metadata (if audio is present)
    audio_metadata = None
    if audio_stream is not None:
        audio_metadata = {
            "codec": audio_stream.codec_context.name,
            "sample_rate": audio_stream.codec_context.sample_rate,
            "format": audio_stream.codec_context.format.name,
            "channels": audio_stream.codec_context.channels,
            "layout": audio_stream.layout.name,
            "time_base": audio_stream.time_base,
            "bit_rate": audio_stream.codec_context.bit_rate,
        }

    # Prepare video-related metadata (if video is present)
    video_metadata = None
    if video_stream is not None:
        # Check if dimensions are flipped (90 degree rotation needed)
        needs_rotation = (output_width == ENGINE_DIMENSIONS["height"] and output_height == ENGINE_DIMENSIONS["width"])
        if needs_rotation:
            logging.info(f"Dimensions flipped detected: Input {video_stream.codec_context.width}x{video_stream.codec_context.height} -> Output {output_width}x{output_height}, rotation will be applied")
        
        video_metadata = {
            "codec": video_stream.codec_context.name,
            "width": video_stream.codec_context.width,
            "height": video_stream.codec_context.height,
            "pix_fmt": video_stream.codec_context.pix_fmt,
            "time_base": video_stream.time_base,
            # framerate is usually unreliable, especially with webrtc
            "framerate": video_stream.codec_context.framerate,
            "sar": video_stream.codec_context.sample_aspect_ratio,
            "dar": video_stream.codec_context.display_aspect_ratio,
            "format": str(video_stream.codec_context.format),
            "output_width": output_width,
            "output_height": output_height,
            "needs_rotation": needs_rotation,
        }

    if video_metadata is None and audio_metadata is None:
        logging.error("No audio or video streams found in the input.")
        container.close()
        return

    metadata = { 'video': video_metadata, 'audio': audio_metadata }
    logging.info(f"Metadata: {metadata}")
    put_metadata(metadata)

    reformatter = VideoReformatter()
    frame_interval = 1.0 / MAX_FRAMERATE
    next_pts_time = 0.0
    try:
        for packet in container.demux():
            if packet.dts is None:
                continue

            if audio_stream and packet.stream == audio_stream:
                # Decode audio frames
                for aframe in packet.decode():
                    aframe = cast(av.AudioFrame, aframe)
                    if aframe.pts is None:
                        continue

                    avframe = InputFrame.from_av_audio(aframe)
                    avframe.log_timestamps["frame_init"] = time.time()
                    frame_callback(avframe)
                    continue

            elif video_stream and packet.stream == video_stream:
                # Decode video frames
                for frame in packet.decode():
                    frame = cast(av.VideoFrame, frame)
                    if frame.pts is None:
                        continue
                    # drop frames that come in too fast
                    # TODO also check timing relative to wall clock
                    pts_time = frame.time
                    if pts_time < next_pts_time:
                        # frame is too early, so drop it
                        continue
                    if pts_time > next_pts_time + frame_interval:
                        # frame is delayed, so reset based on frame pts
                        next_pts_time = pts_time + frame_interval
                    else:
                        # not delayed, so use prev pts to allow more jitter
                        next_pts_time = next_pts_time + frame_interval
                    # Convert frame to image
                    image = frame.to_image()
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    width, height = image.size

                    # Apply rotation first if needed
                    if video_metadata["needs_rotation"]:
                        logging.debug(f"Rotating frame 90 degrees clockwise: {width}x{height} -> {height}x{width}")
                        image = image.rotate(90, expand=True)
                        width, height = image.size
                        # When rotation is needed, we resize to engine dimensions
                        target_width = ENGINE_DIMENSIONS["width"]
                        target_height = ENGINE_DIMENSIONS["height"]
                    else:
                        target_width = output_width
                        target_height = output_height

                    # Calculate aspect ratios
                    input_ratio = width / height
                    output_ratio = target_width / target_height

                    if input_ratio != output_ratio:
                        # Need to crop to match output aspect ratio
                        if input_ratio > output_ratio:
                            # Input is wider than output - crop width
                            new_width = int(height * output_ratio)
                            start_x = (width - new_width) // 2
                            logging.debug(f"Cropping width: {width}x{height} -> {new_width}x{height}")
                            image = image.crop((start_x, 0, start_x + new_width, height))
                        else:
                            # Input is taller than output - crop height
                            new_height = int(width / output_ratio)
                            start_y = (height - new_height) // 2
                            logging.debug(f"Cropping height: {width}x{height} -> {width}x{new_height}")
                            image = image.crop((0, start_y, width, start_y + new_height))

                    # Resize to target dimensions
                    if (target_width, target_height) != image.size:
                        logging.debug(f"Resizing frame: {image.size} -> {target_width}x{target_height}")
                        image = image.resize((target_width, target_height))

                    # Convert to tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    
                    # Create tensor, ensuring array is contiguous to avoid negative strides
                    tensor = torch.tensor(np.ascontiguousarray(image_np)).unsqueeze(0)

                    avframe = InputFrame.from_av_video(tensor, frame.pts, frame.time_base)
                    avframe.log_timestamps["frame_init"] = time.time()
                    frame_callback(avframe)
                    continue

    except Exception as e:
        logging.error(f"Exception while decoding: {e}")
        raise # should be caught upstream

    finally:
        container.close()

    logging.info("Decoder stopped")
