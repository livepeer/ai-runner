import av
from av.video.reformatter import VideoReformatter
from av.container import InputContainer
import time
import logging
from typing import cast
import numpy as np
import torch
from PIL import Image

from .frame import InputFrame

MAX_FRAMERATE=24

def resize_and_crop(image, target_width, target_height, crop_mode='center'):
    """
    Resize and crop an image to match target dimensions while maintaining aspect ratio.
    
    Args:
        image: PIL Image to process
        target_width: Target width
        target_height: Target height
        crop_mode: One of 'center', 'top', 'bottom', 'left', 'right'
    
    Returns:
        Resized and cropped PIL Image
    """
    # Calculate aspect ratios
    input_ratio = image.width / image.height
    target_ratio = target_width / target_height
    
    if input_ratio == target_ratio:
        # Direct resize if ratios match
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # First resize to match the larger dimension
    if input_ratio > target_ratio:
        # Input is wider - resize to match height
        new_height = target_height
        new_width = int(new_height * input_ratio)
    else:
        # Input is taller - resize to match width
        new_width = target_width
        new_height = int(new_width / input_ratio)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Then crop to target dimensions
    if input_ratio > target_ratio:
        # Need to crop width
        excess_width = new_width - target_width
        if crop_mode == 'center':
            left = excess_width // 2
        elif crop_mode == 'left':
            left = 0
        elif crop_mode == 'right':
            left = excess_width
        else:
            left = excess_width // 2
        return resized.crop((left, 0, left + target_width, target_height))
    else:
        # Need to crop height
        excess_height = new_height - target_height
        if crop_mode == 'center':
            top = excess_height // 2
        elif crop_mode == 'top':
            top = 0
        elif crop_mode == 'bottom':
            top = excess_height
        else:
            top = excess_height // 2
        return resized.crop((0, top, target_width, top + target_height))

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
                    
                    # Resize and crop the image
                    image = resize_and_crop(image, output_width, output_height, crop_mode='center')
                    
                    # Convert to tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    tensor = torch.tensor(image_np).unsqueeze(0)

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
