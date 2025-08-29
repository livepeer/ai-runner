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

def decode_av(pipe_input, frame_callback, put_metadata, target_width, target_height):
    """
    Reads from a pipe (or file-like object).

    :param pipe_input: File path, 'pipe:', sys.stdin, or another file-like object.
    :param frame_callback: A function that accepts an InputFrame object
    :param put_metadata: A function that accepts audio/video metadata
    """

    container = cast(InputContainer, av.open(pipe_input, mode='r', options={
        # probe for up to 2.5 seconds of pts time
        'analyzeduration': '2500000',
    }))

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
            "target_width": target_width,
            "target_height": target_height,
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

                    # Use efficient reformatter method while maintaining aspect ratio
                    if (frame.width, frame.height) != (target_width, target_height):
                        target_aspect_ratio = float(target_width) / float(target_height)
                        frame_aspect_ratio = float(frame.width) / float(frame.height)
                        if target_aspect_ratio < frame_aspect_ratio:
                            # We will need to crop the width below, so resize to match the target_height
                            h = target_height
                            w = int((target_height * frame.width / frame.height) / 2) * 2  # force divisible by 2
                        else:
                            # We will need to crop the height below, so resize to match the target_width
                            w = target_width
                            h = int((target_width * frame.height / frame.width) / 2) * 2  # force divisible by 2

                        frame = reformatter.reformat(frame, format='rgba', width=w, height=h)

                    image = frame.to_image()
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    width, height = image.size

                    if (width, height) != (target_width, target_height):
                        # Crop to the center to match target dimensions
                        start_x = width // 2 - target_width // 2
                        start_y = height // 2 - target_height // 2
                        image = image.crop((start_x, start_y, start_x + target_width, start_y + target_height))

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
