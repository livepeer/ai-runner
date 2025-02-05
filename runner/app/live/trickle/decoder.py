import av
import logging
import sys
from PIL import Image

from .frame import InputFrame

MAX_FRAMERATE=24

def decode_av(pipe_input, frame_callback, put_metadata):
    """
    Reads from a pipe (or file-like object).

    :param pipe_input: File path, 'pipe:', sys.stdin, or another file-like object.
    :param frame_callback: A function that accepts an InputFrame object
    :param put_metadata: A function that accepts audio/video metadata
    """
    container = av.open(pipe_input)

    # Locate the first video and first audio stream (if they exist)
    video_stream = None
    audio_stream = None
    for s in container.streams:
        if s.type == 'video' and video_stream is None:
            video_stream = s
        elif s.type == 'audio' and audio_stream is None:
            audio_stream = s

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
        }

    if video_metadata is None and audio_metadata is None:
        logging.error("No audio or video streams found in the input.")
        container.close()
        return

    metadata = { 'video': video_metadata, 'audio': audio_metadata }
    logging.info(f"Metadata: {metadata}")
    put_metadata(metadata)

    reformatter = av.video.reformatter.VideoReformatter()
    frame_interval = 1.0 / MAX_FRAMERATE
    next_pts_time = 0.0
    try:
        for packet in container.demux():
            if packet.dts is None:
                continue

            if audio_stream and packet.stream == audio_stream:
                # Decode audio frames
                for aframe in packet.decode():
                    if aframe.pts is None:
                        continue

                    avframe = InputFrame.from_av_audio(aframe)
                    frame_callback(avframe)
                    continue

            elif video_stream and packet.stream == video_stream:
                # Decode video frames
                for frame in packet.decode():
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

                    w = 512
                    h = int((512 * frame.height / frame.width) / 2) * 2 # force divisible by 2
                    if frame.height > frame.width:
                        h = 512
                        w = int((512 * frame.width / frame.height) / 2) * 2
                    frame = reformatter.reformat(frame, format='rgba', width=w, height=h)
                    avframe = InputFrame.from_av_video(frame)
                    frame_callback(avframe)
                    continue

    except Exception as e:
        logging.error(f"Exception while decoding: {e}")

    finally:
        container.close()

    logging.info("Decoder stopped")
