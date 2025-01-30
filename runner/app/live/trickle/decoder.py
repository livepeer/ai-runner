import av
import logging
import sys
from PIL import Image

from .frame import InputFrame

def decode_av(pipe_input, frame_callback, put_metadata):
    """
    Reads from a pipe (or file-like object). If both audio and video
    streams exist, for each decoded video frame, we gather all audio frames
    whose PTS is <= the video frame's PTS, then call `frame_callback`.

    Cases handled:
      - No audio (video only).
      - No video (audio only).
      - Both audio and video.

    :param pipe_input: File path, 'pipe:', sys.stdin, or another file-like object.
    :param frame_callback A function that accepts an InputFrame object
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
            # usually unreliable, especially with webrtc
            "framerate": video_stream.codec_context.framerate,
        }

    if video_metadata is None and audio_metadata is None:
        logging.error("No audio or video streams found in the input.")
        container.close()
        return

    metadata = { 'video': video_metadata, 'audio': audio_metadata }
    logging.info(f"Metadata: {metadata}")
    put_metadata(metadata)

    reformatter = av.video.reformatter.VideoReformatter()
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

                    w = 512
                    h = int((512 * frame.width / frame.height) / 2) * 2 # force divisible by 2
                    if frame.height > frame.width:
                        h = 512
                        w = int((512 * frame.height / frame.width) / 2) * 2
                    frame = reformatter.reformat(frame, format='rgba', width=w, height=h)
                    avframe = InputFrame.from_av_video(frame)
                    frame_callback(avframe)
                    continue

    except Exception as e:
        logging.error(f"Exception while decoding: {e}")

    finally:
        container.close()

    logging.info("Decoder stopped")
