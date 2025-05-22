import av
import time
import datetime
import logging
import os
from typing import Optional
from fractions import Fraction
from collections import deque
from PIL import Image

from .frame import VideoOutput, AudioOutput, InputFrame

# use mpegts default time base
OUT_TIME_BASE=Fraction(1, 90_000)
GOP_SECS=2

# maximum buffer duration in seconds
AUDIO_BUFFER_SECS=2
# maximum buffer size in number of packets. Roughly 2s at 20ms / frame
AUDIO_BUFFER_LEN=100
# how many out-of-sync video packets to drop. Roughly 4s at 25fps
MAX_DROPPED_VIDEO=100

def encode_av(
    input_queue,
    output_callback,
    get_metadata,
    video_codec: Optional[str] ='libx264',
    audio_codec: Optional[str] ='libfdk_aac'
):
    logging.info("Starting encoder")

    decoded_metadata = get_metadata()
    if not decoded_metadata:
        logging.info("Metadata was empty, exiting encoder")
        return

    video_meta = decoded_metadata['video']
    audio_meta = decoded_metadata['audio']

    logging.info(f"Encoder recevied metadata video={video_meta is not None} audio={audio_meta is not None}")

    def custom_io_open(url: str, flags: int, options: dict):
        read_fd, write_fd = os.pipe()
        read_file  = os.fdopen(read_fd,  'rb', buffering=0)
        write_file = os.fdopen(write_fd, 'wb', buffering=0)
        output_callback(read_file, write_file, url)
        return write_file

    # Open the output container in write mode
    output_container = av.open("%d.ts", format='segment', mode='w', io_open=custom_io_open)

    # Create corresponding output streams if input streams exist
    output_video_stream = None
    output_audio_stream = None

    if video_meta and video_codec:
        # Add a new stream to the output using the desired video codec
        video_opts = { 'video_size':'512x512', 'bf':'0' }
        if video_codec == 'libx264':
            video_opts = video_opts | { 'preset':'superfast', 'tune':'zerolatency', 'forced-idr':'1' }
        output_video_stream = output_container.add_stream(video_codec, options=video_opts)
        output_video_stream.time_base = OUT_TIME_BASE

    if audio_meta and audio_codec:
        # Add a new stream to the output using the desired audio codec
        output_audio_stream = output_container.add_stream(audio_codec)
        output_audio_stream.time_base = OUT_TIME_BASE
        output_audio_stream.sample_rate = 48000
        output_audio_stream.layout = 'mono'
        # Optional: set other encoding parameters, e.g.:
        # output_audio_stream.bit_rate = 128_000

    # Now read packets from the input, decode, then re-encode, and mux.
    start = datetime.datetime.now()
    last_kf = None
    audio_received = False
    first_video_ts = None
    dropped_video = 0
    dropped_audio = 0
    audio_buffer = deque()

    while True:
        avframe = input_queue()
        if avframe is None:
            break

        if isinstance(avframe, VideoOutput):
            if not output_video_stream:
                # received video but no video output, so drop
                continue
            avframe.log_timestamps["frame_end"] = time.time()
            log_frame_timestamps("Video", avframe.frame)

            tensor = avframe.tensor.squeeze(0)
            image_np = (tensor * 255).byte().cpu().numpy()
            image = Image.fromarray(image_np)

            frame = av.video.frame.VideoFrame.from_image(image)
            frame.pts = rescale_ts(avframe.timestamp, avframe.time_base, output_video_stream.codec_context.time_base)
            frame.time_base = output_video_stream.codec_context.time_base
            current = avframe.timestamp * avframe.time_base

            # if there is pending audio, check whether video is too far behind
            if len(audio_buffer) > 0:
                audio_ts = audio_buffer[0].timestamp * audio_buffer[0].time_base
                delta = audio_ts - current
                if delta > AUDIO_BUFFER_SECS:
                    # video is too far behind audio, drop the frame to try and catch up
                    dropped_video += 1
                    if dropped_video > MAX_DROPPED_VIDEO:
                        # dropped too many frames, may be unlikely to catch up so stop the stream
                        logging.error(f"A/V is out of sync, exiting from video audio_ts={float(audio_ts)} video_ts={float(current)} delta={float(delta)}")
                        break
                    # drop the frame by skipping the rest of the following code
                    continue

            if not last_kf or float(current - last_kf) >= GOP_SECS:
                frame.pict_type = av.video.frame.PictureType.I
                last_kf = current
                if first_video_ts is None:
                    first_video_ts = current
            encoded_packets = output_video_stream.encode(frame)
            for ep in encoded_packets:
                output_container.mux(ep)
            continue

        if isinstance(avframe, AudioOutput):
            if not output_audio_stream:
                # received audio but no audio output, so drop
                continue
            if output_video_stream and first_video_ts is None:
                # Buffer up audio until we receive video, up to a point
                # because video could be extremely delayed and we don't
                # want to send out audio-only segments since that confuses
                # downstream tools

                # shortcut to assume len(audio_buffer) is at least 1
                if len(avframe.frames) <= 0:
                    continue
                for af in avframe.frames:
                    audio_buffer.append(af)
                while len(audio_buffer) > AUDIO_BUFFER_LEN:
                    af = audio_buffer.popleft()
                    dropped_audio += 1
                continue

            while len(audio_buffer) > 0:
                # if we hit this point then we have a video frame
                # Check whether audio is too far behind first video packet.
                af = audio_buffer[0]
                first_audio_ts = af.timestamp * af.time_base
                if first_video_ts - first_audio_ts > AUDIO_BUFFER_SECS:
                    audio_buffer.popleft()
                    dropped_audio += 1
                    continue

                # NB: video being too far behind audio is handled within the video code
                #     so we can drop video frames if necessary

                break

            if len(audio_buffer) > 0:
                first_ts = float(audio_buffer[0].timestamp * audio_buffer[0].time_base)
                last_ts = float(audio_buffer[-1].timestamp * audio_buffer[-1].time_base)
                avframe.frames[:0] = audio_buffer
                logging.info(f"Flushing {len(audio_buffer)} audio frames from={first_ts} to={last_ts} dropped_audio={dropped_audio} dropped_video={dropped_video}")
                audio_buffer.clear()

            av_broken = False
            for af in avframe.frames:
                af.log_timestamps["frame_end"] = time.time()
                log_frame_timestamps("Audio", af)
                if not audio_received and first_video_ts is not None:
                    first_audio_ts = af.timestamp * af.time_base
                    delta = first_audio_ts - first_video_ts
                    if abs(delta) > AUDIO_BUFFER_SECS:
                        # A/V is out of sync badly enough so exit for now
                        av_broken = True
                        logging.error(f"A/V is out of sync, exiting from audio audio_ts={float(first_audio_ts)} video_ts={float(first_video_ts)} delta={float(delta)}")
                        break
                    logging.info(f"Received first audio_ts={float(first_audio_ts)} video_ts={float(first_video_ts)} delta={float(delta)}")
                    audio_received = True
                frame = av.audio.frame.AudioFrame.from_ndarray(af.samples, format=af.format, layout=af.layout)
                frame.sample_rate = af.rate
                frame.pts = rescale_ts(af.timestamp, af.time_base, output_audio_stream.codec_context.time_base)
                frame.time_base = output_audio_stream.codec_context.time_base
                encoded_packets = output_audio_stream.encode(frame)
                for ep in encoded_packets:
                    output_container.mux(ep)
            if av_broken:
                # too far out of sync, so stop encoding
                break
            continue

        logging.warning(f"Unsupported output frame type {type(avframe)}")

    # After reading all packets, flush encoders
    logging.info("Stopping encoder")
    if output_video_stream:
        encoded_packets = output_video_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    if output_audio_stream:
        encoded_packets = output_audio_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    # Close the output container to finish writing
    output_container.close()

def rescale_ts(pts: int, orig_tb: Fraction, dest_tb: Fraction):
    if orig_tb == dest_tb:
        return pts
    return int(round(float((Fraction(pts) * orig_tb) / dest_tb)))


def log_frame_timestamps(frame_type: str, frame: InputFrame):
    ts = frame.log_timestamps

    def log_duration(start_key: str, end_key: str):
        if start_key in ts and end_key in ts:
            duration = ts[end_key] - ts[start_key]
            logging.debug(f"frame_type={frame_type} start_tag={start_key} end_tag={end_key} duration_s={duration}s")

    log_duration('frame_init', 'pre_process_frame')
    log_duration('pre_process_frame', 'post_process_frame')
    log_duration('post_process_frame', 'frame_end')
    log_duration('frame_init', 'frame_end')
