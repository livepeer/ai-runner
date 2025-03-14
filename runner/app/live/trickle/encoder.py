import asyncio
import av
import time
import datetime
import logging
import os
from typing import Optional
from fractions import Fraction

from .frame import VideoOutput, AudioOutput, InputFrame

# use mpegts default time base
OUT_TIME_BASE=Fraction(1, 90_000)
GOP_SECS=3

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
        output_audio_stream.sample_rate = audio_meta['sample_rate'] # TODO take from inference if not passthru
        output_audio_stream.layout = 'mono'
        # Optional: set other encoding parameters, e.g.:
        # output_audio_stream.bit_rate = 128_000

    # Now read packets from the input, decode, then re-encode, and mux.
    start = datetime.datetime.now()
    last_kf = None
    video_received = False
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
            frame = av.video.frame.VideoFrame.from_image(avframe.image)
            frame.pts = rescale_ts(avframe.timestamp, avframe.time_base, output_video_stream.codec_context.time_base)
            frame.time_base = output_video_stream.codec_context.time_base
            current = avframe.timestamp * avframe.time_base
            if not last_kf or float(current - last_kf) >= GOP_SECS:
                frame.pict_type = av.video.frame.PictureType.I
                last_kf = current
            encoded_packets = output_video_stream.encode(frame)
            for ep in encoded_packets:
                output_container.mux(ep)
            video_received = True
            continue

        if isinstance(avframe, AudioOutput):
            if not output_audio_stream:
                # received audio but no audio output, so drop
                continue
            if output_video_stream and not video_received:
                # Wait until we receive video to start sending audio
                # because video could be extremely delayed and we don't
                # want to send out audio-only segments since that confuses
                # downstream tools
                continue
            for af in avframe.frames:
                af.log_timestamps["frame_end"] = time.time()
                log_frame_timestamps("Audio", af)
                frame = av.audio.frame.AudioFrame.from_ndarray(af.samples, format=af.format, layout=af.layout)
                frame.sample_rate = af.rate
                frame.pts = rescale_ts(af.timestamp, af.time_base, output_audio_stream.codec_context.time_base)
                frame.time_base = output_audio_stream.codec_context.time_base
                encoded_packets = output_audio_stream.encode(frame)
                for ep in encoded_packets:
                    output_container.mux(ep)
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
