import av
from av.video.reformatter import VideoReformatter
from av.container import InputContainer
import time
import logging
from typing import cast, List
import copy
from fractions import Fraction
from .frame import InputFrame, AudioFrame, VideoFrame

MAX_FRAMERATE=500
TARGET_FRAMERATE=120

def decode_av(pipe_input, frame_callback, put_metadata):
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

    # Store frames for looping
    stored_video_frames: List[VideoFrame] = []
    stored_audio_frames: List[AudioFrame] = []
    max_time = 5.0  # seconds
    decoding = True
    try:
        for packet in container.demux():
            if not decoding:
                break
            if packet.dts is None:
                continue

            if audio_stream and packet.stream == audio_stream:
                # Decode audio frames
                for aframe in packet.decode():
                    aframe = cast(av.AudioFrame, aframe)
                    if aframe.pts is None:
                        continue
                    avframe = InputFrame.from_av_audio(aframe)
                    pts_time = aframe.pts * aframe.time_base
                    if pts_time <= max_time:
                        stored_audio_frames.append(copy.deepcopy(avframe))
                        avframe.log_timestamps["frame_init"] = time.time()
                        frame_callback(avframe)
                    else:
                        decoding = False
                        break

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

                    h = 512
                    w = int((512 * frame.width / frame.height) / 2) * 2 # force divisible by 2
                    if frame.height > frame.width:
                        w = 512
                        h = int((512 * frame.height / frame.width) / 2) * 2
                    frame = reformatter.reformat(frame, format='rgba', width=w, height=h)
                    avframe = InputFrame.from_av_video(frame)
                    if pts_time <= max_time:
                        stored_video_frames.append(copy.deepcopy(avframe))
                        avframe.log_timestamps["frame_init"] = time.time()
                        frame_callback(avframe)
                    else:
                        decoding = False
                        break
    except Exception as e:
        logging.error(f"Exception while decoding: {e}")
        raise # should be caught upstream

    finally:
        container.close()

    logging.info(f"Decoder: Collected {len(stored_video_frames)} video and {len(stored_audio_frames)} audio frames for looping.")

    # If no frames, just return
    if not stored_video_frames and not stored_audio_frames:
        logging.error("No frames collected for looping.")
        return

    # Loop and replay frames with updated timestamps
    start_time = time.time()
    video_idx = 0
    audio_idx = 0
    last_pts = stored_video_frames[-1].timestamp * stored_video_frames[-1].time_base
    frames_sent = 0
    frame_callback_latency = []
    sleep_latency = []
    TARGET_FRAME_DURATION_SECONDS = 1.0 / TARGET_FRAMERATE
    while True:
        now = time.time()
        elapsed = now - start_time
        # Video
        if stored_video_frames:
            vframe = copy.deepcopy(stored_video_frames[video_idx])
            # Update timestamp to simulate real-time
            pts_time = last_pts + Fraction(elapsed)
            vframe.timestamp = int(pts_time / vframe.time_base)
            vframe.log_timestamps = {"frame_init": now}
            frame_callback(vframe)
            frame_callback_latency.append(time.time() - now)
            video_idx = (video_idx + 1) % len(stored_video_frames)
            frames_sent += 1
            # Sleep to maintain frame rate
            target_completion_time_for_this_frame_iteration = start_time + ((frames_sent + 1) * TARGET_FRAME_DURATION_SECONDS)
            current_wall_time_before_sleep = time.time()
            sleep_duration_needed = target_completion_time_for_this_frame_iteration - current_wall_time_before_sleep

            actual_sleep_taken = 0.0
            if sleep_duration_needed > 0:
                time_before_sleep_call = time.time()
                time.sleep(sleep_duration_needed)
                actual_sleep_taken = time.time() - time_before_sleep_call

            sleep_latency.append(actual_sleep_taken)
        if frames_sent % 600 == 0:
            logging.info(f"Decoder: Sent {frames_sent} frames in {elapsed}s fps={frames_sent / elapsed:.2f}")
            logging.info(f"Decoder: Frame callback latency: {sum(frame_callback_latency) / len(frame_callback_latency):.5f}s")
            avg_sleep_duration = sum(sleep_latency) / len(sleep_latency) if sleep_latency else 0.0
            logging.info(f"Decoder: Avg Pacing Sleep: {avg_sleep_duration:.5f}s")
            frame_callback_latency = []
            sleep_latency = []
        # Audio (optional: you may want to sync audio more precisely)
        # if stored_audio_frames:
        #     aframe = copy.deepcopy(stored_audio_frames[audio_idx])
        #     aframe.timestamp = int((elapsed / aframe.time_base))
        #     aframe.log_timestamps = {"frame_init": now}
        #     frame_callback(aframe)
        #     audio_idx = (audio_idx + 1) % len(stored_audio_frames)
        # If neither, break
        if not stored_video_frames and not stored_audio_frames:
            break

    logging.info("Decoder stopped (looping mode)")
