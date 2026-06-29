from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_PATH = REPO_ROOT / "live" / "streamdiffusion" / "pipeline" / "pipeline.py"
PROCESS_PATH = REPO_ROOT / "runner" / "src" / "runner" / "live" / "process" / "process.py"
GUARDIAN_PATH = (
    REPO_ROOT / "runner" / "src" / "runner" / "live" / "process" / "process_guardian.py"
)
TRICKLE_PATH = (
    REPO_ROOT
    / "runner"
    / "src"
    / "runner"
    / "live"
    / "streamer"
    / "protocol"
    / "trickle.py"
)
ENCODER_PATH = REPO_ROOT / "runner" / "src" / "runner" / "live" / "trickle" / "encoder.py"
INTERFACE_PATH = (
    REPO_ROOT / "runner" / "src" / "runner" / "live" / "pipelines" / "interface.py"
)


def _function_source(text: str, signature: str, next_marker: str) -> str:
    start = text.index(signature)
    end = text.index(next_marker, start)
    return text[start:end]


def test_streamdiffusion_cleanup_stream_drains_queue_in_place():
    text = PIPELINE_PATH.read_text()
    cleanup_source = _function_source(
        text,
        "    async def cleanup_stream(self):",
        "    @classmethod",
    )

    assert "self.frame_queue =" not in cleanup_source
    assert "_clear_asyncio_queue(self.frame_queue)" in cleanup_source
    assert "gc.collect" in text
    assert "malloc_trim" in text
    assert "LIVE_PIPELINE_STREAM_CLEANUP_CUDA_CACHE" in text


def test_process_cleanup_message_reaches_child_pipeline():
    text = PROCESS_PATH.read_text()

    assert "STREAM_CLEANUP_CONTROL_KEY" in text
    assert "cleanup_stream: bool = False" in text
    assert "message[STREAM_CLEANUP_CONTROL_KEY] = True" in text
    assert "await pipeline.cleanup_stream()" in text
    assert "PipelineProcess: stream cleanup requested" in text


def test_guardian_requests_cleanup_when_stream_goes_offline():
    text = GUARDIAN_PATH.read_text()

    assert 'self.process.reset_stream("", "", "", cleanup_stream=True)' in text
    assert "self.process.update_params(self.pipeline_spec.initial_params)" in text


def test_pipeline_interface_exposes_cleanup_hook():
    text = INTERFACE_PATH.read_text()

    assert "async def cleanup_stream(self):" in text
    assert "while keeping the warmed pipeline alive" in text


def test_publish_queue_is_bounded_and_drops_stale_frames():
    text = TRICKLE_PATH.read_text()

    assert "DEFAULT_PUBLISH_QUEUE_MAXSIZE = 1" in text
    assert "LIVE_TRICKLE_PUBLISH_QUEUE_MAXSIZE" in text
    assert "queue.Queue[OutputFrame](maxsize=_publish_queue_maxsize())" in text
    assert "def _put_drop_oldest" in text
    assert "old = _queue.get_nowait()" in text
    assert "del old" in text
    assert "self.publish_queue_dropped" in text


def test_trickle_egress_releases_output_frame_reference():
    text = TRICKLE_PATH.read_text()
    egress_source = _function_source(
        text,
        "    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):",
        "    async def emit_monitoring_event",
    )

    assert "_put_drop_oldest(publish_queue, frame)" in egress_source
    assert "finally:" in egress_source
    assert "del frame" in egress_source


def test_encoder_releases_video_frame_conversion_locals():
    text = ENCODER_PATH.read_text()

    assert "tensor = None" in text
    assert "image_np = None" in text
    assert "encoded_packets = None" in text
    assert "finally:" in text
    assert "del avframe" in text
