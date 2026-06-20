import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.utils.job_store import JobStore
from src.models.schemas import JobState, AudioMetadata

def test_job_store_lifecycle():
    """Test job store creation and updates."""
    store = JobStore(max_jobs=10)
    job_id = store.create_job("test.wav")
    
    assert job_id is not None
    
    status = store.get_status(job_id)
    assert status.state == JobState.QUEUED
    
    # Update progress
    store.update_progress(job_id, JobState.PROCESSING, "ingestion", 10.0, "Testing")
    status = store.get_status(job_id)
    assert status.state == JobState.PROCESSING
    assert status.progress.percent == 10.0
    
    # Test events
    events = store.get_events(job_id)
    assert len(events) == 1
    assert events[0]["type"] == "progress"

def test_audio_metadata_validation():
    """Test Pydantic model validation for audio metadata."""
    meta = AudioMetadata(
        filename="test.wav",
        duration_seconds=10.5,
        sample_rate=16000,
        channels=1,
        file_size_bytes=1024,
        format="wav"
    )
    assert meta.filename == "test.wav"
    
    # Negative duration should fail
    with pytest.raises(ValueError):
        AudioMetadata(
            filename="test.wav",
            duration_seconds=-5.0,
            sample_rate=16000,
            channels=1,
            file_size_bytes=1024,
            format="wav"
        )


def test_diarizer_keeps_torch_load_compatibility_during_device_move(monkeypatch):
    """PyAnnote can lazily load weights while moving the pipeline to a device."""
    torch = pytest.importorskip("torch")
    from src.pipeline.diarization import SpeakerDiarizer

    load_kwargs = []

    def fake_torch_load(*args, **kwargs):
        load_kwargs.append(kwargs.copy())
        return object()

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            torch.load("fake-checkpoint.pt", weights_only=True)

    pyannote = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePipeline
    pyannote_audio_core = ModuleType("pyannote.audio.core")
    pyannote_model = ModuleType("pyannote.audio.core.model")
    pyannote_pipeline = ModuleType("pyannote.audio.core.pipeline")
    pyannote_pipelines = ModuleType("pyannote.audio.pipelines")
    pyannote_speaker_verification = ModuleType(
        "pyannote.audio.pipelines.speaker_verification"
    )
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda *args, **kwargs: "downloaded"

    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", pyannote_audio_core)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", pyannote_model)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", pyannote_pipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", pyannote_pipelines)
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio.pipelines.speaker_verification",
        pyannote_speaker_verification,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setattr(torch, "load", fake_torch_load)

    diarizer = SpeakerDiarizer()
    diarizer.config = SimpleNamespace(
        hf_token="token",
        model_name="pyannote/speaker-diarization-3.1",
        device="cpu",
    )
    diarizer.load_model()

    assert load_kwargs == [{"weights_only": False}]
    assert torch.load is fake_torch_load
