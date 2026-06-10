import os
import pytest
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture
def mock_audio_path(tmp_path):
    """Fixture to provide a dummy audio file path."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00w\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return audio_file

@pytest.fixture
def mock_config(monkeypatch):
    """Fixture to set common environment variables for tests."""
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "50")
