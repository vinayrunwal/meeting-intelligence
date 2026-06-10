"""
Meeting Intelligence System — Centralized Configuration
========================================================
All settings are loaded from environment variables with sensible defaults.
Use a .env file for local development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for _dir in (UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, MODEL_CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with a fallback."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WhisperConfig:
    """Configuration for Whisper ASR engine."""
    model_size: str = _env("WHISPER_MODEL_SIZE", "large-v3")
    device: str = _env("DEVICE", "auto")
    compute_type: str = _env("WHISPER_COMPUTE_TYPE", "float16")
    beam_size: int = _env_int("WHISPER_BEAM_SIZE", 5)
    language: Optional[str] = _env("WHISPER_LANGUAGE", "") or None
    task: str = "transcribe"  # 'transcribe' keeps original language
    vad_filter: bool = True
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None


@dataclass(frozen=True)
class DiarizationConfig:
    """Configuration for PyAnnote speaker diarization."""
    model_name: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = _env("HF_TOKEN", "")
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    device: str = _env("DEVICE", "auto")


@dataclass(frozen=True)
class EmotionConfig:
    """Configuration for emotion classification."""
    model_name: str = _env("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    device: str = _env("DEVICE", "auto")
    batch_size: int = _env_int("EMOTION_BATCH_SIZE", 16)
    max_length: int = 512


@dataclass(frozen=True)
class IntentConfig:
    """Configuration for intent classification (zero-shot)."""
    model_name: str = _env("INTENT_MODEL", "facebook/bart-large-mnli")
    device: str = _env("DEVICE", "auto")
    batch_size: int = _env_int("INTENT_BATCH_SIZE", 8)
    candidate_labels: tuple[str, ...] = (
        "question",
        "statement",
        "suggestion",
        "action_request",
        "agreement",
        "disagreement",
        "clarification",
        "greeting",
    )


@dataclass(frozen=True)
class DebriefConfig:
    """Configuration for FLAN-T5 meeting debrief generation."""
    model_name: str = _env("FLAN_T5_MODEL", "google/flan-t5-large")
    device: str = _env("DEVICE", "auto")
    max_input_tokens: int = _env_int("DEBRIEF_MAX_INPUT", 768)
    max_output_tokens: int = _env_int("DEBRIEF_MAX_OUTPUT", 512)
    chunk_overlap_sentences: int = 3
    temperature: float = 0.3
    num_beams: int = 4


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing constraints."""
    target_sample_rate: int = 16_000
    target_channels: int = 1  # mono
    max_upload_size_mb: int = _env_int("MAX_UPLOAD_SIZE_MB", 500)
    max_duration_minutes: int = _env_int("MAX_AUDIO_DURATION_MIN", 120)
    allowed_extensions: tuple[str, ...] = (
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus",
    )
    allowed_mime_types: tuple[str, ...] = (
        "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/ogg", "audio/mp4", "audio/x-m4a",
        "audio/aac", "audio/opus", "audio/x-ms-wma",
    )


@dataclass(frozen=True)
class APIConfig:
    """Flask API configuration."""
    host: str = _env("API_HOST", "0.0.0.0")
    port: int = _env_int("API_PORT", 8000)
    debug: bool = _env_bool("FLASK_DEBUG", False)
    secret_key: str = _env("SECRET_KEY", "dev-secret-change-me")
    max_concurrent_jobs: int = _env_int("MAX_CONCURRENT_JOBS", 3)
    cors_origins: str = _env("CORS_ORIGINS", "*")


@dataclass(frozen=True)
class LogConfig:
    """Logging configuration."""
    level: str = _env("LOG_LEVEL", "INFO")
    format: str = _env("LOG_FORMAT", "json")  # 'json' or 'console'
    log_dir: Path = LOG_DIR


# ---------------------------------------------------------------------------
# Master settings object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Master configuration container — single source of truth."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    debrief: DebriefConfig = field(default_factory=DebriefConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)


# Module-level singleton
settings = Settings()
