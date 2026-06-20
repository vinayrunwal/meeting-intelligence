"""
Meeting Intelligence System — Audio Utilities
===============================================
Audio format conversion, validation, and preprocessing.
Requires ffmpeg installed on the system.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass


def check_ffmpeg() -> bool:
    """Verify that ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_audio_info(file_path: Path) -> dict:
    """
    Extract audio metadata using ffprobe.

    Returns
    -------
    dict
        Keys: duration, sample_rate, channels, codec, format_name, file_size
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise AudioProcessingError(f"Audio file not found: {file_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"ffprobe failed: {e.stderr}") from e
    except FileNotFoundError:
        raise AudioProcessingError(
            "ffprobe not found. Install ffmpeg: sudo apt install ffmpeg"
        )

    import json
    probe = json.loads(result.stdout)

    # Find the audio stream
    audio_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise AudioProcessingError("No audio stream found in file")

    fmt = probe.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0)),
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
        "codec": audio_stream.get("codec_name", "unknown"),
        "format_name": fmt.get("format_name", "unknown"),
        "file_size": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
    }


def convert_to_wav(
    input_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """
    Convert any audio file to 16kHz mono WAV using ffmpeg.

    Parameters
    ----------
    input_path : Path
        Source audio file (any format ffmpeg supports).
    output_path : Path, optional
        Destination WAV path. Auto-generated if not provided.
    sample_rate : int
        Target sample rate (default: 16000 Hz).
    channels : int
        Target channel count (default: 1 / mono).

    Returns
    -------
    Path
        Path to the converted WAV file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise AudioProcessingError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".converted.wav")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                       # Overwrite output
        "-i", str(input_path),      # Input file
        "-vn",                      # No video
        "-acodec", "pcm_s16le",     # 16-bit PCM
        "-ar", str(sample_rate),    # Sample rate
        "-ac", str(channels),       # Channel count
        "-loglevel", "warning",
        str(output_path),
    ]

    logger.info(
        "Converting audio: %s → %s (sr=%d, ch=%d)",
        input_path.name, output_path.name, sample_rate, channels,
    )

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"ffmpeg conversion failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise AudioProcessingError("Audio conversion timed out (>5 minutes)")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioProcessingError("Conversion produced an empty file")

    logger.info("Conversion complete: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1e6)
    return output_path


def load_audio_array(
    file_path: Path,
    target_sr: int = 16_000,
) -> tuple[np.ndarray, int]:
    """
    Load audio as a numpy array, resampling if necessary.

    Parameters
    ----------
    file_path : Path
        Path to a WAV file (preferably pre-converted).
    target_sr : int
        Target sample rate.

    Returns
    -------
    tuple[np.ndarray, int]
        (audio_data, sample_rate) — audio is float32 in [-1, 1].
    """
    try:
        audio, sr = sf.read(str(file_path), dtype="float32")
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {e}") from e

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if sample rate doesn't match
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            logger.warning(
                "librosa not installed — cannot resample from %d to %d Hz. "
                "Pre-convert audio with convert_to_wav() instead.",
                sr, target_sr,
            )

    return audio, sr


def trim_silence(
    audio: np.ndarray,
    sr: int = 16_000,
    top_db: int = 30,
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal as float32.
    sr : int
        Sample rate.
    top_db : int
        Threshold in dB below peak to consider as silence.

    Returns
    -------
    np.ndarray
        Trimmed audio.
    """
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        duration_removed = (len(audio) - len(trimmed)) / sr
        if duration_removed > 0.1:
            logger.info("Trimmed %.1f seconds of silence", duration_removed)
        return trimmed
    except ImportError:
        logger.warning("librosa not available — skipping silence trimming")
        return audio


def split_audio_chunks(
    audio: np.ndarray,
    sr: int = 16_000,
    chunk_duration_seconds: float = 300.0,  # 5 minutes
    overlap_seconds: float = 10.0,
) -> list[tuple[np.ndarray, float, float]]:
    """
    Split long audio into overlapping chunks for processing.

    Parameters
    ----------
    audio : np.ndarray
        Full audio array.
    sr : int
        Sample rate.
    chunk_duration_seconds : float
        Duration of each chunk in seconds.
    overlap_seconds : float
        Overlap between consecutive chunks in seconds.

    Returns
    -------
    list[tuple[np.ndarray, float, float]]
        List of (chunk_audio, start_time, end_time) tuples.
    """
    chunk_samples = int(chunk_duration_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    stride = chunk_samples - overlap_samples
    total_samples = len(audio)

    chunks = []
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        start_time = start / sr
        end_time = end / sr
        chunks.append((chunk, start_time, end_time))
        start += stride

        if end >= total_samples:
            break

    logger.info(
        "Split audio into %d chunks (%.0fs each, %.0fs overlap)",
        len(chunks), chunk_duration_seconds, overlap_seconds,
    )
    return chunks


def save_audio(
    audio: np.ndarray,
    file_path: Path,
    sr: int = 16_000,
) -> Path:
    """Save a numpy audio array to a WAV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), audio, sr, subtype="PCM_16")
    logger.debug("Saved audio: %s (%.1f seconds)", file_path.name, len(audio) / sr)
    return file_path
