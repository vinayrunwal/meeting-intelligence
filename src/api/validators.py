"""
Meeting Intelligence System — API Validators
============================================
Validates incoming audio files before processing.
"""

import mimetypes
from pathlib import Path
from typing import Optional

from werkzeug.datastructures import FileStorage

from config.settings import settings
from src.api.error_handlers import ValidationError


def validate_upload(file: Optional[FileStorage]) -> None:
    """
    Validate an uploaded file from a Flask request.
    
    Parameters
    ----------
    file : FileStorage
        The file object from request.files
        
    Raises
    ------
    ValidationError
        If the file is missing, invalid type, or too large.
    """
    if not file or not file.filename:
        raise ValidationError("No file provided in the request")

    # 1. Validate Extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.audio.allowed_extensions:
        raise ValidationError(
            f"File extension '{ext}' not allowed. "
            f"Supported: {', '.join(settings.audio.allowed_extensions)}"
        )

    # 2. Validate MIME Type (basic check)
    mimetype = file.mimetype
    if not mimetype:
        # Try to guess
        mimetype, _ = mimetypes.guess_type(file.filename)
        
    if mimetype not in settings.audio.allowed_mime_types and not mimetype.startswith("audio/"):
        raise ValidationError(f"Invalid MIME type: {mimetype}. Must be an audio file.")

    # 3. Validate Size (if stream is seekable, though usually handled by NGINX or Flask Max-Content-Length)
    file.seek(0, 2)  # seek to end
    size_bytes = file.tell()
    file.seek(0)     # reset back to start
    
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.audio.max_upload_size_mb:
        raise ValidationError(
            f"File too large: {size_mb:.1f}MB exceeds limit of {settings.audio.max_upload_size_mb}MB"
        )
