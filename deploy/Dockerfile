# Use official NVIDIA PyTorch base image for CUDA support
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HUGGINGFACE_HUB_CACHE=/app/models

# Install system dependencies (ffmpeg is required for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models to bake them into the image (Optional but recommended)
# COPY config/ /app/config/
# COPY scripts/download_models.py /app/scripts/
# RUN python scripts/download_models.py --all

# Copy project code
COPY . .

# Expose API port
EXPOSE 8000

# Run with Gunicorn using gevent worker for SSE
CMD ["gunicorn", "--config", "deploy/gunicorn_config.py", "src.api.app:create_app()"]
