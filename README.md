# Real-Time Multilingual Meeting Intelligence System

A comprehensive ML system that ingests audio, transcribes it with Whisper, diarizes speakers with PyAnnote, classifies emotion and intent using RoBERTa, and generates a structured meeting debrief with FLAN-T5.

## Features
- **Multilingual ASR:** High-accuracy transcription using `faster-whisper`.
- **Speaker Diarization:** Precise speaker tracking with `pyannote.audio`.
- **Emotion & Intent:** Utterance-level classification.
- **Meeting Debrief:** Auto-generated summaries, action items, and decisions.
- **REST API:** Flask API with SSE streaming for real-time progress.

## Quickstart

1. **Install Dependencies** (Requires Python 3.10+, CUDA recommended)
   ```bash
   pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and ADD YOUR HF_TOKEN (Required for PyAnnote)
   ```

3. **Download Models** (Optional, avoids downloading during first run)
   ```bash
   python scripts/download_models.py --all
   ```

4. **Run API Server**
   ```bash
   python src/api/app.py
   ```

## API Usage

**1. Upload Audio**
```bash
curl -X POST -F "file=@meeting.wav" http://localhost:8000/api/v1/upload
```
Returns a `job_id`.

**2. Stream Progress**
```bash
curl http://localhost:8000/api/v1/stream/<job_id>
```

**3. Get Results**
```bash
curl http://localhost:8000/api/v1/result/<job_id>
```

## Deployment
Check the `deploy/` directory for Gunicorn, NGINX, systemd, and Docker configurations.
