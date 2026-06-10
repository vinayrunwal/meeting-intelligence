"""
Meeting Intelligence System — Gunicorn Configuration
====================================================
Gunicorn config for running the Flask API in production with GPU workloads.
"""

import multiprocessing

# Bind to unix socket for NGINX or port for testing
bind = "0.0.0.0:8000"

# Gevent is required for Server-Sent Events (SSE) streaming
worker_class = "gevent"

# For GPU workloads, keep worker count low to avoid VRAM exhaustion.
# 1 worker with gevent handles multiple I/O bound connections efficiently.
workers = 1

# Max simultaneous connections per worker
worker_connections = 1000

# Long timeout because ML inference can take minutes for large files
timeout = 600

# Restart workers periodically to prevent memory leaks over time
max_requests = 100
max_requests_jitter = 10

# Preload app so imports happen before forking (saves RAM)
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

def post_fork(server, worker):
    """Called after a worker is forked."""
    # Monkey-patch gevent for the worker
    from gevent import monkey
    monkey.patch_all()
    
    # Optional: configure PyTorch thread count per worker
    import torch
    torch.set_num_threads(1)
    
    server.log.info("Worker spawned (pid: %s), monkey-patched for gevent", worker.pid)
