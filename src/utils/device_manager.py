"""
Meeting Intelligence System — Device Manager
==============================================
Intelligent GPU / CPU / MPS device detection with memory-aware model loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""
    device: torch.device
    name: str
    is_gpu: bool
    total_memory_gb: Optional[float] = None
    available_memory_gb: Optional[float] = None


class DeviceManager:
    """
    Manages compute device selection and memory monitoring.

    Usage::

        dm = DeviceManager()
        device = dm.get_device()
        dm.check_memory(required_gb=4.0)
    """

    _instance: Optional[DeviceManager] = None
    _device_info: Optional[DeviceInfo] = None

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_device(self, preference: str = "auto") -> torch.device:
        """
        Select the best available compute device.

        Parameters
        ----------
        preference : str
            One of 'auto', 'cuda', 'mps', 'cpu'.
            'auto' probes in order: CUDA → MPS → CPU.

        Returns
        -------
        torch.device
        """
        if preference == "auto":
            device = self._auto_detect()
        elif preference == "cuda":
            device = self._try_cuda()
        elif preference == "mps":
            device = self._try_mps()
        else:
            device = torch.device("cpu")

        self._device_info = self._build_device_info(device)
        logger.info(
            "Device selected: %s (GPU=%s, Memory=%.1f GB)",
            self._device_info.name,
            self._device_info.is_gpu,
            self._device_info.total_memory_gb or 0,
        )
        return device

    def get_device_info(self) -> DeviceInfo:
        """Return cached device information."""
        if self._device_info is None:
            self.get_device()
        return self._device_info  # type: ignore[return-value]

    def get_compute_type(self, preference: str = "auto") -> str:
        """
        Determine the best compute type for CTranslate2 / faster-whisper.

        Returns one of: 'float16', 'int8_float16', 'int8', 'float32'
        """
        info = self.get_device_info()
        if not info.is_gpu:
            return "int8"
        if preference != "auto":
            return preference
        # Use float16 on GPUs with >= 6 GB VRAM, int8_float16 otherwise
        if info.total_memory_gb and info.total_memory_gb >= 6.0:
            return "float16"
        return "int8_float16"

    def check_memory(self, required_gb: float) -> bool:
        """
        Check if enough GPU memory is available for a model.

        Parameters
        ----------
        required_gb : float
            Required GPU memory in gigabytes.

        Returns
        -------
        bool
            True if sufficient memory is available.
        """
        info = self.get_device_info()
        if not info.is_gpu:
            logger.warning("Running on CPU — memory check skipped")
            return True

        available = info.available_memory_gb or 0.0
        if available < required_gb:
            logger.warning(
                "Insufficient GPU memory: %.1f GB available, %.1f GB required",
                available,
                required_gb,
            )
            return False

        logger.info(
            "Memory check passed: %.1f GB available, %.1f GB required",
            available,
            required_gb,
        )
        return True

    def get_gpu_memory_usage(self) -> dict[str, float]:
        """Return current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3),
            "total": torch.cuda.get_device_properties(0).total_mem / (1024**3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect(self) -> torch.device:
        """Probe devices in priority order: CUDA → MPS → CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _try_cuda(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")

    def _try_mps(self) -> torch.device:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available — falling back to CPU")
        return torch.device("cpu")

    def _build_device_info(self, device: torch.device) -> DeviceInfo:
        """Build a DeviceInfo from the selected device."""
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_mem / (1024**3)
            free_gb = (
                props.total_mem - torch.cuda.memory_allocated()
            ) / (1024**3)
            return DeviceInfo(
                device=device,
                name=props.name,
                is_gpu=True,
                total_memory_gb=round(total_gb, 2),
                available_memory_gb=round(free_gb, 2),
            )
        elif device.type == "mps":
            return DeviceInfo(
                device=device,
                name="Apple Silicon (MPS)",
                is_gpu=True,
                total_memory_gb=None,
                available_memory_gb=None,
            )
        else:
            return DeviceInfo(
                device=device,
                name="CPU",
                is_gpu=False,
            )


# Module-level singleton for convenience
device_manager = DeviceManager()
