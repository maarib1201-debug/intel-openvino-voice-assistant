"""Coherent Vulkan runtime loader and programmatic offload verification for llama.cpp on Windows.

Ensures matching Python package and native DLL set without mutating the baseline CPU virtual environment.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VulkanVerification:
    gpu_verified: bool
    actual_backend: str
    gpu_device: str | None
    layers_offloaded: int
    total_layers: int
    vulkan_layer_count: int
    cpu_layer_count: int
    has_arc: bool
    has_vulkan_backend: bool
    diagnostic_message: str


class FdCapture:
    """Captures OS-level stderr (fd 2) output written directly by native llama.cpp / ggml."""

    def __init__(self) -> None:
        self.path = ""
        self._file: Any = None
        self._saved_fd = -1
        self.text = ""

    def __enter__(self) -> "FdCapture":
        self.path = tempfile.mktemp(prefix="v2v_vk_diag_")
        self._file = open(self.path, "w+", encoding="utf-8", errors="replace")
        self._saved_fd = os.dup(2)
        os.dup2(self._file.fileno(), 2)
        return self

    def __exit__(self, *args: Any) -> bool:
        try:
            if self._saved_fd >= 0:
                os.dup2(self._saved_fd, 2)
                os.close(self._saved_fd)
        except OSError:
            pass
        try:
            if self._file:
                self._file.flush()
                self._file.seek(0)
                self.text = self._file.read()
                self._file.close()
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass
        return False


def get_vulkan_runtime_path() -> Path | None:
    """Find a verified coherent Vulkan llama_cpp package directory."""
    candidates = []
    env_dir = os.environ.get("V2V_VULKAN_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # Vendored models/vulkan_runtime
    candidates.append(ROOT / "models" / "vulkan_runtime")
    # Proven reference environment
    candidates.append(ROOT / ".venv-vulkan" / "Lib" / "site-packages")

    for cand in candidates:
        if cand.is_dir():
            lib_dir = cand / "llama_cpp" / "lib"
            vk_dll = lib_dir / "ggml-vulkan.dll"
            llama_dll = lib_dir / "llama.dll"
            if vk_dll.exists() and llama_dll.exists():
                return cand.resolve()
    return None


def bootstrap_vulkan_runtime() -> bool:
    """Configures sys.path and DLL search paths for the coherent Vulkan runtime."""
    vk_path = get_vulkan_runtime_path()
    if vk_path is None:
        return False

    lib_dir = (vk_path / "llama_cpp" / "lib").resolve()
    str_lib = str(lib_dir)

    # Ensure DLL search path on Windows
    if hasattr(os, "add_dll_directory") and lib_dir.exists():
        try:
            os.add_dll_directory(str_lib)
        except OSError:
            pass

    # Ensure sys.path prioritizes the coherent Vulkan package
    str_path = str(vk_path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

    # Add to os.environ["PATH"] for legacy DLL resolution
    if str_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str_lib + os.pathsep + os.environ.get("PATH", "")

    return True


def parse_vulkan_diagnostics(log_text: str) -> VulkanVerification:
    """Parses native startup logs to verify real GPU offload vs CPU fallback."""
    lines = (log_text or "").splitlines()
    vulkan_layers = [ln for ln in lines if "assigned to device Vulkan" in ln]
    cpu_layers = [ln for ln in lines if "assigned to device CPU" in ln]
    has_arc = any("Arc" in ln and "140V" in ln for ln in lines)
    has_vulkan_backend = any("ggml_vulkan:" in ln or "using device Vulkan" in ln for ln in lines)

    offloaded_match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log_text, re.I)
    if offloaded_match:
        layers_offloaded = int(offloaded_match.group(1))
        total_layers = int(offloaded_match.group(2))
    else:
        layers_offloaded = len(vulkan_layers)
        total_layers = 25

    # Verification requires Arc 140V, Vulkan backend, full (or near-full) offload, and no CPU layers
    gpu_verified = bool(
        has_vulkan_backend
        and has_arc
        and layers_offloaded >= 24
        and len(cpu_layers) == 0
    )

    if gpu_verified:
        device_name = "Intel Arc 140V"
        msg = f"Verified Arc Vulkan offload: {layers_offloaded}/{total_layers} layers on Vulkan0 (0 CPU layers)"
        actual = "vulkan"
    else:
        device_name = "Intel Arc 140V" if has_arc else None
        if len(cpu_layers) > 0 and layers_offloaded == 0:
            msg = f"CPU execution: all {len(cpu_layers)} layers assigned to CPU"
            actual = "cpu"
        elif not has_vulkan_backend:
            msg = "Vulkan backend not detected in llama.cpp startup logs"
            actual = "cpu"
        elif not has_arc:
            msg = "Intel Arc 140V not detected by Vulkan backend"
            actual = "vulkan-unverified"
        else:
            msg = f"Partial offload: {layers_offloaded}/{total_layers} layers offloaded, {len(cpu_layers)} on CPU"
            actual = "vulkan-partial"

    return VulkanVerification(
        gpu_verified=gpu_verified,
        actual_backend=actual,
        gpu_device=device_name,
        layers_offloaded=layers_offloaded,
        total_layers=total_layers,
        vulkan_layer_count=len(vulkan_layers),
        cpu_layer_count=len(cpu_layers),
        has_arc=has_arc,
        has_vulkan_backend=has_vulkan_backend,
        diagnostic_message=msg,
    )
