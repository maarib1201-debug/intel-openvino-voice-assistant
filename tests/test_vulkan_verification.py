from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v2v.vulkan_runtime import parse_vulkan_diagnostics, VulkanVerification


SAMPLE_ARC_VULKAN_LOG = """
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB) (Intel Corporation) | uma: 1 | fp16: 1
llama_prepare_model_devices: using device Vulkan0 (Intel(R) Arc(TM) 140V GPU (16GB))
load_tensors: layer   0 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   1 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   2 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   3 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   4 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   5 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   6 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   7 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   8 assigned to device Vulkan0, is_swa = 0
load_tensors: layer   9 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  10 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  11 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  12 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  13 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  14 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  15 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  16 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  17 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  18 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  19 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  20 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  21 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  22 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  23 assigned to device Vulkan0, is_swa = 0
load_tensors: layer  24 assigned to device Vulkan0, is_swa = 0
load_tensors: offloaded 25/25 layers to GPU
"""

SAMPLE_CPU_FALLBACK_LOG = """
load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer   1 assigned to device CPU, is_swa = 0
load_tensors: layer   2 assigned to device CPU, is_swa = 0
load_tensors: offloaded 0/25 layers to GPU
"""

SAMPLE_PARTIAL_OFFLOAD_LOG = """
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB)
load_tensors: layer   0 assigned to device Vulkan0
load_tensors: layer   1 assigned to device CPU
load_tensors: offloaded 12/25 layers to GPU
"""


class VulkanVerificationTests(unittest.TestCase):
    def test_genuine_arc_vulkan_log_verifies(self) -> None:
        diag = parse_vulkan_diagnostics(SAMPLE_ARC_VULKAN_LOG)
        self.assertTrue(diag.gpu_verified)
        self.assertEqual(diag.actual_backend, "vulkan")
        self.assertEqual(diag.gpu_device, "Intel Arc 140V")
        self.assertEqual(diag.layers_offloaded, 25)
        self.assertEqual(diag.cpu_layer_count, 0)
        self.assertTrue(diag.has_arc)
        self.assertTrue(diag.has_vulkan_backend)

    def test_cpu_log_fails_verification(self) -> None:
        diag = parse_vulkan_diagnostics(SAMPLE_CPU_FALLBACK_LOG)
        self.assertFalse(diag.gpu_verified)
        self.assertEqual(diag.actual_backend, "cpu")
        self.assertIsNone(diag.gpu_device)
        self.assertEqual(diag.layers_offloaded, 0)
        self.assertEqual(diag.cpu_layer_count, 3)

    def test_partial_offload_fails_strict_verification(self) -> None:
        diag = parse_vulkan_diagnostics(SAMPLE_PARTIAL_OFFLOAD_LOG)
        self.assertFalse(diag.gpu_verified)
        self.assertEqual(diag.actual_backend, "vulkan-partial")
        self.assertGreater(diag.cpu_layer_count, 0)


if __name__ == "__main__":
    unittest.main()
