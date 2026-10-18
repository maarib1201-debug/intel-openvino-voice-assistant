from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2v.config import Settings
from v2v.engine import health_snapshot
from v2v.server import parse_args


class ServerHealthTests(unittest.TestCase):
    def test_health_reports_component_states(self) -> None:
        models = SimpleNamespace(
            asr=SimpleNamespace(device="ov-npu"),
            llm=SimpleNamespace(backend="llama", gpu_offload=True, fallback_reason=""),
            tts=SimpleNamespace(model=object(), sr=24000),
        )
        health = health_snapshot(models)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["components"]["stt"]["device"], "ov-npu")
        self.assertTrue(health["components"]["vulkan"]["available"])
        self.assertIn("tool_providers", health["components"])

    def test_cpu_fallback_is_visible_as_degraded_vulkan(self) -> None:
        models = SimpleNamespace(
            asr=SimpleNamespace(device="ov-npu"),
            llm=SimpleNamespace(backend="llama", gpu_offload=False, fallback_reason="Vulkan initialization failed"),
            tts=SimpleNamespace(model=object(), sr=24000),
        )
        health = health_snapshot(models)
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["components"]["vulkan"]["status"], "degraded")

    def test_non_local_bind_requires_explicit_flag(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--host", "0.0.0.0"])

    def test_non_local_bind_can_be_explicitly_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {"V2V_AUTH_TOKEN": "test-secret", "V2V_ALLOWED_ORIGINS": "http://10.0.0.5:8088"},
        ):
            cfg = parse_args(["--host", "0.0.0.0", "--allow-lan"])
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertTrue(cfg.allow_lan)

    def test_non_local_bind_rejects_missing_auth(self) -> None:
        with patch.dict(os.environ, {"V2V_AUTH_TOKEN": "", "V2V_ALLOWED_ORIGINS": "*"}, clear=False):
            with self.assertRaises(SystemExit):
                parse_args(["--host", "0.0.0.0", "--allow-lan"])

    def test_ready_contract_supports_session_id(self) -> None:
        source = (Path(__file__).parents[1] / "v2v" / "server.py").read_text(encoding="utf-8")
        self.assertIn('ready_payload["session_id"] = session_id', source)
        self.assertIn('BROWSER_PLAYBACK_EVENT', source)


if __name__ == "__main__":
    unittest.main()
