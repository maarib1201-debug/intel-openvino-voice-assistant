from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2v.config import Settings
from v2v.llm import TinyLLM


class GenerationSettingsTests(unittest.TestCase):
    def test_generation_environment_overrides(self) -> None:
        env = {
            "V2V_LLM_TEMPERATURE": "0.35",
            "V2V_LLM_TOP_P": "0.85",
            "V2V_LLM_TOP_K": "24",
            "V2V_LLM_REPEAT_PENALTY": "1.08",
            "V2V_MAX_NEW_TOKENS": "96",
            "V2V_FACTUAL_GUARD_ENABLED": "false",
            "V2V_FACTUAL_GUARD_SPECIFICITY_SCORE": "3",
            "V2V_FACTUAL_GUARD_CAUSAL_SCORE": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Settings()
        self.assertEqual(cfg.llm_temperature, 0.35)
        self.assertEqual(cfg.llm_top_p, 0.85)
        self.assertEqual(cfg.llm_top_k, 24)
        self.assertEqual(cfg.llm_repeat_penalty, 1.08)
        self.assertEqual(cfg.max_new_tokens, 96)
        self.assertFalse(cfg.factual_guard_enabled)
        self.assertEqual(cfg.factual_guard_specificity_score, 3)
        self.assertEqual(cfg.factual_guard_causal_score, 4)

    def test_answer_sampling_is_forwarded_to_llama(self) -> None:
        captured = {}

        class FakeLlama:
            def reset(self) -> None:
                return

            def create_completion(self, *args, **kwargs):
                captured.update(kwargs)
                def chunks():
                    yield {"choices": [{"text": "answer"}]}

                return chunks()

        llm = TinyLLM.__new__(TinyLLM)
        llm.cfg = Settings(
            llm_temperature=0.31,
            llm_top_p=0.82,
            llm_top_k=17,
            llm_repeat_penalty=1.07,
            max_new_tokens=99,
        )
        llm.llama = FakeLlama()
        llm.history = []
        llm.thinking = "disabled"
        llm._llama_stopping_cls = None
        llm.t_prompt_eval = 0.0
        llm.t_first_visible = 0.0
        llm.last_prompt_n = 0
        self.assertEqual("".join(llm._stream_llama("hello", threading.Event(), [])), "answer")
        self.assertEqual(captured["temperature"], 0.31)
        self.assertEqual(captured["top_p"], 0.82)
        self.assertEqual(captured["top_k"], 17)
        self.assertEqual(captured["repeat_penalty"], 1.07)
        self.assertEqual(captured["max_tokens"], 99)

    def test_strict_vulkan_failure_never_loads_hf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.gguf"
            model.touch()
            cfg = Settings(gguf_path=model, llm_n_gpu_layers=-1, llm_cpu_fallback=False)
            with (
                patch("v2v.llm.bootstrap_vulkan_runtime", side_effect=RuntimeError("no Vulkan")),
                patch.object(TinyLLM, "_load_hf") as load_hf,
            ):
                with self.assertRaisesRegex(RuntimeError, "Strict Vulkan mode failed"):
                    TinyLLM(cfg)
            load_hf.assert_not_called()

    def test_strict_vulkan_missing_gguf_never_loads_hf(self) -> None:
        cfg = Settings(
            gguf_path=Path("definitely-missing-model.gguf"),
            llm_n_gpu_layers=-1,
            llm_cpu_fallback=False,
        )
        with patch.object(TinyLLM, "_load_hf") as load_hf:
            with self.assertRaisesRegex(RuntimeError, "Strict Vulkan mode requires"):
                TinyLLM(cfg)
        load_hf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
