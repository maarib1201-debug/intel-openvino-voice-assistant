from __future__ import annotations

import unittest

from v2v.stt_correct import apply_stt_vocab, load_vocab, normalize_application_text


class SttVocabCorrectionTests(unittest.TestCase):
    def test_vocab_contains_only_the_three_seed_terms(self) -> None:
        terms = load_vocab()
        self.assertEqual({term.canonical for term in terms}, {"Kubernetes", "OpenVINO", "Qwen"})

    def test_openveino_in_technical_context(self) -> None:
        text, hits = apply_stt_vocab("OpenVeino NPU inference")
        self.assertEqual(text, "OpenVINO NPU inference")
        self.assertEqual([hit.term for hit in hits], ["OpenVINO"])

    def test_quinn_in_model_context(self) -> None:
        text, hits = apply_stt_vocab("run Quinn on the GPU")
        self.assertEqual(text, "run Qwen on the GPU")
        self.assertEqual([hit.term for hit in hits], ["Qwen"])

    def test_kubernetes_phonetic_misses_in_technical_context(self) -> None:
        cases = {
            "explain culbinates simply": "explain Kubernetes simply",
            "Explain corbell nets simply": "Explain Kubernetes simply",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                text, hits = apply_stt_vocab(original)
                self.assertEqual(text, expected)
                self.assertEqual([hit.term for hit in hits], ["Kubernetes"])

    def test_no_correction_in_unrelated_sentences(self) -> None:
        prompts = [
            "Quinn went to the store",
            "I talked to Quinn yesterday",
            "Meet me at the bar tonight",
            "Give me a good job",
            "The nets are in the backyard",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                text, hits = apply_stt_vocab(prompt)
                self.assertEqual(text, prompt)
                self.assertEqual(hits, [])

    def test_ambiguous_quinn_without_model_context_is_untouched(self) -> None:
        text, hits = apply_stt_vocab("Quinn is coming over later")
        self.assertEqual(text, "Quinn is coming over later")
        self.assertEqual(hits, [])

    def test_multiple_technical_terms_in_one_sentence(self) -> None:
        text, hits = apply_stt_vocab("run Quinn with OpenVeino NPU inference")
        self.assertEqual(text, "run Qwen with OpenVINO NPU inference")
        self.assertEqual({hit.term for hit in hits}, {"Qwen", "OpenVINO"})

    def test_does_not_rewrite_already_correct_terms(self) -> None:
        original = "Explain Kubernetes and OpenVINO with Qwen"
        text, hits = apply_stt_vocab(original)
        self.assertEqual(text, original)
        self.assertEqual(hits, [])

    def test_normalizes_spoken_numbers_only_for_arithmetic(self) -> None:
        self.assertEqual(normalize_application_text("What is forty five times nineteen?"), "What is 45 times 19")
        self.assertEqual(normalize_application_text("I saw forty five birds"), "I saw forty five birds")


if __name__ == "__main__":
    unittest.main()
