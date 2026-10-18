from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


class BrowserPlaybackTests(unittest.TestCase):
    def test_tail_horizon_and_cancellation_lifecycle(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node is required for the mocked WebAudio scheduler test")
        root = Path(__file__).parents[1]
        result = subprocess.run(
            [node, "tests/browser_playback_harness.js"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertLessEqual(data["max_scheduled_ahead_ms"], 920)
        self.assertGreater(data["max_pending_buffer_ms"], 3000)
        self.assertLessEqual(data["scheduled_before_tts_complete"], 6)
        self.assertGreater(data["pending_before_tts_complete"], 10)
        self.assertFalse(data["playback_complete_immediately_after_tts_complete"])
        self.assertLess(data["subtitle_to_first_audio_ms"], 400)
        self.assertEqual(data["normal_tts_complete_stopped_sources"], 0)
        self.assertFalse(data["playback_done_before_last_source_ended"])
        self.assertTrue(data["last_source_ended"])
        self.assertTrue(data["playback_generation_complete"])
        self.assertTrue(data["all_final_audio_played"])
        self.assertTrue(data["final_tail_chunks_played"])
        self.assertTrue(data["no_gaps"])
        self.assertTrue(data["cancelled_sources_stopped"])
        self.assertEqual(data["cancelled_pending_chunks"], 0)
        self.assertFalse(data["cancelled_generation_complete_sent"])
        self.assertLessEqual(data["bounded_generation_count"], 9)
        self.assertTrue(data["active_generation_retained"])


if __name__ == "__main__":
    unittest.main()
