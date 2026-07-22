import unittest

from transcription_utils import get_chunk_windows


class TranscriptionChunkingTests(unittest.TestCase):
    def test_large_audio_is_split_into_smaller_windows(self):
        windows = get_chunk_windows(
            total_ms=3_600_000,
            max_bytes=2_000_000,
            max_chunk_ms=10 * 60 * 1000,
            bitrate_kbps=32,
            min_chunk_ms=60_000,
        )

        self.assertGreater(len(windows), 1)
        self.assertTrue(all(end - start <= 10 * 60 * 1000 for start, end in windows))


if __name__ == "__main__":
    unittest.main()
