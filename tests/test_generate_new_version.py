import unittest

from transcription_utils import generate_new_version


class GenerateNewVersionTests(unittest.TestCase):
    def test_skips_duplicates_ignoring_whitespace_and_case(self):
        outputs = iter(["Agenda\n- A", "agenda   - a ", "Agenda\n- B"])
        result = generate_new_version(lambda: next(outputs), ["Agenda\n- A"])
        self.assertEqual(result, "Agenda\n- B")

    def test_returns_none_when_every_attempt_repeats(self):
        result = generate_new_version(lambda: "same", ["same"], attempts=3)
        self.assertIsNone(result)

    def test_first_version_accepted_when_no_history(self):
        self.assertEqual(generate_new_version(lambda: "x", []), "x")
