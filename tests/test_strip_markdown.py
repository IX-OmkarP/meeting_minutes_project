import unittest

from transcription_utils import strip_markdown


class StripMarkdownTests(unittest.TestCase):
    def test_removes_heading_markers_but_keeps_heading_text(self):
        self.assertEqual(strip_markdown("## Agenda"), "Agenda")

    def test_converts_asterisk_bullets_to_hyphens(self):
        out = strip_markdown("* First point\n* Second point")
        self.assertEqual(out, "- First point\n- Second point")

    def test_keeps_indentation_on_nested_bullets(self):
        out = strip_markdown("* Top\n    * Nested")
        self.assertEqual(out, "- Top\n    - Nested")

    def test_unwraps_bold_and_italic_without_losing_text(self):
        out = strip_markdown("**Decisions** and *Action Items* stay")
        self.assertEqual(out, "Decisions and Action Items stay")

    def test_leaves_no_asterisk_or_hash_anywhere(self):
        messy = "### Meeting Minutes\n**Owner:** Omkar\n* fix the *web part*\n#1 priority"
        out = strip_markdown(messy)
        self.assertNotIn("*", out)
        self.assertNotIn("#", out)
        self.assertIn("Owner: Omkar", out)
        self.assertIn("- fix the web part", out)

    def test_handles_empty_output(self):
        self.assertEqual(strip_markdown(""), "")
        self.assertIsNone(strip_markdown(None))


if __name__ == "__main__":
    unittest.main()
