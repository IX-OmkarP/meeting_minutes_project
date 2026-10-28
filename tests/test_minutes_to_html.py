import unittest

from transcription_utils import minutes_to_html


class MinutesToHtmlTests(unittest.TestCase):
    def test_headings_bold_bullets_listed_text_escaped(self):
        out = minutes_to_html(
            "Agenda\nReviewed Q3 <budget> & plan.\n\nDecisions:\n- Ship v2\n- Hire QA\nAction Items\n- Omkar: deploy"
        )
        self.assertEqual(
            out,
            "<p><b>Agenda</b></p>\n"
            "<p>Reviewed Q3 &lt;budget&gt; &amp; plan.</p>\n"
            "<p><b>Decisions</b></p>\n"
            "<ul>\n<li>Ship v2</li>\n<li>Hire QA</li>\n</ul>\n"
            "<p><b>Action Items</b></p>\n"
            "<ul>\n<li>Omkar: deploy</li>\n</ul>",
        )
