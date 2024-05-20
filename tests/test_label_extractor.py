# just run this one with python -m unittest

import unittest
from label_extractor import LabelExtractor

class TestLabelExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = LabelExtractor('fallacies.json')

    def test_slippery_slope(self):
        response = ("This is a Slipper Slope fallacy because it assumes "
                    "that allowing children to play video games will inevitably "
                    "lead them down a nasty path.")
        lvl1, lvl2 = self.extractor.extract_label(response)
        self.assertEqual(lvl2, "slippery slope")

    def test_no_fallacy(self):
        response = "Take a shot, all cool kids do that!"
        lvl1, lvl2 = self.extractor.extract_label(response)
        self.assertEqual(lvl2, "No Fallacy")

    def test_appeal_to_fear(self):
        response = ("This is an Appeal to   Fear because it tries to scare people "
                    "into believing woogie-boogies eat your pants.")
        lvl1, lvl2 = self.extractor.extract_label(response)
        self.assertEqual(lvl2, "Appeal to Fear")

    def test_false_authority(self):
        response = ("This is a False  authority fallacy because it cites a "
                    "random redditor as an expert.")
        lvl1, lvl2 = self.extractor.extract_label(response)
        self.assertEqual(lvl2, "False Authority")

if __name__ == "__main__":
    unittest.main()
