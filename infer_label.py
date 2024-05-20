import json
import re

from collections import defaultdict

class LabelExtractor:
    def __init__(self, fallacies_file: str):
        with open(fallacies_file, 'r') as f:
            self.fallacies = json.load(f)

        self.fallacy_mapping = defaultdict(lambda: "")
        for category, fallacies in self.fallacies.items():
            for fallacy in fallacies:
                self.fallacy_mapping[fallacy.lower()] = category

    def extract_label(self, llm_response: str) -> str:
        llm_response = llm_response.lower()

        for fallacy in self.fallacy_mapping.keys():
            if re.search(r'\b' + re.escape(fallacy) + r'\b', llm_response):
                return self.fallacy_mapping[fallacy], fallacy

        return "No Fallacy"

if __name__ == "__main__":
    extractor = LabelExtractor('fallacies.json')

    response = ("This is a Slippery Slope fallacy because it assumes "
                "that allowing children to play video games will inevitably "
                "lead them down a nasty path.")
    lvl1, lvl2 = extractor.extract_label(response)
    print(f"Level 1 and labels: {lvl1, lvl2}")