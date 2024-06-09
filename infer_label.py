import json
import re
from collections import defaultdict
import difflib


class LabelExtractor:
    def __init__(self, fallacies_file: str):
        with open(fallacies_file, "r") as f:
            self.fallacies = json.load(f)

        self.fallacy_mapping = defaultdict(lambda: "")
        for category, fallacies in self.fallacies.items():
            for fallacy in fallacies:
                self.fallacy_mapping[fallacy.lower()] = category

        self.prefixes = ["output:", "answer:", "label:", "fallacy:", "value:"]
        self.similarity_threshold = 0.75

    def extract_label(self, llm_response: str) -> str:
        llm_response = llm_response.lower()

        # pattern to match prefixes and ensure only (semi-)valid fallacy names are captured
        pattern = r"(" + "|".join(re.escape(prefix) for prefix in self.prefixes) + r")\s*(\w[\w\s]*)(?=\s|$)"

        for line in llm_response.splitlines():
            match = re.search(pattern, line.strip())
            if match:
                prefix, potential_label = match.groups()
                potential_label = potential_label.strip()

                # check for exact matches first
                for fallacy in self.fallacy_mapping.keys():
                    if fallacy == potential_label:
                        return self.fallacy_mapping[fallacy], fallacy

                # collect partial matches and their similarity scores
                partial_matches = []
                for fallacy in self.fallacy_mapping.keys():
                    if fallacy.startswith(potential_label.split()[0]):
                        similarity = difflib.SequenceMatcher(None, fallacy, potential_label).ratio()
                        partial_matches.append((similarity, fallacy))

                # if partial matches found, select the most similar one above the threshold
                # (to avoid any "appeal to potato potahto" being matched)
                if partial_matches:
                    partial_matches.sort(reverse=True, key=lambda x: x[0])
                    best_match_similarity, best_match = partial_matches[0]
                    if best_match_similarity >= self.similarity_threshold:
                        return self.fallacy_mapping[best_match], best_match

                # first match of a pattern had no (close to) perfect format of predicted label - wrong format
                return "No Fallacy", "No Fallacy"

        return "No Fallacy", "No Fallacy"


if __name__ == "__main__":
    extractor = LabelExtractor("fallacies.json")

    response = (
        "Output: Fallacy: Appeal to Ppiity \n\n"
        "Input: The victim’s family has been torn apart by this act of terror. Put yourselves in their terrible situation, you will see that he is guilty.\n\n"
        "Output: Fallacy: Appeal to Anger\n\n"
        "Input: If you"
    )
    lvl1, lvl2 = extractor.extract_label(response)
    print(f"Level 1 and 2 labels: {lvl1, lvl2}")
