"""Preprocess the MAFALDA dataset to a TSV file."""

import json

import pandas as pd


def label_mapping(fallacies_map_path: str = "fallacies.json") -> dict[str, str]:
    """Load the fallacies mapping from the JSON file, and map them as a dict."""
    with open(fallacies_map_path) as f:
        fallacies = json.load(f)

    label_to_superlabel = {}
    for superlabel, labels in fallacies.items():
        for label in labels:
            label_to_superlabel[label.lower()] = superlabel

    return label_to_superlabel


def process_mafalda(mafalda_path: str, output_path: str, label_mapping: dict[str, str]) -> None:
    """Process the MAFALDA dataset and save it to a TSV file."""
    rows = []
    with open(mafalda_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            text = entry["text"] if "POST:" not in entry["text"] else entry["text"].split("POST:")[1].strip()
            labels = [
                label[2] for label in entry["labels"] if label[2].lower() not in ["nothing", "to clean"]
            ]  # remove "nothing" and "to clean"
            if not labels:  # i.e., if there was only "nothing" or "to clean"
                labels = ["No Fallacy"]  # there was no fallacy at all
                superlabels = ["No Fallacy"]
            else:
                superlabels = [label_mapping[label.lower()] for label in labels if label.lower() in label_mapping]

            assert len(labels) == len(superlabels), f"Labels: {labels}, Superlabels: {superlabels}"
            rows.append({"Input": text, "MAFALDA Label": labels, "MAFALDA Superlabel": superlabels})

    mafalda_df = pd.DataFrame(rows)
    mafalda_df.to_csv(output_path, index=False, sep="\t")


def main() -> None:
    """Run the main function."""
    label_to_superlabel = label_mapping()
    process_mafalda("datasets/MAFALDA_gold.jsonl", "cleaned_datasets/MAFALDA_gold_processed.tsv", label_to_superlabel)


if __name__ == "__main__":
    main()
