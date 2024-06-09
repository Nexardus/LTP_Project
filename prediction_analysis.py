"""Analyze the prediction biases of the models."""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        help="Path to the cleaned output data (csv file with tab as separator). "
        "For example: 'cleaned_output/inference_output_cleaned.tsv'.",
        type=str,
        default="cleaned_output/inference_output_cleaned.tsv",
    )

    args = parser.parse_args()

    dataset_name = Path(args.dataset).stem

    data = pd.read_csv(args.dataset, sep="\t")

    print(data)

    data = data.rename(columns={"Extracted Label": "MAFALDA Label", "Extracted Superlabel": "MAFALDA Superlabel"})

    for technique in data["Technique"].unique():
        # Histogram of labels
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(
            data=data[data["Technique"] == technique],
            y="MAFALDA Label",
            order=data["MAFALDA Label"].value_counts().index,
            hue="Model",
        )
        plt.tight_layout()
        plt.savefig(f"plots/{dataset_name}_predictions_label_distribution_{technique}.png")

        # Histogram of labels
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(
            data=data[data["Technique"] == technique],
            y="MAFALDA Superlabel",
            order=data["MAFALDA Superlabel"].value_counts().index,
            hue="Model",
        )
        plt.tight_layout()
        plt.savefig(f"plots/{dataset_name}_predictions_superlabel_distribution_{technique}.png")


if __name__ == "__main__":
    main()
