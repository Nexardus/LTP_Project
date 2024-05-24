from pathlib import Path

from nltk import wordpunct_tokenize
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main():

    data = pd.read_csv("output/inference_output_21_may_cleaned.csv", sep="\t")

    print(data)

    for technique in data["Technique"].unique():

        # Histogram of labels
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(
            data=data[data['Technique'] == technique],
            y="MAFALDA Label",
            order=data["MAFALDA Label"].value_counts().index,
            hue="Model",

        )
        plt.tight_layout()
        plt.savefig(f"plots/downsampled_validation_predictions_label_distribution_{technique}.png")

        # Histogram of labels
        plt.figure()
        plt.xticks(rotation=90)
        sns.countplot(
            data=data[data['Technique'] == technique],
            y="MAFALDA Superlabel",
            order=data["MAFALDA Superlabel"].value_counts().index,
            hue="Model",

        )
        plt.tight_layout()
        plt.savefig(f"plots/downsampled_validation_predictions_superlabel_distribution_{technique}.png")

if __name__ == "__main__":
    main()
