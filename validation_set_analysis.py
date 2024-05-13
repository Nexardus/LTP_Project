import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main():
    data = pd.read_csv("datasets/unified_validation_set.tsv", sep="\t")

    # Number of samples with only a superlabel
    print(data["MAFALDA Superlabel"].notna().sum())

    # Number of samples with both a superlabel and a label
    print(data["MAFALDA Label"].notna().sum())

    # Text length distribution
    # Should be a boxplot
    print(data["Input"].apply(len).describe())
    data["Input length"] = data["Input"].apply(len)
    plt.figure()
    sns.boxplot(data=data, x="Input length", y="MAFALDA Superlabel")
    plt.tight_layout()
    plt.savefig("plots/validation_text_length_distribution.png")

    # Histogram of superlabels
    plt.figure()
    sns.countplot(
        data=data,
        x="MAFALDA Superlabel",
        order=data["MAFALDA Superlabel"].value_counts().index,
    )
    plt.tight_layout()
    plt.savefig("plots/validation_superlabel_distribution.png")

    # Histogram of labels
    plt.figure()
    plt.xticks(rotation=90)
    sns.countplot(
        data=data,
        x="MAFALDA Label",
        order=data["MAFALDA Label"].value_counts().index,
        hue="MAFALDA Superlabel",
    )
    plt.tight_layout()
    plt.savefig("plots/validation_label_distribution.png")

    print(data)


if __name__ == "__main__":
    main()
