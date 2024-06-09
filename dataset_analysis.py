"""Analyze and plot some statistics of the datasets."""

import argparse
import ast

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main() -> None:  # noqa: PLR0915
    """Run the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        "-g",
        help="Path to the cleaned gold data (csv file with tab as separator). "
        "For example: 'cleaned_datasets/MAFALDA_gold_processed.tsv'.",
        type=str,
        default="cleaned_datasets/MAFALDA_gold_processed.tsv",
    )
    parser.add_argument(
        "--validation",
        "-v",
        help="Path to the cleaned validation data (csv file with tab as separator). "
        "For example: 'cleaned_datasets/unified_validation_set.tsv'.",
        type=str,
        default="cleaned_datasets/unified_validation_set.tsv",
    )
    parser.add_argument(
        "--downsampled",
        "-d",
        help="Path to the cleaned downsampled validation data (csv file with tab as separator). "
        "For example: 'cleaned_datasets/unified_validation_set_downsampled.tsv'.",
        type=str,
        default="cleaned_datasets/unified_validation_set_downsampled.tsv",
    )

    args = parser.parse_args()

    palette = {"MAFALDA": "C0", "Validation": "C1", "Downsampled": "C2"}

    downsampled = pd.read_csv(args.downsampled, sep="\t")
    gold = pd.read_csv(args.gold, sep="\t")

    valid = pd.read_csv(args.validation, sep="\t").sample(n=len(gold), random_state=42)

    valid["Dataset"] = "Validation"
    downsampled["Dataset"] = "Downsampled"
    gold["Dataset"] = "MAFALDA"

    gold_labels = gold.drop(columns="MAFALDA Superlabel").dropna()
    gold_superlabels = gold.drop(columns="MAFALDA Label").dropna()

    gold_labels["MAFALDA Label"] = gold_labels["MAFALDA Label"].apply(ast.literal_eval)
    gold_labels = gold_labels.explode("MAFALDA Label")

    gold_superlabels["MAFALDA Superlabel"] = gold_superlabels["MAFALDA Superlabel"].apply(ast.literal_eval)
    gold_superlabels = gold_superlabels.explode("MAFALDA Superlabel")

    gold_labels["MAFALDA Label"] = gold_labels["MAFALDA Label"].replace(
        {"ad hominem": "Abusive Ad Hominem", "appeal to (false) authority": "Appeal to False Authority"}
    )

    print(gold_superlabels["MAFALDA Superlabel"].unique())
    print(gold_labels["MAFALDA Label"].unique())

    # Concatenate the gold, valid, and downsampled dataframes
    merged_df_labels = pd.concat(
        [
            gold_labels,
            valid.drop(columns="MAFALDA Superlabel").dropna(),
            downsampled.drop(columns="MAFALDA Superlabel").dropna(),
        ],
        ignore_index=True,
    )

    merged_df_superlabels = pd.concat(
        [
            gold_superlabels,
            valid.drop(columns="MAFALDA Label").dropna(),
            downsampled.drop(columns="MAFALDA Label").dropna(),
        ],
        ignore_index=True,
    )

    merged_df_labels["MAFALDA Label"] = merged_df_labels["MAFALDA Label"].apply(lambda x: str(x).lower())
    merged_df_superlabels["MAFALDA Superlabel"] = merged_df_superlabels["MAFALDA Superlabel"].apply(
        lambda x: str(x).lower()
    )

    print(merged_df_labels)
    print(merged_df_superlabels)

    # Text length distribution
    # Should be a boxplot
    merged_df_labels["Input length"] = merged_df_labels["Input"].apply(len)
    merged_df_superlabels["Input length"] = merged_df_superlabels["Input"].apply(len)

    plt.figure()
    sns.boxplot(data=merged_df_superlabels, x="Input length", y="MAFALDA Superlabel", hue="Dataset", palette=palette)
    plt.tight_layout()
    plt.savefig("plots/text_length_distribution_superlabels.png")

    plt.figure()
    sns.boxplot(data=merged_df_labels, x="Input length", y="MAFALDA Label", hue="Dataset", palette=palette)
    plt.tight_layout()
    plt.savefig("plots/text_length_distribution_labels.png")

    # Histogram of superlabels
    plt.figure()
    sns.countplot(
        data=merged_df_superlabels,
        y="MAFALDA Superlabel",
        order=merged_df_superlabels["MAFALDA Superlabel"].value_counts().index,
        hue="Dataset",
        palette=palette,
    )
    plt.tight_layout()
    plt.savefig("plots/superlabel_distribution.png")

    # Histogram of labels
    plt.figure()
    plt.xticks(rotation=90)
    sns.countplot(
        data=merged_df_labels,
        y="MAFALDA Label",
        order=merged_df_labels["MAFALDA Label"].value_counts().index,
        hue="Dataset",
        palette=palette,
    )
    plt.tight_layout()
    plt.savefig("plots/label_distribution.png")

    # Vocab size per label
    # This is actually a pointless visualization now that I think about it
    vocab = (
        merged_df_labels.groupby(["MAFALDA Label", "Dataset"])["Input"]
        .apply(lambda x: len(set(" ".join(x).split())))
        .reset_index()
    )

    plt.figure()
    plt.xticks(rotation=90)
    sns.barplot(
        data=vocab,
        y="MAFALDA Label",
        x="Input",
        hue="Dataset",
        palette=palette,
    )
    plt.tight_layout()
    plt.savefig("plots/label_vocab_size_distribution.png")

    # Vocab size per superlabel
    # This is actually a pointless visualization now that I think about it
    vocab = (
        merged_df_superlabels.groupby(["MAFALDA Superlabel", "Dataset"])["Input"]
        .apply(lambda x: len(set(" ".join(x).split())))
        .reset_index()
    )
    plt.figure()
    plt.xticks(rotation=90)
    sns.barplot(data=vocab, y="MAFALDA Superlabel", x="Input", hue="Dataset", palette=palette)
    plt.tight_layout()
    plt.savefig("plots/superlabel_vocab_size_distribution.png")


if __name__ == "__main__":
    main()
