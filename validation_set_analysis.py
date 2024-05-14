from nltk import wordpunct_tokenize
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
    data["Input length"] = data["Input"].apply(len)
    data["Unique words"] = data["Input"].apply(lambda x: len(wordpunct_tokenize(x)))

    print(data["Input length"].describe())
    print(data["Unique words"].describe())

    plt.figure()
    sns.boxplot(data=data, x="Unique words", y="MAFALDA Superlabel")
    plt.tight_layout()
    plt.savefig("plots/validation_unique_word_distribution.png")

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

    # Vocab size per class
    # This is actually a pointless visualization now that I think about it
    print(data)
    vocab = data.groupby('MAFALDA Label')['Input'].apply(lambda x: len(set(" ".join(x).split())))
    plt.figure()
    plt.xticks(rotation=90)
    sns.barplot(
        data=vocab,
        order=vocab.sort_values(ascending=False).index,
    )
    plt.tight_layout()
    plt.savefig("plots/validation_label_vocab_size_distribution.png")



if __name__ == "__main__":
    main()
