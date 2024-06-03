import argparse
import ast

import pandas as pd
from transformers import set_seed
from datasets import Dataset
import logging
import warnings
import os

from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Use Python logging for logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(file_path: str):
    """
    Function to read dataframe with columns.
    Args:
        file_path (str): Path to the file containing the validation/development data.
    Returns:
        Dataset object: The data as a Dataset object.
    """
    data_df = pd.read_csv(file_path, sep="\t")
    logger.info(f"{data_df.shape = }")
    data_df = data_df[["Technique", "Extracted Label", "Extracted Superlabel", "True Label", "True Superlabel"]]
    logger.info(f"{data_df.shape = }")
    # Drop rows with None/NaN for either of the five retained columns
    data_df = data_df.dropna(subset=["True Label", "True Superlabel"])
    logger.info(f"{data_df.shape = }")

    return Dataset.from_pandas(data_df)


def calculate_macro_f1(precision_recall_f1):
    f1_scores = [metrics['f1'] for metrics in precision_recall_f1.values()]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return macro_f1


def calculate_precision_recall_f1_per_class(predictions, true_labels):
    # Convert string to list
    true_labels = [ast.literal_eval(label) for label in true_labels]

    # Initialize counts for each class
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    # Unique classes from predictions and true labels
    classes = set(predictions)
    for true_list in true_labels:
        classes.update(true_list)

    print(classes)

    # Calculate True Positives, False Positives, and False Negatives for each class
    for pred, true_list in zip(predictions, true_labels):
        for cls in classes:
            if cls == pred:
                if cls in true_list:
                    true_positive[cls] += 1
                else:
                    false_positive[cls] += 1
            if cls in true_list and cls != pred:
                false_negative[cls] += 1

    # Calculate precision and recall for each class
    precision_recall_f1 = {}
    for cls in classes:
        precision = true_positive[cls] / (true_positive[cls] + false_positive[cls]) if (true_positive[cls] + false_positive[cls]) > 0 else 0
        recall = true_positive[cls] / (true_positive[cls] + false_negative[cls]) if (true_positive[cls] + false_negative[cls]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_recall_f1[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

    return precision_recall_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file_path",
        "-dp",
        help="Path to the input data (csv file with tab as separator). For example: 'output/output_data.csv'.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        # required=True,
        help="The random seed to use. Default: 0",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    # Set seed for replication
    set_seed(args.random_seed)

    if not os.path.exists(args.data_file_path):
        raise FileNotFoundError(f"Data file '{args.data_file_path}' not found.")

    data_path = args.data_file_path  # For example, "output/output_data.csv"

    # Get the data for the analyses
    logger.info("Loading the data...")
    val_data = get_data(data_path)
    # Lowercase all the columns
    val_data = val_data.map(lambda example: {"True Label": example["True Label"].lower()})
    val_data = val_data.map(lambda example: {"True Label": example["True Label"].replace('(', '').replace(')', '')})
    val_data = val_data.map(lambda example: {"True Superlabel": example["True Superlabel"].lower()})
    val_data = val_data.map(lambda example: {"Extracted Label": example["Extracted Label"].lower()})
    val_data = val_data.map(lambda example: {"Extracted Label": example["Extracted Label"].replace('appeal to worse problem', 'appeal to worse problems')})
    val_data = val_data.map(lambda example: {"Extracted Superlabel": example["Extracted Superlabel"].lower()})
    val_data = val_data.map(lambda example: {"True Superlabel": "appeal to emotion"} if example["True Superlabel"] == "fallacy of emotion" else {"True Superlabel": example["True Superlabel"]})

    print(val_data)

    for label_type in ["Label", "Superlabel"]:
        print(f"Calculating metrics for {label_type}...")
        precision_recall_f1_per_class = calculate_precision_recall_f1_per_class(val_data[f"Extracted {label_type}"], val_data[f"True {label_type}"])
        for cls, metrics in precision_recall_f1_per_class.items():
            print(f"Class: {cls}")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1: {metrics['f1']:.2f}")

        macro_f1 = calculate_macro_f1(precision_recall_f1_per_class)
        print(f"Macro F1 Score: {macro_f1:.2f}")
