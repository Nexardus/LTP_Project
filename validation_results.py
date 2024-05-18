import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset, load_dataset
import logging
import warnings
import os

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

    return Dataset.from_pandas(data_df)


def get_prec_rec(example, idx):
    """"""
    # Precision dictionary:
    # For each label, document how often the label has been predicted (pred)\
    # and how much of the time it was the gold label as well (corr).
    prec_dict = {}

    # Recall dictionary:
    # For each label, document how often the label was the ground truth / gold label (occu)\
    # and how much of the time it was predicted as such (pred).
    rec_dict = {}

    # Insert all the (Super) labels into both dicts
    for label in np.unique(example["MAFALDA Label"]):
        prec_dict[f"{label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{label}"] = {"occu": 0, "pred": 0}

    for sup_label in np.unique(example["MAFALDA Superlabel"].unique):
        prec_dict[f"{sup_label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{sup_label}"] = {"occu": 0, "pred": 0}

    # Iterate over the predicted labels and document in pred_dict
    for pred_label in example["<predicted label>"]:
        prec_dict[pred_label]["pred"] += 1
        if example[pred_label] in example["MAFALDA Label"]:
            prec_dict["MAFALDA Label"]["corr"] += 1

    # Iterate over the predicted Super labels and document in pred_dict
    for pred_sup_label in example["<predicted super label>"]:
        prec_dict[pred_sup_label]["pred"] += 1
        if example[pred_sup_label] in example["MAFALDA Superlabel"]:
            prec_dict["MAFALDA Label"]["corr"] += 1

    # Iterate over the gold labels and document in rec_dict
    for gold_label in example["MAFALDA Label"]:
        rec_dict[gold_label]["occu"] += 1
        if gold_label in example["<predicted label>"]:
            rec_dict[gold_label]["pred"] += 1

    # Iterate over the gold Super labels and document in rec_dict
    for gold_sup_label in example["MAFALDA Superlabel"]:
        rec_dict[gold_sup_label]["occu"] += 1
        if gold_sup_label in example["<predicted super label>"]:
            rec_dict[gold_sup_label]["pred"] += 1

    # Compute precision, recall and F1 per label and in total



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file_path",
        "-dp",
        help="Path to the input data.",
        required=True,
        type=str,
    )
    parser.add_argument(
       "--output_file_path",
       "-out",
       required=True,
       help="Path where to save the output file.",
       type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
        help="The random seed to use.",
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
    logger.info(f"Loading the data...")
    val_data = get_data(data_path)

    val_data.map(
        get_prec_rec,
        with_indices=True
    )

    # Export dataframe
    #os.makedirs(args.output_file_path, exist_ok=True)
    #logger.info(f"Exporting dataframe to '{args.output_file_path}.[json|csv]'...")
    #output_df.to_csv(args.output_file_path + ".csv", index=False, sep=',')
    #output_df.to_json(args.output_file_path + ".json", orient="records", lines=True)



