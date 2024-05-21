import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset, load_dataset
import logging
import warnings
import os
from statistics import harmonic_mean

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


def make_dicts(data):
    """"""
    # Precision dictionary:
    # For each label, document how often the label has been predicted (pred)\
    # and how much of the time it was the gold label as well (corr).
    prec_dict = {}

    # Recall dictionary:
    # For each label, document how often the label was the ground truth / gold label (occu)\
    # and how much of the time it was predicted as such (pred).
    rec_dict = {}

    # F1 dictionary:
    # For each label, document the precision, recall and F1 score
    f1_dict = {}

    # Insert all the (Super) labels into both dicts
    for label in np.unique(data["MAFALDA Label"]):
        prec_dict[f"{label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{label}"] = {"occu": 0, "pred": 0}
        f1_dict[f"{label}"] = {"precision": 0, "recall": 0, "f1": 0}

    for sup_label in np.unique(data["MAFALDA Superlabel"]):
        prec_dict[f"{sup_label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{sup_label}"] = {"occu": 0, "pred": 0}
        f1_dict[f"{sup_label}"] = {"precision": 0, "recall": 0, "f1": 0}

    return prec_dict, rec_dict, f1_dict

def get_prec_rec(example, idx, **fn_kwargs):
    """"""

    # Iterate over the predicted labels and document in prec_dict
    for pred_label in example["<predicted label>"]:
        fn_kwargs["prec_dict"][pred_label]["pred"] += 1
        if example[pred_label] in example["MAFALDA Label"]:
            fn_kwargs["prec_dict"]["MAFALDA Label"]["corr"] += 1

    # Iterate over the predicted Super labels and document in prec_dict
    for pred_sup_label in example["<predicted super label>"]:
        # Exclude "No fallacy" because it overlaps with the level 2 label
        if pred_sup_label == "No fallacy":
            continue
        fn_kwargs["prec_dict"][pred_sup_label]["pred"] += 1
        if example[pred_sup_label] in example["MAFALDA Superlabel"]:
            fn_kwargs["prec_dict"]["MAFALDA Label"]["corr"] += 1

    # Iterate over the gold labels and document in rec_dict
    for gold_label in example["MAFALDA Label"]:
        fn_kwargs["rec_dict"][gold_label]["occu"] += 1
        if gold_label in example["<predicted label>"]:
            fn_kwargs["rec_dict"][gold_label]["pred"] += 1

    # Iterate over the gold Super labels and document in rec_dict
    for gold_sup_label in example["MAFALDA Superlabel"]:
        # Exclude "No fallacy" because it overlaps with the level 2 label
        if gold_sup_label == "No fallacy":
            continue
        fn_kwargs["rec_dict"][gold_sup_label]["occu"] += 1
        if gold_sup_label in example["<predicted super label>"]:
            fn_kwargs["rec_dict"][gold_sup_label]["pred"] += 1

    # Compute precision, recall and F1 per label
    for item in fn_kwargs["prec_dict"].items():
        k = item[0], v = item[1]
        # Add rounding and error handling such as ZeroDivisionError --> try-except statement
        try:
            fn_kwargs["f1_dict"][k]["precision"] = v["corr"] / v["pred"]
        except ZeroDivisionError:
            fn_kwargs["f1_dict"][k]["precision"] = 0

    for item in fn_kwargs["rec_dict"].items():
        k = item[0], v = item[1]
        # Add rounding and error handling such as ZeroDivisionError --> try-except statement
        try:
            fn_kwargs["f1_dict"][k]["recall"] = v["pred"] / v["occu"]
        except ZeroDivisionError:
            fn_kwargs["f1_dict"][k]["recall"] = 0

    for item in fn_kwargs["f1_dict"].items():
        k = item[0], v = item[1]
        try:
            fn_kwargs["f1_dict"][k]["f1"] = harmonic_mean([v["precision"], v["recall"]])
        except:
            fn_kwargs["f1_dict"][k]["f1"] = 0
        # Or this fancy-pancy manual calculation
        # len([v["precision"], v["recall"]]) / sum([1/x for x in [v["precision"], v["recall"]]])

    # Compute macro (ignore the label distribution) F1 in total
    label_c = 0
    label_f1 = 0
    sup_label_c = 0
    sup_label_f1 = 0
    for key in fn_kwargs["f1_dict"].keys():
        # Exclude "No fallacy" because it overlaps with the level 2 label
        if key not in ["Fallacy of Credibility", "Fallacy of Logic", "Appeal to Emotion"]:
            label_c += 1
            label_f1 += fn_kwargs["f1_dict"][key]["f1"]
        if key in ["Fallacy of Credibility", "Fallacy of Logic", "Appeal to Emotion"]:
            sup_label_c += 1
            sup_label_f1 += fn_kwargs["f1_dict"][key]["f1"]

    # Add rounding and error handling such as ZeroDivisionError --> try-except statement
    try:
        fn_kwargs["f1_dict"]["total_label_macro"] = label_f1 / label_c
    except ZeroDivisionError:
        fn_kwargs["f1_dict"]["total_label_macro"] = 0
    try:
        fn_kwargs["f1_dict"]["total_super_label_macro"] = sup_label_f1 / sup_label_c
    except ZeroDivisionError:
        fn_kwargs["f1_dict"]["total_super_label_macro"] = 0

    return fn_kwargs["f1_dict"]



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

    prec_dict, rec_dict, f1_dict = make_dicts(val_data)

    eval_scores = val_data.map(
        get_prec_rec,
        with_indices=True,
        fn_kwargs={
            "prec_dict": prec_dict,
            "rec_dict": rec_dict,
            "f1_dict": f1_dict
        }
    )

    # Export dataframe
    #os.makedirs(args.output_file_path, exist_ok=True)
    #logger.info(f"Exporting dataframe to '{args.output_file_path}.[json|csv]'...")
    #output_df.to_csv(args.output_file_path + ".csv", index=False, sep=',')
    #output_df.to_json(args.output_file_path + ".json", orient="records", lines=True)



