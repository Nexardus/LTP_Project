import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset, load_dataset
import logging
import warnings
import os
from statistics import harmonic_mean
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

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
    #logger.info(data["True Label"][:10])
    logger.info(f"All unique true labels: {np.unique(data['True Label'])}")

    for label in np.unique(data["True Label"]):
        prec_dict[f"{label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{label}"] = {"occu": 0, "pred": 0}
        f1_dict[f"{label}"] = {"precision": 0, "recall": 0, "f1": 0}

    for sup_label in np.unique(data["True Superlabel"]):
        prec_dict[f"{sup_label}"] = {"pred": 0, "corr": 0}
        rec_dict[f"{sup_label}"] = {"occu": 0, "pred": 0}
        f1_dict[f"{sup_label}"] = {"precision": 0, "recall": 0, "f1": 0}

    return prec_dict, rec_dict, f1_dict


def get_prec_rec(example, idx, **fn_kwargs: dict[str, dict[str, any]]):
    """"""

    # Iterate over the predicted labels and document in prec_dict
    #for pred_label in example["Extracted Label"]:
    #logger.info(f"{pred_label = }")
    logger.info(f"{example['Extracted Label'] = }")
    logger.info((f"{fn_kwargs['prec_dict'] = }"))
    logger.info((f"{fn_kwargs['rec_dict'] = }"))
    logger.info((f"{fn_kwargs['f1_dict'] = }"))
    fn_kwargs["prec_dict"][example["Extracted Label"]]["pred"] += 1
    if example["Extracted Label"] in example["True Label"]:
        fn_kwargs["prec_dict"][example["Extracted Label"]]["corr"] += 1

    # Iterate over the predicted Super labels and document in prec_dict
    #for pred_sup_label in example["Extracted Superlabel"]:
    # Exclude "No fallacy" because it overlaps with the level 2 label
    if example["Extracted Superlabel"] != "No fallacy":
        fn_kwargs["prec_dict"][example["Extracted Superlabel"]]["pred"] += 1
        if example["Extracted Superlabel"] in example["True Superlabel"]:
            fn_kwargs["prec_dict"][example["Extracted Superlabel"]]["corr"] += 1

    # Iterate over the gold labels and document in rec_dict
    #for gold_label in example["True Label"]:
    fn_kwargs["rec_dict"][example["True Label"]]["occu"] += 1
    if example["True Label"] in example["Extracted Label"]:
        fn_kwargs["rec_dict"][example["True Label"]]["pred"] += 1

    # Iterate over the gold Super labels and document in rec_dict
    #for gold_sup_label in example["True Superlabel"]:
    # Exclude "No fallacy" because it overlaps with the level 2 label
    if example["True Superlabel"] != "No fallacy":
        fn_kwargs["rec_dict"][example["True Superlabel"]]["occu"] += 1
        if example["True Superlabel"] in example["Extracted Superlabel"]:
            fn_kwargs["rec_dict"][example["True Superlabel"]]["pred"] += 1

    # Compute precision, recall and F1 per label
    for item in fn_kwargs["prec_dict"].items():
        k = item[0], v = item[1]
        logger.info(f"{k = }")
        logger.info(f"{v = }")
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


def visuals(data_dict: dict, model_name: str, tech: str) -> pd.DataFrame:
    """"""
    # Read data into Pandas df so Seaborn and Matplotlib can handle it \
    # Put each (super) label on a row with the prec, rec and F1 as columns.
    data_df = pd.DataFrame.from_dict(
        data=data_dict,
        orient="index",
        columns=["Precision", "Recall", "F1 score"]
    )

    # Set style for Seaborn plots
    sns.set(style="whitegrid")

    # Precision per label
    plt.figure(figsize=(8, 6))
    sns.barplot(x=data_df.index, y=data_df["Precision"])
    plt.title("Precision per label")
    plt.xticks(rotation=45)
    plt.xlabel("Label")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"temp_plots/{tech}/{model_name}_val_precision.png")

    # Recall per label
    plt.figure(figsize=(8, 6))
    sns.barplot(x=data_df.index, y=data_df["Recall"])
    plt.title("Recall per label")
    plt.xticks(rotation=45)
    plt.xlabel("Label")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(f"temp_plots/{tech}/{model_name}_val_recall.png")

    # F1 score per label
    plt.figure(figsize=(8, 6))
    sns.barplot(x=data_df.index, y=data_df["F1 score"])
    plt.title("F1 score per label")
    plt.xticks(rotation=45)
    plt.xlabel("Label")
    plt.ylabel("F1 score")
    plt.tight_layout()
    plt.savefig(f"temp_plots/{tech}/{model_name}_val_f1.png")

    # Return dataframe so we can output it as a csv
    #return data_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file_path",
        "-dp",
        help="Path to the input data (csv file with tab as separator). For example: 'output/output_data.csv'.",
        required=True,
        type=str,
    )
    #parser.add_argument(
    #   "--output_file_path",
    #   "-out",
    #   required=True,
    #   help="Path where to save the output file.",
    #   type=str,
    #)
    parser.add_argument(
       "--model_name",
       "-m",
       required=True,
       help="The name of the used model to include in the plot savings. For example: T5-base.",
       type=str,
    )
    parser.add_argument(
        "--random_seed",
        "-seed",
        #required=True,
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
    logger.info(f"Loading the data...")
    val_data = get_data(data_path)

    logger.info(f"All unique true labels: {np.unique(val_data['True Label'])}")
    logger.info(f"All unique extracted labels: {np.unique(val_data['Extracted Label'])}")
    logger.info(f"All unique true superlabels: {np.unique(val_data['True Superlabel'])}")
    logger.info(f"All unique extracted superlabels: {np.unique(val_data['Extracted Superlabel'])}")

    logger.info(f"Labels (Level 2):")
    print(
       classification_report(
           y_true=val_data["True Label"],
           y_pred=val_data["Extracted Label"],
           zero_division=0.0
       )
    )
    logger.info(f"Superlabels (Level 1):")
    print(
       classification_report(
           y_true=val_data["True Superlabel"],
           y_pred=val_data["Extracted Superlabel"],
           zero_division=0.0
       )
    )
    exit()

    for tech in np.unique(val_data["Technique"]):
        #print(
        #    classification_report(
        #        y_true=val_data["True Label"],
        #        y_pred=val_data["Extracted Label"],
        #        zero_division=0.0
        #    )
        #)
        #print(
        #    classification_report(
        #        y_true=val_data["True Superlabel"],
        #        y_pred=val_data["Extracted Superlabel"],
        #        zero_division=0.0
        #    )
        #)
        #exit()
        prec_dict, rec_dict, f1_dict = make_dicts(val_data)

        logger.info(f"Computing evaluation scores for technique {tech}...")
        eval_scores = val_data.map(
            get_prec_rec,
            with_indices=True,
            fn_kwargs={
                "prec_dict": prec_dict,
                "rec_dict": rec_dict,
                "f1_dict": f1_dict
            }
        )
        del val_data, prec_dict, rec_dict

        f1_label = eval_scores.pop("total_label_macro", "F1 score of all labels not found.")
        f1_superlabel = eval_scores.pop("total_super_label_macro", "F1 score of all super labels not found.")
        logger.info(f"Total F1 score of the labels for {tech}: {f1_label}.")
        logger.info(f"Total F1 score of the super labels for {tech}: {f1_superlabel}.")

        logger.info(f"Generating the visuals for {tech}...")
        #output_df = visuals(eval_scores, args.model_name)
        visuals(eval_scores, args.model_name, tech)

        # Export dataframe
        #os.makedirs(args.output_file_path, exist_ok=True)
        #logger.info(f"Exporting dataframe to '{args.output_file_path}.csv'...")
        #output_df.to_csv(args.output_file_path + ".csv", index=False, sep='\t')
        #output_df.to_json(args.output_file_path + ".json", orient="records", lines=True)



