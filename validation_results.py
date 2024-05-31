import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from datasets import Dataset, load_dataset
import logging
import warnings
import os
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
    # Lowercase all the columns
    val_data = val_data.map(lambda example: {"True Label": example["True Label"].lower()})
    val_data = val_data.map(lambda example: {"True Superlabel": example["True Superlabel"].lower()})
    val_data = val_data.map(lambda example: {"Extracted Label": example["Extracted Label"].lower()})
    val_data = val_data.map(lambda example: {"Extracted Superlabel": example["Extracted Superlabel"].lower()})
    val_data = val_data.map(lambda example: {"True Superlabel": "appeal to emotion"} if example["True Superlabel"] == "fallacy of emotion" else {"True Superlabel": example["True Superlabel"]})

    #logger.info(f"All unique true labels: {np.unique(val_data['True Label'])}")
    #logger.info(f"All unique extracted labels: {np.unique(val_data['Extracted Label'])}")
    logger.info(f"Predicted non-gold labels: {set(np.unique(val_data['Extracted Label'])).difference(np.unique(val_data['True Label']))}")
    logger.info(f"Non-predicted gold labels: {set(np.unique(val_data['True Label'])).difference(np.unique(val_data['Extracted Label']))}")

    #logger.info(f"All unique true superlabels: {np.unique(val_data['True Superlabel'])}")
    #logger.info(f"All unique extracted superlabels: {np.unique(val_data['Extracted Superlabel'])}")
    logger.info(f"Predicted non-gold superlabels: {set(np.unique(val_data['Extracted Superlabel'])).difference(np.unique(val_data['True Superlabel']))}")
    logger.info(f"Non-predicted gold superlabels: {set(np.unique(val_data['True Superlabel'])).difference(np.unique(val_data['Extracted Superlabel']))}")

    clas_rep_label = classification_report(
        y_true=val_data["True Label"],
        y_pred=val_data["Extracted Label"],
        zero_division=0.0,
        output_dict=True
    )
    logger.info(f"Labels (Level 2) overall:")
    print(
        classification_report(
            y_true=val_data["True Label"],
            y_pred=val_data["Extracted Label"],
            zero_division=0.0
        )
    )
    clas_rep_superlabel = classification_report(
        y_true=val_data["True Superlabel"],
        y_pred=val_data["Extracted Superlabel"],
        zero_division=0.0,
        output_dict=True
    )
    logger.info(f"Superlabels (Level 1) overall:")
    print(
        classification_report(
            y_true=val_data["True Superlabel"],
            y_pred=val_data["Extracted Superlabel"],
            zero_division=0.0
        )
    )

    for tech in np.unique(val_data["Technique"]):
        temp_data = val_data.filter(lambda x: x["Technique"] == tech)
        logger.info(f"Labels (Level 2) for technique {tech}:")
        print(
            classification_report(
                y_true=temp_data["True Label"],
                y_pred=temp_data["Extracted Label"],
                zero_division=0.0
            )
        )
        logger.info(f"Superlabels (Level 1) for technique {tech}:")
        print(
            classification_report(
                y_true=temp_data["True Superlabel"],
                y_pred=temp_data["Extracted Superlabel"],
                zero_division=0.0
            )
        )



