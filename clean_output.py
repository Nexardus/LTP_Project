"""Clean raw inference output data and merge it with the gold data."""

import argparse

import pandas as pd

from infer_label import LabelExtractor


def read_prompts(prompt_techniques: list[str]) -> dict[str, str]:
    """Read the prompts from the files."""
    prompts = {}

    for technique in prompt_techniques:
        with open(f"./prompts/{technique}.txt", encoding="utf-8") as file:
            prompts[technique] = file.read().strip()

    return prompts


def extract_response(row: pd.DataFrame, prompts: dict[str, str]) -> str:
    """Extract the response from the model output."""
    technique = row["Technique"]
    base_prompt = prompts[technique].strip()
    input_text = row["Input"].replace("\n", "").strip('"')

    output = row["Output"].strip()
    return str(output.replace(base_prompt, "").strip().replace(input_text, "").strip())


def normalize_text(text: str) -> str:
    """Normalize the text by lowercasing and removing extra spaces."""
    return " ".join(text.lower().strip().split())


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        "-g",
        help="Path to the cleaned gold data (csv file with tab as separator). "
        "For example: 'cleaned_datasets/MAFALDA_gold_processed.tsv'.",
        type=str,
        default="cleaned_datasets/unified_validation_set_downsampled.tsv",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Path to the raw output data data (csv file with tab as separator). "
        "For example: 'output/inference_output_21_may.tsv'.",
        type=str,
        default="output/inference_output_21_may.tsv",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to the merged cleaned output file (csv file with tab as separator). "
        "For example: 'cleaned_output/inference_output_cleaned.tsv'.",
        type=str,
        default="cleaned_output/inference_output_cleaned.tsv",
    )
    args = parser.parse_args()

    prompt_techniques = ["gcot", "logicot", "ccot"]
    prompts = read_prompts(prompt_techniques)
    original_df = pd.read_csv(args.dataset, sep="\t")
    original_df["Cleared_output"] = original_df.apply(lambda row: extract_response(row, prompts), axis=1)
    original_df["Original Input"] = original_df["Input"]
    original_df["Input"] = original_df["Input"].apply(normalize_text)

    extractor = LabelExtractor("fallacies.json")

    original_df["Extracted Label"] = original_df.apply(
        lambda row: extractor.extract_label(row["Cleared_output"])[1], axis=1
    )
    original_df["Extracted Superlabel"] = original_df.apply(
        lambda row: extractor.extract_label(row["Cleared_output"])[0], axis=1
    )

    true_labels = pd.read_csv(args.gold, sep="\t")
    true_labels["Input"] = true_labels["Input"].apply(normalize_text)
    true_labels = true_labels.rename(columns={"MAFALDA Label": "True Label", "MAFALDA Superlabel": "True Superlabel"})

    merged_df = original_df.merge(true_labels[["Input", "True Label", "True Superlabel"]], on="Input", how="left")
    merged_df["Input"] = merged_df["Original Input"]
    merged_df = merged_df.drop(columns=["Original Input"])

    # check if there are any NA values in the merged dataframe
    print(merged_df.isna().sum())  # none

    merged_df.to_csv(args.output, index=False, sep="\t")

    models = merged_df["Model"].unique()
    for model in models:
        model_df = merged_df[merged_df["Model"] == model]
        model_df.to_csv(
            f"./cleaned_output/inference_output_{model.replace('/', '_')}_cleaned.tsv", index=False, sep="\t"
        )


if __name__ == "__main__":
    main()
