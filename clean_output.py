import pandas as pd

from infer_label import LabelExtractor


def read_prompts(prompt_techniques):
    prompts = {}

    for technique in prompt_techniques:
        with open(f"./prompts/{technique}.txt", "r", encoding="utf-8") as file:
            prompts[technique] = file.read().strip()

    return prompts

def extract_response(row, prompts):
    technique = row['Technique']
    base_prompt = prompts[technique].strip()
    input_text = row['Input'].replace('\n', '').strip('"')

    output = row['Output'].strip()
    response = output.replace(base_prompt, '').strip().replace(input_text, '').strip()

    return response

if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot"]
    prompts = read_prompts(prompt_techniques)
    df = pd.read_csv("./output/inference_output_21_may.csv", sep="\t")
    df['Cleared_output'] = df.apply(lambda row: extract_response(row, prompts), axis=1)

    extractor = LabelExtractor('fallacies.json')

    df['Extracted Label'] = df.apply(lambda row: extractor.extract_label(row["Cleared_output"])[1], axis=1)
    df['Extracted Superlabel'] = df.apply(lambda row: extractor.extract_label(row["Cleared_output"])[0], axis=1)
    # df.to_csv("./output/inference_output_21_may_cleaned.csv", index=False, sep="\t")

    true_labels = pd.read_csv("datasets/unified_validation_set_downsampled.tsv", sep="\t")
    true_labels.rename(columns={"MAFALDA Label": "True Label", "MAFALDA Superlabel": "True Superlabel"}, inplace=True)
    merged_df = df.merge(true_labels[['Input', 'True Label', 'True Superlabel']], on='Input', how='left')

    models = merged_df['Model'].unique()
    for model in models:
        model_df = merged_df[merged_df['Model'] == model]
        model_df.to_csv(f"./output/inference_output_{model.replace('/', '_')}_cleaned.csv", index=False, sep="\t")