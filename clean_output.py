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

def normalize_text(text):
    return ' '.join(text.lower().strip().split())

if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot"]
    prompts = read_prompts(prompt_techniques)
    df = pd.read_csv("./output/MAFALDA_gold_output.csv", sep="\t")
    df['Cleared_output'] = df.apply(lambda row: extract_response(row, prompts), axis=1)
    df['Original Input'] = df['Input']
    df['Input'] = df['Input'].apply(normalize_text)

    extractor = LabelExtractor('fallacies.json')

    df['Extracted Label'] = df.apply(lambda row: extractor.extract_label(row["Cleared_output"])[1], axis=1)
    df['Extracted Superlabel'] = df.apply(lambda row: extractor.extract_label(row["Cleared_output"])[0], axis=1)
    # df.to_csv("./output/inference_output_21_may_cleaned.csv", index=False, sep="\t")

    true_labels = pd.read_csv("datasets/MAFALDA_gold_processed.tsv", sep="\t")
    true_labels['Input'] = true_labels['Input'].apply(normalize_text)
    true_labels.rename(columns={"MAFALDA Label": "True Label", "MAFALDA Superlabel": "True Superlabel"}, inplace=True)

    merged_df = df.merge(true_labels[['Input', 'True Label', 'True Superlabel']], on='Input', how='left')
    merged_df['Input'] = merged_df['Original Input']
    merged_df.drop(columns=['Original Input'], inplace=True)

    # check if there are any NA values in the merged dataframe
    print(merged_df.isna().sum()) # none

    models = merged_df['Model'].unique()
    for model in models:
        model_df = merged_df[merged_df['Model'] == model]
        model_df.to_csv(f"./output/inference_output_{model.replace('/', '_')}_cleaned.csv", index=False, sep="\t")