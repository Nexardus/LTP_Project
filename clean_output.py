import pandas as pd

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

    df.to_csv("./output/inference_output_21_may_cleaned.csv", index=False, sep="\t")
