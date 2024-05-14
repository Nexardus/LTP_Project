import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_dataset():
    data = pd.read_csv("datasets/unified_validation_set.tsv", sep="\t")
    data = data.dropna(subset=["MAFALDA Label", "Input"]).sort_values(by=['Input'])
    return data


def read_prompt(prompt_technique, input_text):
    with open(f"prompts/{prompt_technique}.txt", "r", encoding="utf-8") as file:
        prompt = file.read()
        if "Input:" in prompt:
            prompt += input_text
    return prompt


def generate(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    input = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**input, max_length=1024, num_return_sequences=1)
    return tokenizer.decode(output[0])


def generate_all_models(models, prompt_techniques):
    output_data = read_dataset()
    for model in models:
        for prompt_technique in prompt_techniques:
            for input_text in output_data["Input"]:
                prompt = read_prompt(prompt_technique, input_text)
                output = generate(model, prompt)
                print(f"Model: {model}, Prompt Technique: {prompt_technique}")
                print(f"Input Text: {input_text}")
                print(f"Output: {output}")
                output_data[f"{model}_{prompt_technique}"] = output
    return output_data


if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot"]
    models = ["Salesforce/xgen-7b-8k-base", "lmsys/vicuna-7b-v1.5", "NousResearch/Hermes-2-Pro-Llama-3-8B"]
    output_data = generate_all_models(models, prompt_techniques)
    output_data.to_csv("output/output_data.csv", index=False, sep="\t")
