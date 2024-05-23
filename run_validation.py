import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_dataset():
    data = pd.read_csv("datasets/unified_validation_set_downsampled.tsv", sep="\t")
    data = data.dropna(subset=["MAFALDA Label", "Input"]).sort_values(by=['Input'])
    return data


def read_prompt(prompt_technique, input_text):
    with open(f"prompts/{prompt_technique}.txt", "r", encoding="utf-8") as file:
        prompt = file.read()
        if "Input:" in prompt:
            prompt += input_text.replace('\n', ' ').strip('"')
    return prompt


def generate(model, tokenizer, prompt, max_new_tokens):
    input = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Perhaps we should use different parameters here
    output = model.generate(**input, max_new_tokens=max_new_tokens, num_return_sequences=1)

    # return only the model's output
    decoded_output = tokenizer.decode(output[0])
    generated_response = decoded_output[len(tokenizer.decode(input["input_ids"][0])):]
    return generated_response


def generate_all_models(models, prompt_techniques):
    output_data = read_dataset()
    new_rows = []
    progress = 0
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Perhaps we should use a different model type
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

        for prompt_technique in prompt_techniques:

            max_new_tokens = 64
            if prompt_technique == "ccot":
                max_new_tokens = 512

            for input_text in output_data["Input"]:
                prompt = read_prompt(prompt_technique, input_text)
                output = generate(model, tokenizer, prompt, max_new_tokens)
                print(f"Sample: {progress}, Model: {model_name}, Prompt Technique: {prompt_technique}")
                print(f"Input Text: {input_text}")
                print(f"Output: {output}")
                new_rows.append({
                        "Model": model_name,
                        "Technique": prompt_technique,
                        "Input": input_text,
                        "Output": output
                    })
                pd.DataFrame(new_rows).to_csv("output/output_data_temp.csv", index=False, sep="\t")
                progress += 1
        del model # Nuke from memory just to be sure
        del tokenizer
    return new_rows


if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot"]
    models = ["Salesforce/xgen-7b-8k-base", "lmsys/vicuna-7b-v1.5", "NousResearch/Hermes-2-Pro-Llama-3-8B"]
    new_rows = generate_all_models(models, prompt_techniques)
    pd.DataFrame(new_rows).to_csv("output/output_data.csv", index=False, sep="\t")