import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_dataset(dataset_path: str):
    data = pd.read_csv(dataset_path, sep="\t")
    print(f"Column names: {data.columns}")
    data = data.dropna(subset=["MAFALDA Label", "Input"]).sort_values(by=['Input'])
    return data


def read_prompt(prompt_technique, input_text):
    with open(f"prompts/{prompt_technique}.txt", "r", encoding="utf-8") as file:
        prompt = file.read()
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


def generate_all_models(models, prompt_techniques, dataset_path, output_file):
    output_data = read_dataset(dataset_path)
    new_rows = []
    progress = 0
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Perhaps we should use a different model type
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

        for prompt_technique in prompt_techniques:

            max_new_tokens = 64

            if prompt_technique in ["ccot", "multi-agent"]:
                max_new_tokens = 512

            for input_text in output_data["Input"]:
                # In this approach the model has two roles: the reasoner, and the checker
                if prompt_technique == "multi-agent":
                    print(progress)
                    # Load the logicot prompt as the initial prompt for the reasoner and feed it into the model
                    prompt_model1 = read_prompt("logicot", input_text)
                    output1 = generate(model, tokenizer, prompt_model1, max_new_tokens)

                    # Load the prompt for the checker and the output from the reasoner to feed into the model
                    prompt_model2 = read_prompt(prompt_technique+"-checker", output1)
                    output2 = generate(model, tokenizer, prompt_model2, max_new_tokens)

                    if output2 == "OK":     # The checker concluded that the fallacy classification was okay
                        # Use this reasoning as the final answer
                        output = output1
                    else:                   # The checker concluded that the fallacy classification was not okay
                        # Generate a new response
                        prompt_model1_interaction1 = read_prompt(prompt_technique+"-reasoner-regenerate1", output1)
                        prompt_model1_interaction2 = read_prompt(prompt_technique+"-reasoner-regenerate2", output2)
                        prompt_interaction = prompt_model1_interaction1 + "\n" + prompt_model1_interaction2
                        output1_regenerated = generate(model, tokenizer, prompt_interaction, max_new_tokens)

                        # Use this latest response as the final answer
                        output = output1_regenerated
                else:
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

    pd.DataFrame(new_rows).to_csv(output_file, index=False, sep="\t")


if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot", "multi-agent"]
    models = ["Salesforce/xgen-7b-8k-base", "lmsys/vicuna-7b-v1.5", "NousResearch/Hermes-2-Pro-Llama-3-8B"]
    generate_all_models(models,
                        prompt_techniques,
                        "cleaned_datasets/unified_validation_set_downsampled.tsv",
                        "output/output_data.csv")

    # TODO make this use CLI params and include configuration for run_test