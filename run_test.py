import pandas as pd
from run_validation import read_prompt, generate_all_models

if __name__ == "__main__":
    prompt_techniques = ["gcot", "logicot", "ccot", "multi-agent"]
    models = ["Salesforce/xgen-7b-8k-base", "lmsys/vicuna-7b-v1.5", "NousResearch/Hermes-2-Pro-Llama-3-8B"]
    dataset_path = "datasets/MAFALDA_gold_processed.tsv"
    output_file = "output/MAFALDA_gold_output.csv"
    generate_all_models(models, prompt_techniques, dataset_path, output_file)
