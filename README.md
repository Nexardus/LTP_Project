# Few-Shot in the Dark: On the Reliability of Various Chain of Thought Prompting Strategies for Fallacy Detection with Open-Source Large Language Models

## Installation and requirements
This project uses Python 3.11. Dependencies are specified in `requirements.txt`.

### Code quality
We use `ruff format`, `ruff check`, and `mypy *.py` to format and check our code. See `pyproject.toml` for more information.

## Datasets
- [Argotario](https://github.com/UKPLab/argotario), [paper](https://www.aclweb.org/anthology/D17-2002)
- [ElecDeb60to20](https://github.com/pierpaologoffredo/ElecDeb60to20), [paper](https://aclanthology.org/2023.emnlp-main.684.pdf)
- [LogicClimate](https://github.com/causalNLP/logical-fallacy), [paper](https://arxiv.org/abs/2202.13758)
- [LogicEdu](https://github.com/causalNLP/logical-fallacy), [paper](https://arxiv.org/abs/2202.13758)
- [MAFALDA](https://github.com/ChadiHelwe/MAFALDA), [paper](https://arxiv.org/abs/2311.09761)

These datasets are included in the `datasets` folder for convenience.

## Preprocessing
To clean and preprocess the validation datasets, you can run `preprocess_unified_validation_set.py`. This script will read the original datasets from the `datasets` folder and output two unified TSV-formatted files in `cleaned_datasets` (one complete, and one equally downsampled).

For the MAFALDA test set, you can run `preprocess_mafalda.py`. 

## Dataset analysis
Running `dataset_analysis.py` on a given **cleaned** TSV-formatted dataset will output and plot some statistics about the dataset. These plots are used in our paper as well.

## Running on Habrok (RUG university cluster)
We have included the scripts used for running GPU-accelerated inference on Habrok in the `jobscripts` folder. These can be run through e.g., `sbatch jobscripts/run_inference_test.sh`.
In addition, the output from our runs are included in the `output` folder.

These outputs can be cleaned with `clean_output.py`, and will be stored in `cleaned_output`. 
This cleaned output can be analyzed with `prediction_analysis.py`, which will output some statistics and plots about the model's preferences of certain classes.
Finally, metric scores can be calculated with `calculate_scores.py`.

## Reproducing the pipeline
The command line argument defaults are set to reproduce the pipeline for the downsampled validation set as described in the paper. The following steps are required to reproduce both the validation set and the MAFALDA results (after having installed the requirements):
1. `python3 preprocess_unified_validation_set.py`
2. `python3 preprocess_mafalda.py`
3. `python3 dataset_analysis.py`
4. `sbatch jobscripts/run_inference_validation.sh`
5. `sbatch jobscripts/run_inference_test.sh`
6. `python3 clean_output.py`
7. `python3 clean_output.py -g cleaned_datasets/MAFALDA_gold_processed.tsv -d output/MAFALDA_gold_output.tsv -o cleaned_output/MAFALDA_gold_output_cleaned.tsv`
8. `python3 prediction_analysis.py`
8. `python3 calculate_scores.py`
9. `python3 calculate_scores.py -g -d cleaned_output/MAFALDA_gold_output_cleaned.tsv`