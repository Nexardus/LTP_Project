# Few-Shot in the Dark: On the Reliability of Various Chain of Thought Prompting Strategies for Fallacy Detection with Open-Source Large Language Models

## Installation and requirements
This project uses Python 3.11. Dependencies are specified in `requirements.txt`.

### Code quality
We use `ruff format`, `ruff check`, and `mypy *.py` to format and check our code. See `pyproject.toml` for more information.

## Datasets
- [Argotario](https://github.com/UKPLab/argotario), [paper](www.aclweb.org/anthology/D17-2002)
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
We have included the scripts used for running GPU-accelerated inference on Habrok in the `jobscripts` folder. In addition, the output from our runs are included in the `output` folder.

These outputs can be cleaned with `clean_output.py`, and will be stored in `cleaned_output`. 
This cleaned output can be analyzed with `prediction_analysis.py`, which will output some statistics and plots about the model's preferences of certain classes.
Finally, metric scores can be calculated with `calculate_metrics.py`.