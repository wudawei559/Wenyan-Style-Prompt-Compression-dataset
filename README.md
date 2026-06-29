# Wenyan-Style Prompt Compression Dataset

This repository contains supplementary materials and anonymized data for the study:

**Wenyan-Style Prompt Compression for Chinese LLM Interaction: Token Cost, Task Quality, and User Evaluation**

The study compares original prompts, unconstrained compressed prompts, and Wenyan-style compressed prompts in Chinese LLM interaction. It examines token cost, model output quality, and user evaluation.

## Repository contents

* `openai_experiment_outputs.zip`
  Model outputs and token-usage records for the OpenAI model.

* `gemini_experiment_outputs.zip`
  Model outputs and token-usage records for the Gemini model.

* `doubao_experiment_outputs.zip`
  Model outputs and token-usage records for the Doubao model.

* `Score sheet.zip`
  Human quality rating sheets and related scoring materials. Rater identities have been anonymized.

* `Survey.zip`
  User questionnaire materials and anonymized questionnaire data.

* `openai_batch_run_tokens_with_txt_py39_explicit_params.py`
  Batch-running script used for the OpenAI model.

* `doubao_prompts15_grouped_from_excel.py`
  Batch-running script used for the Doubao model.

* `gemini_batch_run_tokens_with_txt_py39.py`
  Example/template batch-running script for the Gemini model. Users may need to adjust file names, prompt inputs, API configuration, and local paths before running it.

* self_contained_prompts_15.xlsx
  All the prompts are here.
## Data description

The dataset includes materials related to 15 Chinese prompts across three input conditions:

1. original prompt;
2. unconstrained compressed prompt;
3. Wenyan-style compressed prompt.

The model-output data include prompt identifiers, input conditions, model names, repeated runs, token-usage records, and generated outputs. The human-rating data include quality scores for task completion, information retention, clarity, and translation fidelity where applicable. The questionnaire data include anonymized user responses related to prompt preference, clarity, acceptability, willingness to use, and output preference.

## Privacy and anonymization

The questionnaire data were anonymized before sharing. Directly identifying information such as names, phone numbers, email addresses, IP addresses, and precise submission metadata was removed where applicable. Raters are identified only by anonymized labels.

## Code notes

The scripts are provided to document the batch execution workflow used in the study. API keys and account-specific credentials are not included. Users who wish to run the scripts need to configure their own API credentials and may need to adapt model names, file paths, prompt files, and output directories.

The scripts are intended mainly for transparency and reproducibility of the experimental workflow. Exact reproduction of model outputs may not be possible because commercial LLMs, APIs, tokenizers, and model versions may change over time.

