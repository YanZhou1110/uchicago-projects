# Qwen LoRA Fine Tuning Coursework

A parameter-efficient fine-tuning notebook using Qwen and low-rank adapters.

## Contents

- `qwen_lora.ipynb`

## Setup and run

Use a project-specific environment and start from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab qwen_lora.ipynb
```

Dependencies are an inventory, not a tested lockfile. Some notebooks target Colab, GPU runtimes or additional optional packages. Inspect configuration and data paths before running. File paths formerly pointing to a personal computer or Drive were replaced with local `data/` paths.

## Data and results

GPU and dataset/model access may be required. Adapter weights and training data are not included. The notebook may include environment-specific installation cells; inspect model identifiers and package compatibility before running. No independently verified fine-tuning improvement is claimed.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
