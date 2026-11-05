# Tabular Machine Learning Coursework

Selected submissions on dimensionality reduction, airline analytics, fraud detection, Titanic survival and loan prediction.

## Contents

- `dimensionality_reduction.ipynb`
- `airlines.ipynb`
- `fraud_detection.ipynb`
- `titanic/01_titanic_train_save.py`
- `titanic/02_titanic_load_apply.py`
- `loan/01_loan_train_save.py`
- `loan/02_loan_load_apply.py`

## Setup and run

Use a project-specific environment and start from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Dependencies are an inventory, not a tested lockfile. Some notebooks target Colab, GPU runtimes or additional optional packages. Inspect configuration and data paths before running. File paths formerly pointing to a personal computer or Drive were replaced with local `data/` paths.

## Data and results

Each notebook or train/apply script has its own original data schema and local file requirements. Provide the original authorized datasets. Loan/fraud models are educational exercises and have not been assessed for real-world eligibility or financial decisions. Review split design, preprocessing and class imbalance before interpreting results.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
