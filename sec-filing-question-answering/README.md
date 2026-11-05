# SEC Filing Question Answering

A companion to the SEC risk classification project: interactive financial filing QA and a notebook comparing retrieval augmentation with full-document prompting.

## Contents

- `app.py`
- `section_recovery.py`
- `evaluation.ipynb`

## Setup and run

Use a project-specific environment and start from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py -- --corpus_dir ./data/sec_corpus_clean
```

Dependencies are an inventory, not a tested lockfile. Some notebooks target Colab, GPU runtimes or additional optional packages. Inspect configuration and data paths before running. File paths formerly pointing to a personal computer or Drive were replaced with local `data/` paths.

## Data and results

Provide authorized cleaned filing sections with ticker, fiscal year and section identifiers in the expected filenames. The lightweight app and GPU evaluation notebook have different configurations. The notebook expects a local/Drive corpus, a stub manifest and manually verified reference answers; configure these before evaluation. An optional API judge may incur usage charges. Code does not establish that all reference answers or reported evaluations are final.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
