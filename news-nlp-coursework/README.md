# News NLP Collection and Classification

Earlier news NLP assignments covering corpus collection and text classification, complementing the separate News RAG project.

## Contents

- `01_news_collection.ipynb`
- `collection_app.py`
- `02_news_classification.ipynb`
- `classification_app.py`

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

Live article availability and extraction can change. Respect source access and reuse terms. Generate the required output files in the notebooks before starting the relevant Streamlit app. Corpus files, article text and saved models are excluded; record any fallback dataset explicitly.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
