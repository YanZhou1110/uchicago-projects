# Yoga Pose Recognition and Deployment

Coursework covering custom CNNs, transfer learning, feature clustering, pose landmarks and TensorFlow Lite deployment for 107 yoga classes.

## Contents

- `01_yoga_cnn_experiments.ipynb`
- `02_yoga_transfer_learning.ipynb`
- `app.py`
- `classes.json`

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

The source notebooks document Kaggle data acquisition and cleaned splits. The second notebook depends on previously generated experiment artifacts and is not a standalone training pipeline. The Streamlit app requires `models/efficientnet_fine_tuned_float16.tflite` and `models/efficientnet_head_float16.tflite`; supply them from your own authorized training outputs and run `streamlit run app.py`. Images, artifacts and weights are omitted. Pose accuracy is conditional on successful detection where indicated in the notebook.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
