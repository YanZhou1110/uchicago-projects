# CartPole Imitation Learning

Behavior-cloning coursework with a human demonstration recorder and an imitation-learning notebook.

## Contents

- `imitation_learning.ipynb`
- `cartpole_human_recorder.py`

## Setup and run

Use a project-specific environment and start from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab imitation_learning.ipynb
```

Dependencies are an inventory, not a tested lockfile. Some notebooks target Colab, GPU runtimes or additional optional packages. Inspect configuration and data paths before running. File paths formerly pointing to a personal computer or Drive were replaced with local `data/` paths.

## Data and results

Record or provide your own demonstrations. Interactive recording requires a local display. Saved demonstrations and trained policy weights are not bundled. Evaluation should distinguish demonstrations, training rollouts and independent evaluation seeds.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
