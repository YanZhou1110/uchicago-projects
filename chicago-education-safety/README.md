# Chicago Education and Safety Visualization

Related course analyses exploring Chicago schools, community characteristics and public safety data.

## Contents

- `01_analysis.ipynb`
- `02_education.ipynb`
- `03_analysis.ipynb`

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

Provide the original school, crime and community boundary inputs. Raw crime records and student/community datasets are omitted. Associations across communities do not establish causal effects, individual attributes or school quality rankings.

This source-only collection omits raw records, user uploads, personal calendar events, credentials, caches, checkpoints and model weights. Notebook outputs and embedded attachments were cleared. Historical narrative results in the original notebook have not been independently reproduced; no new performance claim is made by this upload.

## Attribution and scope

Organized from Yan Zhou's graduate coursework. Original course prompts and inline attributions are retained where present; their inclusion is not a claim of sole authorship. Third-party data, pretrained models and libraries remain subject to their original terms. No new license is assigned to third-party material.
