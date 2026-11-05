# Music Descriptor Embeddings and Recommendation

Explore album descriptors, build latent music representations, and retrieve similar albums. A companion notebook explores supervised genre classification.

**Tools:** Python, weighted TF-IDF, Truncated SVD, cosine similarity

## Project status

Coursework source organized for review. Notebook outputs have been cleared and personal file paths removed. The original modeling logic is preserved; this preparation does not certify an end-to-end rerun or establish sole authorship of team work.

## Run

Use Python 3.10 or 3.11 and start in this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Dependencies are an unpinned inventory from source imports, not a tested lockfile. Heavy models may require substantial downloads. Optional notebook sections may install extra packages.

## Data and artifacts

Provide the original `rym_clean1.csv` in this project folder. Album metadata is omitted pending source and redistribution review.

## Evaluation and limitations

The recommendation notebook includes mood proportions, t-SNE and artist trajectories. The genre notebook fits text features before cross-validation, so its validation scores need a leakage review. Its final full-data classification report is training performance, not held-out performance.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
