# SEC 10-K Risk Factor Classification

Classify financial risk disclosures into six categories and compare classical NLP, transformer embeddings, and a local prompt-based challenger.

**Tools:** Python, TF-IDF, Logistic Regression, FinBERT, Qwen

## Project status

Coursework source organized for review. Notebook outputs have been cleared and personal file paths removed. The original modeling logic is preserved; this preparation does not certify an end-to-end rerun or establish sole authorship of team work.

## Run

Use Python 3.10 or 3.11 and start in this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python risk_classification.py
```

Dependencies are an unpinned inventory from source imports, not a tested lockfile. Heavy models may require substantial downloads. Optional notebook sections may install extra packages.

## Data and artifacts

Place the original annotation dataset at `labels/annotation_dataset.csv` and the gold examples at `example_labels/annotated_dataset_labeled.csv`. Inputs are omitted pending redistribution review.

## Evaluation and limitations

The saved model comparison reports accuracy 0.734 and macro F1 0.721 for TF-IDF + Logistic Regression. Most full-corpus labels were model-assisted, so this is not an independent human-gold benchmark. The local Qwen challenger used 60 examples; comparisons require matching evaluation sets. Rerunning the pipeline may regenerate labels and results.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
