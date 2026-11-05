# Customer Marketing Response Analysis

Analyze customer spending, purchasing channels and campaign response, then build a logistic regression baseline.

**Tools:** Python, pandas, exploratory analysis, Logistic Regression

## Project status

Coursework source organized for review. Notebook outputs have been cleared and personal file paths removed. The original modeling logic is preserved; this preparation does not certify an end-to-end rerun or establish sole authorship of team work.

## Run

Use Python 3.10 or 3.11 and start in this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab customer_analysis.ipynb
```

Dependencies are an unpinned inventory from source imports, not a tested lockfile. Heavy models may require substantial downloads. Optional notebook sections may install extra packages.

## Data and artifacts

Place the original tab-separated `marketing_campaign.csv` in `data/`. The source notebook describes 2,240 customers. Raw records are omitted pending source and redistribution review.

## Evaluation and limitations

This is a coursework analysis. Review feature timing, split strategy, imputation and response imbalance before treating its evaluation as a generalizable marketing model. Statistical associations do not establish campaign causality.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
