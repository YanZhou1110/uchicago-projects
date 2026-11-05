# Review Analytics and Text Similarity with Spark

Explore reviews and product categories, profile reviewers, and compare text similarity patterns across time periods using distributed processing.

**Tools:** PySpark, HashingTF, MinHashLSH, cloud data workflows

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

Replace the local `data/reviews-source` input location with an authorized Parquet source containing `reviews_parquet` and `meta_parquet`. Output paths now use `data/spark-output/`. Raw reviews and cloud credentials are omitted.

## Evaluation and limitations

Run notebooks in numeric order in one persistent kernel; later notebooks reuse earlier variables. Java and a compatible Spark environment are required. Hash-bucket collisions are approximate similarity signals, not proof of fraudulent reviews. Confirm sampling effects before reporting duplicate rates.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
