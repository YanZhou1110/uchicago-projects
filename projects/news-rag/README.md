# News Retrieval and Grounded Answers

Build a news retrieval pipeline with chunking, embedding comparisons, hybrid retrieval, reranking, citations and evaluation.

**Tools:** Python, Sentence Transformers, retrieval, Streamlit

## Project status

Coursework source organized for review. Notebook outputs have been cleared and personal file paths removed. The original modeling logic is preserved; this preparation does not certify an end-to-end rerun or establish sole authorship of team work.

## Run

Use Python 3.10 or 3.11 and start in this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab news_rag.ipynb
```

Dependencies are an unpinned inventory from source imports, not a tested lockfile. Heavy models may require substantial downloads. Optional notebook sections may install extra packages.

## Data and artifacts

Use an authorized news corpus at `data/news_articles.csv`, or inspect the notebook acquisition and fallback modes. Run the notebook to generate `assignment3_outputs/` before starting the app with `streamlit run app.py`.

## Evaluation and limitations

This local submission differs from the notebook currently on GitHub. Its fallback can use AG News when live scraping fails; a fallback is not a recent-news scrape. The Streamlit app uses extractive answers. Notebook generation and evaluation modes depend on installed models and optional credentials. Always record actual corpus provenance, generator mode and metric implementation. No accuracy or faithfulness claim is made here.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
