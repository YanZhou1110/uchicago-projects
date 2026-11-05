# Air Quality and Health Time Series Forecasting

Compare naive, ETS, SARIMA and gradient boosting approaches to forecasting respiratory admissions, including a multi-region three-day horizon.

**Tools:** Python, statsmodels, Gradient Boosting, TimeSeriesSplit

## Project status

Coursework source organized for review. Notebook outputs have been cleared and personal file paths removed. The original modeling logic is preserved; this preparation does not certify an end-to-end rerun or establish sole authorship of team work.

## Run

Use Python 3.10 or 3.11 and start in this project folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab forecasting.ipynb
```

Dependencies are an unpinned inventory from source imports, not a tested lockfile. Heavy models may require substantial downloads. Optional notebook sections may install extra packages.

## Data and artifacts

The notebook contains its original KaggleHub dataset download reference. Dataset availability and access requirements depend on the original source.

## Evaluation and limitations

Historical notebook commentary is preserved, but forecast metrics have not been independently rerun. Verify horizon alignment and chronological validation separately for each region. Classroom error reductions do not demonstrate actual bed or cost savings.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
