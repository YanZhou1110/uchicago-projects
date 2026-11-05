# Explainable Chest Image Classification Prototype

Compare a baseline CNN and fine-tuned ResNet18 on PneumoniaMNIST, then demonstrate predictions and Grad-CAM overlays in a Gradio interface.

**Tools:** PyTorch, ResNet18, transfer learning, Grad-CAM, Gradio

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

Training uses the MedMNIST downloader. Run the training notebook from `Data & Models/` to produce the two checkpoints, then start `python "App Demo/app.py"` from this project folder. Dataset images and model weights are omitted.

## Evaluation and limitations

An educational prototype, not a clinically validated system. Confidence thresholds in the app are demonstration heuristics and do not establish calibrated clinical risk. Original notebooks include Colab-specific download cells. Team member contributions and final model results should be confirmed before highlighting them in a portfolio.

## Attribution

Prepared from Yan Zhou's course project materials. Original inline attributions are retained. Team authorship and third-party reuse permissions have not been independently established. No new license is granted for course data or third-party code.
