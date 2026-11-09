# Does the Scorer Widen the Gap? SES Fairness in Automated Essay Scoring

DS 6050 Deep Learning (UVA, Fall 2026) course project.

We test whether neural automated essay scoring (AES) models systematically under-score economically
disadvantaged students (the population served by Title I) beyond the gap already present in human scores,
using the public PERSUADE 2.0 corpus.

## Data

PERSUADE 2.0 (Crossley et al., 2024), CC BY 4.0.
- Source: https://github.com/scrosseye/persuade_corpus_2.0
- File used: `persuade_corpus_2.0_train.csv` (Google Drive id `13phHyDzIsb0MHyJr6q-B-qIa9P2tM135`,
  last modified 2024-11-14, 616,894,963 bytes). The script downloads it automatically with `gdown`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run from the `code/` folder. All outputs go to the Overleaf project root passed as `--paper_dir`
(`results/*.tex`, `figs/*.pdf`), which `main.tex` picks up automatically.

```bash
# 1. Baselines (CPU, ~5 min): handcrafted features + Ridge, TF-IDF + Ridge
python run_experiments.py --stage features --paper_dir ..

# 2. DeBERTa-v3 regressor (GPU; ~20-30 min on a Colab T4 for deberta-v3-small)
python run_experiments.py --stage transformer --paper_dir .. --model microsoft/deberta-v3-small

# Rebuild tables/figures only
python run_experiments.py --stage report --paper_dir ..
```

## Reproducibility

- Fixed seed 42; train/val/test = 80/10/10, stratified by prompt x score; split saved to `results/splits.csv`.
- Fairness metric: difference in mean signed error (prediction minus human score) between disadvantaged and
  non-disadvantaged students on the test set, with a 1,000-sample bootstrap 95% CI.
