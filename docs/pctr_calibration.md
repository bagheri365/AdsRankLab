# pCTR calibration experiment

The baseline ranking model is trained on June 6–10. Calibration parameters are
fit **only on June 11 validation predictions**. June 12 remains untouched until
final evaluation.

Three probability versions are compared using the same underlying ranking
model:

1. **Raw pCTR** — unmodified logistic-regression probabilities.
2. **Intercept-only recalibration** — adds one constant to raw log-odds so the
   validation mean prediction matches validation CTR. This transformation is
   strictly monotonic and preserves ranking.
3. **Platt scaling** — fits an affine transformation of raw logits using the
   validation labels, allowing both calibration slope and intercept to change.

## Test metrics

Each method is evaluated on June 12 using ROC-AUC, PR-AUC, log loss, Brier
score, ECE, mean predicted CTR, and predicted/observed CTR ratio. Advertiser-
level calibration summaries are also written for all three methods.

The experiment explicitly records whether test labels were used for calibration
(the expected value is `false`).

## Run

```bash
python -m ads_rank_lab.experiments.pctr_calibration
```

Outputs:

```text
artifacts/results/pctr_calibration_season2.json
artifacts/results/pctr_calibration_season2_advertisers.csv
```
