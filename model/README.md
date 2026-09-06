# Model Artifacts

- `stress_model.joblib` — hybrid ensemble (TF-IDF text + numeric LIWC branch)
- `metrics.json` — training metrics, thresholds, candidate comparison
- `drift_profile.json` — baseline population statistics for drift detection

## Persian model

Place `stress_model_persian.joblib` and `metrics_persian.json` in `model/persian/` after running:

```bash
python train_persian.py --train data/persian-train.csv --test data/persian-test.csv --output model/persian
```
