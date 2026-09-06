"""
Persian stress model scaffolding (ParsBERT / multilingual fine-tune).

This is a PLACEHOLDER pipeline. To activate:
1. Collect labeled Persian stress data (CSV: text,label).
2. Place it in data/persian-train.csv and data/persian-test.csv.
3. Run: python train_persian.py --train data/persian-train.csv --test data/persian-test.csv --output model/persian

The current production model is English-only (Dreaddit). Persian input
falls back to support_only mode until this model is trained and validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


def train_persian(train_path: Path, test_path: Path, output_dir: Path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    text_col = "text"
    label_col = "label"

    X_train = train[text_col].fillna("").astype(str)
    y_train = train[label_col].astype(int)
    X_test = test[text_col].fillna("").astype(str)
    y_test = test[label_col].astype(int)

    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, analyzer="char_wb")),
        ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "persian_tfidf_logreg",
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probas)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "note": "Placeholder model. Replace with ParsBERT fine-tune for production.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "stress_model_persian.joblib")
    (output_dir / "metrics_persian.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="model/persian")
    args = parser.parse_args()
    train_persian(Path(args.train), Path(args.test), Path(args.output))
