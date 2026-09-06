"""
Hybrid model trainer for AramNegar v5.

Model = VotingClassifier over three members:
  1. Text branch  : TF-IDF(1-2 gram) + LogisticRegression   (calibrated)
  2. Text branch  : TF-IDF(1-2 gram) + Calibrated LinearSVC
  3. Numeric branch: StandardScaler + LogisticRegression on the
                     full 109-column Dreaddit numeric feature set
                     (LIWC + DAL + social + syntax + meta)

Both branches are trained on the same rows. The numeric extractor is run
on the raw text column, so any CSV that has a text column can be
re-trained without the original LIWC columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

from stressguard.features import ALL_NUMERIC_COLUMNS, build_feature_vector


# ── column resolution ───────────────────────────────────────────────
def resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    text_col = next((c for c in ["text", "post", "body", "content"] if c in df.columns), df.columns[0])
    label_col = next((c for c in ["label", "stress", "target"] if c in df.columns), df.columns[-1])
    return text_col, label_col


# ── hybrid matrix builder ───────────────────────────────────────────
def build_hybrid_matrix(df: pd.DataFrame, text_col: str, label_col: str):
    """Return (X_text, X_numeric, y).

    If the CSV already has the numeric columns (Dreaddit format) we use
    them; otherwise we extract them from the text so any plain-text CSV
    can be retrained through the same pipeline.
    """
    X_text = df[text_col].fillna("").astype(str).tolist()
    y = df[label_col].astype(int).to_numpy()

    if all(c in df.columns for c in ALL_NUMERIC_COLUMNS):
        X_numeric = df[ALL_NUMERIC_COLUMNS].fillna(0.0).astype(np.float64).to_numpy()
    else:
        X_numeric = np.vstack([build_feature_vector(t) for t in X_text])

    return X_text, X_numeric, y


# ── model builders ──────────────────────────────────────────────────
def build_models(class_weight: dict, calibrate_cv: int = 3) -> dict:
    text_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, solver="liblinear")),
        ]
    )
    svm_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=70000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight=class_weight), cv=calibrate_cv)),
        ]
    )
    numeric_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, solver="liblinear")),
        ]
    )
    return {
        "text_logreg": text_clf,
        "text_svm_calibrated": svm_clf,
        "numeric_logreg": numeric_clf,
    }


# ── training helpers ────────────────────────────────────────────────
def _fit_member(name: str, pipeline, X_text, X_numeric, y_train, sample_weight=None):
    X = X_text if name.startswith("text_") else X_numeric
    if sample_weight is None:
        pipeline.fit(X, y_train)
    else:
        # pipeline ends in "clf"
        pipeline.fit(X, y_train, clf__sample_weight=sample_weight)
    return pipeline


def _predict_proba_member(name: str, model, X_text, X_numeric):
    X = X_text if name.startswith("text_") else X_numeric
    return model.predict_proba(X)[:, 1]


def _predict_label_member(name: str, model, X_text, X_numeric, threshold: float = 0.5):
    return (_predict_proba_member(name, model, X_text, X_numeric) >= threshold).astype(int)


def train_and_evaluate(train_df, test_df, output_dir: Path | None = None, sample_weight=None):
    text_col, label_col = resolve_columns(train_df)

    X_text_train, X_numeric_train, y_train = build_hybrid_matrix(train_df, text_col, label_col)
    X_text_test, X_numeric_test, y_test = build_hybrid_matrix(test_df, text_col, label_col)

    classes = np.array(sorted(np.unique(y_train)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    min_class_count = int(np.bincount(y_train).min())
    calibrate_cv = max(2, min(3, min_class_count)) if min_class_count >= 2 else 2

    results = []
    fitted_members = {}

    for name, pipeline in build_models(class_weight, calibrate_cv=calibrate_cv).items():
        fitted = _fit_member(name, clone(pipeline), X_text_train, X_numeric_train, y_train, sample_weight=sample_weight)
        probas = _predict_proba_member(name, fitted, X_text_test, X_numeric_test)
        preds = (probas >= 0.48).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        result = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc_score(y_test, probas),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }
        results.append(result)
        fitted_members[name] = fitted

    # soft-voting ensemble (equal weights — the release gate decides
    # whether the ensemble beats the best single member)
    ensemble_proba = np.mean(
        [
            _predict_proba_member("text_logreg", fitted_members["text_logreg"], X_text_test, X_numeric_test),
            _predict_proba_member("text_svm_calibrated", fitted_members["text_svm_calibrated"], X_text_test, X_numeric_test),
            _predict_proba_member("numeric_logreg", fitted_members["numeric_logreg"], X_text_test, X_numeric_test),
        ],
        axis=0,
    )
    ensemble_pred = (ensemble_proba >= 0.48).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, ensemble_pred, average="binary", zero_division=0)
    ensemble_result = {
        "model": "voting_ensemble",
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, ensemble_proba),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred).tolist(),
        "classification_report": classification_report(y_test, ensemble_pred, output_dict=True, zero_division=0),
    }
    results.append(ensemble_result)

    # pick best by F1
    best = max(results, key=lambda x: x["f1"])
    best_name = best["model"]

    # pack everything into a single artifact
    artifact = {
        "members": fitted_members,
        "selected_model": best_name,
        "thresholds": {"low": 0.35, "high": 0.60, "decision": 0.48},
    }

    # drift profile uses ensemble probabilities (the production inference path)
    train_proba = np.mean(
        [
            _predict_proba_member("text_logreg", fitted_members["text_logreg"], X_text_train, X_numeric_train),
            _predict_proba_member("text_svm_calibrated", fitted_members["text_svm_calibrated"], X_text_train, X_numeric_train),
            _predict_proba_member("numeric_logreg", fitted_members["numeric_logreg"], X_text_train, X_numeric_train),
        ],
        axis=0,
    )
    drift_profile = build_drift_profile(X_text_train, train_proba, artifact["thresholds"])

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "columns": list(train_df.columns),
        "text_column": text_col,
        "label_column": label_col,
        "train_label_distribution": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
        "test_label_distribution": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        "class_weight": class_weight,
        "candidates": results,
        "selected_model": best_name,
        "selection_metric": "binary_f1",
        "thresholds": artifact["thresholds"],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output_dir / "stress_model.joblib")
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "drift_profile.json").write_text(json.dumps(drift_profile, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary, artifact, drift_profile


def build_drift_profile(texts, probabilities, thresholds):
    token_lengths = [len(str(text).split()) for text in texts]
    triage = {"low": 0, "uncertain": 0, "elevated": 0}
    for p in probabilities:
        if p >= thresholds["high"]:
            triage["elevated"] += 1
        elif p >= thresholds["low"]:
            triage["uncertain"] += 1
        else:
            triage["low"] += 1
    total = len(probabilities) or 1
    triage_distribution = {k: v / total for k, v in triage.items()}
    script_distribution = {"latin_dominant": 1.0, "arabic_dominant": 0.0, "mixed_or_unknown": 0.0}
    return {
        "population_size": len(texts),
        "baseline": {
            "probabilities": [round(float(p), 4) for p in probabilities],
            "token_lengths": token_lengths,
            "triage_distribution": triage_distribution,
            "script_distribution": script_distribution,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="model")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, _, _ = train_and_evaluate(train, test, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
