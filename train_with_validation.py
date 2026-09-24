""" انتخاب مدل با مجموعه اعتبارسنجی"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from train_model import (build_hybrid_matrix, build_models, resolve_columns,
                         _fit_member, _predict_proba_member)


def train_with_validation(train_df, test_df, output_dir="model", val_size=0.2, seed=42):
    text_col, label_col = resolve_columns(train_df)

    # ۱) ۲۰٪ از مجموعه آموزش جدا می‌شود برای اعتبارسنجی (فایل تست هنوز دست‌نخورده است)
    tr, val = train_test_split(train_df, test_size=val_size,
                               stratify=train_df[label_col], random_state=seed)
    X_text_tr, X_num_tr, y_tr = build_hybrid_matrix(tr, text_col, label_col)
    X_text_val, X_num_val, y_val = build_hybrid_matrix(val, text_col, label_col)

    classes = np.array(sorted(np.unique(y_tr)))
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(x) for c, x in zip(classes, w)}

    # ۲) برنده با F1 روی «اعتبارسنجی» انتخاب می‌شود، نه تست
    scores, fitted = {}, {}
    for name, pipe in build_models(class_weight).items():
        m = _fit_member(name, clone(pipe), X_text_tr, X_num_tr, y_tr)
        p_val = _predict_proba_member(name, m, X_text_val, X_num_val)
        scores[name] = float(f1_score(y_val, (p_val >= 0.48).astype(int)))
        fitted[name] = m
    ens_val = np.mean([_predict_proba_member(n, fitted[n], X_text_val, X_num_val)
                       for n in fitted], axis=0)
    scores["voting_ensemble"] = float(f1_score(y_val, (ens_val >= 0.48).astype(int)))
    best_name = max(scores, key=scores.get)

    # ۳) بازآموزی روی «کل آموزش» و بعد فقط «یک‌بار» ارزیابی روی تستِ قفل‌شده
    X_text_all, X_num_all, y_all = build_hybrid_matrix(train_df, text_col, label_col)
    X_text_te, X_num_te, y_te = build_hybrid_matrix(test_df, text_col, label_col)
    final = {n: _fit_member(n, clone(p), X_text_all, X_num_all, y_all)
             for n, p in build_models(class_weight).items()}
    if best_name == "voting_ensemble":
        p_te = np.mean([_predict_proba_member(n, final[n], X_text_te, X_num_te)
                        for n in final], axis=0)
    else:
        p_te = _predict_proba_member(best_name, final[best_name], X_text_te, X_num_te)

    report = {
        "validation_f1_of_candidates": {k: round(v, 4) for k, v in scores.items()},
        "selected_model": best_name,
        "test_evaluated_once": {
            "accuracy": round(float(accuracy_score(y_te, (p_te >= 0.48).astype(int))), 4),
            "f1": round(float(f1_score(y_te, (p_te >= 0.48).astype(int))), 4),
            "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    train = pd.read_csv("dreaddit-train.csv")
    test = pd.read_csv("dreaddit-test.csv")
    print(json.dumps(train_with_validation(train, test), indent=2, ensure_ascii=False))
