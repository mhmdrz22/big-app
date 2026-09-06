from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stressguard.release_gate import build_release_gate_report
from train_model import resolve_columns, train_and_evaluate


def build_candidate_feedback_frame(items: list[dict]) -> pd.DataFrame:
    """Build the candidate-feedback frame used in shadow retraining.

    The direction -> sample-weight mapping is the new calibration knob:
    - 'agree'   (delta ~ +0.05): very light confirmation, weight ~ 0.10.
    - 'under'/'over' (delta ±0.30): bounded correction, weight up to 0.30.
    The weight is ALSO clamped to the project-wide 0.10-0.35 anti-overfit
    band so direction info can never create an outsized single-vote effect.
    """
    rows = []
    for item in items:
        # When consent_to_research is False, raw text is not stored; fall back
        # to the truncated excerpt so the data is still linkable for audit.
        text = item.get("text") or item.get("text_excerpt")
        if not text:
            continue
        direction = item.get("direction") or "agree"
        delta = float(item.get("soft_label_delta") or 0.0)
        stored_w = item.get("sample_weight")
        if stored_w is None:
            # Heuristic weight from direction magnitude, then clamped.
            base = 0.10 if direction == "agree" else min(0.30, 0.15 + abs(delta) * 0.5)
        else:
            base = float(stored_w)
        sample_weight = float(min(max(base, 0.10), 0.35))
        rows.append(
            {
                "text": text,
                "label": int(item["proposed_label"]),
                "sample_weight": sample_weight,
                "source": "approved_feedback",
                "feedback_id": item["id"],
                "direction": direction,
                "soft_label_delta": delta,
            }
        )
    return pd.DataFrame(rows)


MAX_FEEDBACK_SHARE = 0.20  # feedback samples may never exceed 20% of the training mix


def append_candidates_to_train(base_train: pd.DataFrame, candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Anti-overfitting guardrails for learning from user feedback:

    1. The original dataset is the anchor — feedback can never dominate it.
       Feedback rows are capped at MAX_FEEDBACK_SHARE of the final training mix.
    2. Every feedback row carries a small, bounded sample weight (0.10–0.35),
       so even approved feedback moves the model only slightly per run.
    3. This produces many small, safe updates instead of one large risky jump —
       the classic recipe for avoiding catastrophic overfitting to noisy labels.
    """
    text_col, label_col = resolve_columns(base_train)
    base_copy = base_train.copy()
    base_copy["sample_weight"] = 1.0
    if candidate_df.empty:
        return base_copy.drop(columns=["sample_weight"]), [1.0] * len(base_copy)

    max_feedback_rows = int(len(base_copy) * MAX_FEEDBACK_SHARE / (1 - MAX_FEEDBACK_SHARE))
    if len(candidate_df) > max_feedback_rows:
        # Deterministic cap: newest approved samples first, so the model tracks
        # recent drift without ever being flooded by feedback.
        candidate_df = candidate_df.head(max_feedback_rows)

    expanded_rows = []
    for _, row in candidate_df.iterrows():
        payload = {col: None for col in base_copy.columns}
        payload[text_col] = row["text"]
        payload[label_col] = int(row["label"])
        payload["sample_weight"] = float(min(max(row.get("sample_weight", 0.15), 0.10), 0.35))
        expanded_rows.append(payload)
    augmented = pd.concat([base_copy, pd.DataFrame(expanded_rows)], ignore_index=True)
    sample_weights = augmented["sample_weight"].fillna(1.0).astype(float).tolist()
    augmented = augmented.drop(columns=["sample_weight"])
    return augmented, sample_weights


def run_shadow_retraining(settings, approved_feedback: list[dict], current_metrics: dict, drift_report: dict | None) -> dict:
    train_path = Path(settings.baseline_train_path)
    test_path = Path(settings.baseline_test_path)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"baseline_dataset_missing:{train_path}:{test_path}")

    base_train = pd.read_csv(train_path)
    base_test = pd.read_csv(test_path)
    candidate_df = build_candidate_feedback_frame(approved_feedback)
    if len(candidate_df) < settings.min_approved_feedback_for_retraining:
        return {
            "status": "blocked",
            "reason": "not_enough_approved_feedback",
            "approved_feedback_count": int(len(candidate_df)),
            "minimum_required": settings.min_approved_feedback_for_retraining,
        }

    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = Path(settings.retraining_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_df.to_csv(output_dir / "candidate_feedback_dataset.csv", index=False)
    augmented_train, sample_weights = append_candidates_to_train(base_train, candidate_df)
    summary, _, drift_profile = train_and_evaluate(augmented_train, base_test, output_dir=output_dir, sample_weight=sample_weights)

    selected_name = summary["selected_model"]
    selected = next(item for item in summary["candidates"] if item["model"] == selected_name)
    baseline_name = current_metrics.get("selected_model")
    baseline_selected = next((item for item in current_metrics.get("candidates", []) if item.get("model") == baseline_name), None)

    improvements = {
        "f1_delta": round(float(selected.get("f1", 0.0)) - float((baseline_selected or {}).get("f1", 0.0)), 4),
        "roc_auc_delta": round(float(selected.get("roc_auc", 0.0)) - float((baseline_selected or {}).get("roc_auc", 0.0)), 4),
        "accuracy_delta": round(float(selected.get("accuracy", 0.0)) - float((baseline_selected or {}).get("accuracy", 0.0)), 4),
    }
    release_gate = build_release_gate_report(selected, drift_report, artifact_exists=(output_dir / "stress_model.joblib").exists())

    # Champion/challenger rule: a feedback-trained candidate may only be registered
    # when it does NOT regress on the locked original test set AND passes the release
    # gate. Feedback can never silently degrade the production model.
    no_regression = (
        improvements["f1_delta"] >= -0.005
        and improvements["roc_auc_delta"] >= -0.005
    )
    recommendation = "register_candidate" if release_gate["gate_status"] == "passed" and no_regression else "hold"

    report_extra = {
        "guardrails": {
            "max_feedback_share": MAX_FEEDBACK_SHARE,
            "sample_weight_range": [0.10, 0.35],
            "champion_challenger": "candidate must not regress on the locked test set",
            "feedback_cap_applied": True,
        }
    }

    report = {
        "status": "completed",
        "base_model_version": baseline_name,
        "candidate_model_name": selected_name,
        "source_dataset_version": f"{settings.baseline_train_path.name}+approved_feedback",
        "approved_feedback_count": int(len(candidate_df)),
        "output_dir": str(output_dir),
        "candidate_dataset_path": str(output_dir / "candidate_feedback_dataset.csv"),
        "metrics": summary,
        "drift_profile": drift_profile,
        "improvements": improvements,
        "release_gate": release_gate,
        "recommendation": recommendation,
        **report_extra,
    }
    (output_dir / "shadow_run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
