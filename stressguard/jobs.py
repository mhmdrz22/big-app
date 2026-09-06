from __future__ import annotations

import json
from pathlib import Path

from stressguard.drift import DriftMonitor
from stressguard.release_gate import build_release_gate_report
from stressguard.retraining import run_shadow_retraining


def load_training_metrics(model_dir: Path) -> dict | None:
    metrics_path = Path(model_dir) / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def execute_shadow_retraining_job(settings, store, payload: dict, actor_user_id: int | None = None) -> dict:
    metrics = load_training_metrics(settings.model_dir)
    if not metrics:
        raise FileNotFoundError("training_metrics_not_found")

    drift_report_data = DriftMonitor(settings.model_dir).build_report(
        store.list_recent_assessments(limit=200),
        store.list_reviewed_feedback_for_quality(limit=200),
    )
    approved_feedback = store.list_approved_feedback_candidates(limit=500)
    report = run_shadow_retraining(settings, approved_feedback, metrics, drift_report_data)
    if settings.alert_webhook_url and drift_report_data.get("drift_level") in ("warning", "critical"):
        from stressguard.alerts import send_drift_alert
        send_drift_alert(settings.alert_webhook_url, drift_report_data, metrics, settings.alert_webhook_secret)
    run_record = store.save_retraining_run(actor_user_id=actor_user_id or 0, report=report)

    registered_candidate = None
    if report.get("status") == "completed" and report.get("recommendation") == "register_candidate":
        candidate_metrics = report.get("metrics", {})
        selected_name = candidate_metrics.get("selected_model")
        selected = next((item for item in candidate_metrics.get("candidates", []) if item.get("model") == selected_name), None)
        artifact_path = Path(report["output_dir"]) / "stress_model.joblib"
        if selected and artifact_path.exists():
            version_name = f"{selected_name}__shadow__run_{run_record['id']}"
            gate = build_release_gate_report(selected, drift_report_data, artifact_path.exists())
            registered_candidate = store.register_model_version(
                actor_user_id=actor_user_id or 0,
                version_name=version_name,
                f1=float(selected.get("f1", 0.0)),
                roc_auc=float(selected.get("roc_auc", 0.0)),
                accuracy=float(selected.get("accuracy", 0.0)),
                dataset_version=payload.get("dataset_version") or report.get("source_dataset_version", "shadow-dataset"),
                artifact_path=artifact_path,
                thresholds=candidate_metrics.get("thresholds", {}),
                release_gate=gate,
                notes=payload.get("notes") or "Auto-registered from shadow retraining run.",
            )

    return {
        "retraining_run": run_record,
        "candidate_model": registered_candidate,
        "report": report,
    }
