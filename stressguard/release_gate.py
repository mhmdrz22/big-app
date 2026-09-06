from __future__ import annotations


def build_release_gate_report(model_summary: dict, drift_report: dict | None, artifact_exists: bool) -> dict:
    checks = []

    f1 = float(model_summary.get("f1") or 0.0)
    roc_auc = float(model_summary.get("roc_auc") or 0.0)
    accuracy = float(model_summary.get("accuracy") or 0.0)
    drift_level = (drift_report or {}).get("drift_level", "unknown")

    checks.append({
        "name": "artifact_present",
        "passed": bool(artifact_exists),
        "detail": "مدل و آرتیفکت باید روی دیسک و در رجیستری موجود باشد.",
    })
    checks.append({
        "name": "minimum_f1",
        "passed": f1 >= 0.72,
        "detail": f"F1 فعلی {f1:.3f} است و حداقل مورد انتظار 0.720 در نظر گرفته شده.",
    })
    checks.append({
        "name": "minimum_roc_auc",
        "passed": roc_auc >= 0.80,
        "detail": f"ROC-AUC فعلی {roc_auc:.3f} است و حداقل مورد انتظار 0.800 در نظر گرفته شده.",
    })
    checks.append({
        "name": "minimum_accuracy",
        "passed": accuracy >= 0.70,
        "detail": f"Accuracy فعلی {accuracy:.3f} است و حداقل مورد انتظار 0.700 در نظر گرفته شده.",
    })
    checks.append({
        "name": "drift_not_critical",
        "passed": drift_level != "critical",
        "detail": f"سطح رانش فعلی `{drift_level}` است و نباید critical باشد.",
    })

    passed_all = all(item["passed"] for item in checks)
    gate_status = "passed" if passed_all else "blocked"
    return {
        "gate_status": gate_status,
        "checks": checks,
        "summary": {
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "accuracy": round(accuracy, 4),
            "drift_level": drift_level,
        },
        "recommended_next_stage": "staging" if passed_all else "candidate",
    }
