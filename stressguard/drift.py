from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _histogram(values: list[float], bins: list[float]) -> list[float]:
    counts = [0 for _ in range(len(bins) - 1)]
    if not values:
        return [0.0 for _ in counts]
    for value in values:
        placed = False
        for idx in range(len(bins) - 1):
            left, right = bins[idx], bins[idx + 1]
            if idx == len(bins) - 2:
                if left <= value <= right:
                    counts[idx] += 1
                    placed = True
                    break
            elif left <= value < right:
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    total = sum(counts) or 1
    return [count / total for count in counts]


def population_stability_index(reference: list[float], current: list[float], bins: list[float]) -> float:
    ref_hist = _histogram(reference, bins)
    cur_hist = _histogram(current, bins)
    eps = 1e-6
    return round(sum((c - r) * math.log((c + eps) / (r + eps)) for r, c in zip(ref_hist, cur_hist)), 4)


def jensen_shannon_divergence(ref_dist: dict[str, float], cur_dist: dict[str, float]) -> float:
    keys = sorted(set(ref_dist) | set(cur_dist))
    eps = 1e-9
    p = [ref_dist.get(k, 0.0) + eps for k in keys]
    q = [cur_dist.get(k, 0.0) + eps for k in keys]
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a_vals, b_vals):
        return sum(a * math.log(a / b) for a, b in zip(a_vals, b_vals))

    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 4)


class DriftMonitor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.profile_path = self.model_dir / "drift_profile.json"
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8")) if self.profile_path.exists() else {}

    @staticmethod
    def _script_bucket(item: dict) -> str:
        if item.get("arabic_char_count", 0) > item.get("latin_char_count", 0):
            return "arabic_dominant"
        if item.get("latin_char_count", 0) > 0:
            return "latin_dominant"
        return "mixed_or_unknown"

    def build_report(self, recent_assessments: list[dict], reviewed_feedback: list[dict]) -> dict:
        baseline = self.profile.get("baseline", {})
        reference_probs = baseline.get("probabilities", [])
        reference_lengths = baseline.get("token_lengths", [])
        reference_triage = baseline.get("triage_distribution", {"low": 0.34, "uncertain": 0.33, "elevated": 0.33})
        reference_scripts = baseline.get("script_distribution", {"latin_dominant": 1.0})

        current_probs = [float(x["probability"]) for x in recent_assessments if x.get("probability") is not None]
        current_lengths = [float(x.get("token_length", 0)) for x in recent_assessments]

        triage_counter = Counter(x.get("triage_level", "unknown") for x in recent_assessments)
        triage_total = sum(triage_counter.values()) or 1
        current_triage = {k: v / triage_total for k, v in triage_counter.items()}

        script_counter = Counter(self._script_bucket(x) for x in recent_assessments)
        script_total = sum(script_counter.values()) or 1
        current_scripts = {k: v / script_total for k, v in script_counter.items()}

        agreement_values = [1.0 if int(item["model_label"]) == int(item["proposed_label"]) else 0.0 for item in reviewed_feedback]
        quality_proxy_agreement = round(_safe_mean(agreement_values), 4) if agreement_values else None

        # PSI/JSD on a handful of samples is statistically meaningless noise —
        # suppress those metrics until the window is large enough.
        enough_samples = len(recent_assessments) >= 25
        metrics = {
            "psi_probability": population_stability_index(reference_probs, current_probs, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) if enough_samples and reference_probs and current_probs else None,
            "psi_length": population_stability_index(reference_lengths, current_lengths, [0, 20, 50, 100, 180, 400, 10000]) if enough_samples and reference_lengths and current_lengths else None,
            "jsd_triage": jensen_shannon_divergence(reference_triage, current_triage) if enough_samples else None,
            "jsd_script": jensen_shannon_divergence(reference_scripts, current_scripts) if enough_samples else None,
            "quality_proxy_agreement": quality_proxy_agreement,
            "current_probability_mean": round(_safe_mean(current_probs), 4) if current_probs else None,
            "current_token_length_mean": round(_safe_mean(current_lengths), 2) if current_lengths else None,
        }

        flags = []
        level = "stable"
        if metrics["psi_probability"] is not None and metrics["psi_probability"] >= 0.25:
            flags.append("distribution_shift_probability")
            level = "warning"
        if metrics["psi_length"] is not None and metrics["psi_length"] >= 0.25:
            flags.append("distribution_shift_length")
            level = "warning"
        if metrics["jsd_script"] is not None and metrics["jsd_script"] >= 0.12:
            flags.append("language_mix_shift")
            level = "warning"
        if metrics["quality_proxy_agreement"] is not None and metrics["quality_proxy_agreement"] < 0.7:
            flags.append("reviewed_feedback_agreement_drop")
            level = "critical"
        if not enough_samples:
            flags.append("low_sample_window")
            level = "stable" if level != "critical" else level

        return {
            "drift_level": level,
            "window_size": len(recent_assessments),
            "population_size": int(self.profile.get("population_size", 0)),
            "metrics": metrics,
            "reference": {
                "triage_distribution": reference_triage,
                "script_distribution": reference_scripts,
            },
            "current": {
                "triage_distribution": current_triage,
                "script_distribution": current_scripts,
            },
            "flags": flags,
            "recommendations": [
                "اگر PSI یا JSD بالا رفت، داده‌های جدید را به‌صورت domain slice بررسی کن.",
                "اگر quality proxy افت کرد، استقرار خودکار مدل جدید را متوقف کن و error analysis انجام بده.",
                "برای فارسی‌محور شدن ورودی‌ها، مسیر مدل انگلیسی را از support-only فارسی جدا نگه دار.",
            ],
        }
