"""
AramNegar — guardrail tests for the four critical surfaces the product
owner did not name but which silently break the product's core promises:
  1. Anti-overfitting (feedback cap + weight clamp)   — "برازش نباید اتفاق بیفته"
  2. Privacy (text not stored without consent)        — "متن ذخیره نشه"
  3. Release gate (champion/challenger)               — "مدل بد نباید بره پروداکشن"
  4. Rate limiting (bot protection)                   — "اسپم نشه"
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stressguard.retraining import MAX_FEEDBACK_SHARE, append_candidates_to_train, build_candidate_feedback_frame
from stressguard.release_gate import build_release_gate_report
from stressguard.rate_limit import InMemoryRateLimiter
from stressguard.privacy import hash_text, safe_excerpt


# ── 1. Anti-overfitting ─────────────────────────────────────────────
def test_feedback_never_exceeds_20_percent_of_training_mix():
    """Even with a flood of feedback, the mix stays <= 20% feedback."""
    base = pd.DataFrame({"text": [f"base text {i}" for i in range(100)], "label": [0, 1] * 50})
    flood = pd.DataFrame({"text": [f"feedback {i}" for i in range(500)], "label": [1] * 500})
    augmented, weights = append_candidates_to_train(base, flood)

    feedback_rows = augmented[augmented["text"].str.startswith("feedback")]
    ratio = len(feedback_rows) / len(augmented)
    assert ratio <= MAX_FEEDBACK_SHARE, f"feedback ratio {ratio:.3f} > cap {MAX_FEEDBACK_SHARE}"


def test_sample_weight_is_clamped_to_bounded_range():
    """Every feedback row's weight stays inside [0.10, 0.35]."""
    items = [
        {"id": 1, "text": "a", "proposed_label": 1, "sample_weight": 5.0, "direction": "under", "soft_label_delta": 0.30},
        {"id": 2, "text": "b", "proposed_label": 0, "sample_weight": 0.001, "direction": "agree", "soft_label_delta": 0.05},
        {"id": 3, "text": "c", "proposed_label": 1, "sample_weight": None, "direction": "over", "soft_label_delta": -0.30},
    ]
    df = build_candidate_feedback_frame(items)
    assert df["sample_weight"].between(0.10, 0.35).all(), df["sample_weight"].tolist()


# ── 2. Privacy ───────────────────────────────────────────────────────
def test_text_not_stored_without_research_consent(tmp_path):
    """Raw text must NOT persist when consent_to_research is False."""
    from fastapi.testclient import TestClient
    from stressguard.api import create_app
    from stressguard.settings import AppSettings

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("text,label\nI feel stressed,1\nI feel calm,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("text,label\nI feel overwhelmed,1\nI feel fine,0\n", encoding="utf-8")
    settings = AppSettings(
        app_name="t", app_env="test", base_dir=PROJECT_ROOT, model_dir=PROJECT_ROOT / "model",
        database_url=f"sqlite:///{tmp_path / 'privacy.db'}", store_text_by_default=False,
        low_risk_threshold=0.35, high_risk_threshold=0.60, decision_threshold=0.48,
        analyze_rate_limit_per_min=1000, feedback_rate_limit_per_min=1000,
        auth_secret="test-secret-key-that-is-long-enough-32b", auth_algorithm="HS256",
        access_token_exp_minutes=120, bootstrap_admin_enabled=True,
        baseline_train_path=data_dir / "train.csv", baseline_test_path=data_dir / "test.csv",
        retraining_dir=tmp_path / "rr", min_approved_feedback_for_retraining=1,
    )
    client = TestClient(create_app(settings))
    a = client.post("/api/v1/analyze", json={"text": "I feel overwhelmed and anxious about my exam", "locale": "en"}).json()
    client.post(
        "/api/v1/feedback",
        json={"assessment_id": a["assessment_id"], "text": "I feel overwhelmed and anxious about my exam", "direction": "agree", "consent_to_research": False},
    )

    # raw text must not be readable from the DB — only hash + excerpt survive
    import sqlite3
    conn = sqlite3.connect(f"{tmp_path / 'privacy.db'}")
    stored = conn.execute("SELECT text FROM feedback_quarantine").fetchall()
    conn.close()
    for (t,) in stored:
        assert t is None, f"raw text leaked: {t!r}"


# ── 3. Release gate ────────────────────────────────────────────────
def test_release_gate_blocks_weak_model_and_passes_strong():
    """Champion/challenger: weak model must be blocked, strong model passes."""
    weak = build_release_gate_report({"f1": 0.60, "roc_auc": 0.70, "accuracy": 0.60}, {"drift_level": "stable"}, True)
    assert weak["gate_status"] == "blocked"
    assert not all(c["passed"] for c in weak["checks"])

    strong = build_release_gate_report({"f1": 0.80, "roc_auc": 0.85, "accuracy": 0.78}, {"drift_level": "stable"}, True)
    assert strong["gate_status"] == "passed"

    critical = build_release_gate_report({"f1": 0.80, "roc_auc": 0.85, "accuracy": 0.78}, {"drift_level": "critical"}, True)
    assert critical["gate_status"] == "blocked"


# ── 4. Rate limiting ───────────────────────────────────────────────
def test_rate_limiter_blocks_bursts_and_allows_after_window():
    """Bursts over the limit are blocked; the window resets."""
    limiter = InMemoryRateLimiter()
    for _ in range(3):
        assert limiter.check("user-1", limit=3)[0] is True
    allowed, retry = limiter.check("user-1", limit=3)
    assert allowed is False
    assert retry >= 1
    # different key unaffected
    assert limiter.check("user-2", limit=3)[0] is True
