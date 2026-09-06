"""
Quick smoke test for the feedback store.
Run: python data/feedback-store-test.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stressguard.data_store import DataStore
from stressguard.settings import AppSettings

settings = AppSettings(
    app_name="Test", app_env="test", base_dir=Path("."),
    model_dir=Path("model"), database_url="sqlite:///test_feedback.db",
    store_text_by_default=False, low_risk_threshold=0.35,
    high_risk_threshold=0.60, decision_threshold=0.48,
    analyze_rate_limit_per_min=100, feedback_rate_limit_per_min=100,
    auth_secret="x", auth_algorithm="HS256", access_token_exp_minutes=120,
    bootstrap_admin_enabled=True, baseline_train_path=Path("data/dreaddit-train.csv"),
    baseline_test_path=Path("data/dreaddit-test.csv"), retraining_dir=Path("retraining_runs"),
    min_approved_feedback_for_retraining=1,
)
store = DataStore(settings.database_url, settings.base_dir)
aid = store.save_assessment("I feel stressed", False, "en", "test", 1, 0.85, "elevated", "model_inference")
fid = store.save_feedback(aid, "I feel stressed", 1, "note", 1, 0.85, 0.9, "approved_limited_weight", 0.2, [], True, True, direction="agree", soft_label_delta=0.05)
print(f"assessment_id={aid}, feedback_id={fid}")
print("OK")
