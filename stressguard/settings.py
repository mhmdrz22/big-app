from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppSettings:
    app_name: str
    app_env: str
    base_dir: Path
    model_dir: Path
    database_url: str
    store_text_by_default: bool
    low_risk_threshold: float
    high_risk_threshold: float
    decision_threshold: float
    analyze_rate_limit_per_min: int
    feedback_rate_limit_per_min: int
    auth_secret: str
    auth_algorithm: str
    access_token_exp_minutes: int
    bootstrap_admin_enabled: bool
    baseline_train_path: Path
    baseline_test_path: Path
    retraining_dir: Path
    min_approved_feedback_for_retraining: int
    worker_poll_seconds: int = 5
    worker_name: str = "stress-worker-1"
    max_job_attempts: int = 3
    alert_webhook_url: str | None = None
    alert_webhook_secret: str | None = None


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(base: Path) -> None:
    """Read .env from the project root and inject into os.environ (without overriding
    variables already set in the shell). Works with or without python-dotenv."""
    env_path = base / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_sqlite_url(database_url: str, base: Path) -> str:
    """Turn a relative sqlite:///... URL into an absolute one rooted at the project,
    so the DB file is found no matter which directory the app is started from."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return database_url
    raw_path = database_url[len(prefix):]
    if raw_path.startswith("/") or (len(raw_path) > 2 and raw_path[1] == ":"):
        return database_url
    return f"{prefix}{(base / raw_path).as_posix()}"


def _resolve_path(value: str, base: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def load_settings(base_dir: Path | None = None) -> AppSettings:
    base = Path(base_dir or Path(__file__).resolve().parents[1])
    instance_dir = base / "instance"
    instance_dir.mkdir(exist_ok=True)

    _load_env_file(base)

    default_db = f"sqlite:///{(instance_dir / 'stress_sentinel.db').as_posix()}"
    database_url = _resolve_sqlite_url(os.getenv("DATABASE_URL", default_db), base)

    return AppSettings(
        app_name="Stress Sentinel API",
        app_env=os.getenv("APP_ENV", "development"),
        base_dir=base,
        model_dir=_resolve_path(os.getenv("MODEL_DIR", str(base / "model")), base),
        database_url=database_url,
        store_text_by_default=_to_bool(os.getenv("STORE_TEXT_BY_DEFAULT"), False),
        low_risk_threshold=float(os.getenv("LOW_RISK_THRESHOLD", "0.35")),
        high_risk_threshold=float(os.getenv("HIGH_RISK_THRESHOLD", "0.60")),
        decision_threshold=float(os.getenv("DECISION_THRESHOLD", "0.48")),
        analyze_rate_limit_per_min=int(os.getenv("ANALYZE_RATE_LIMIT_PER_MIN", "20")),
        feedback_rate_limit_per_min=int(os.getenv("FEEDBACK_RATE_LIMIT_PER_MIN", "30")),
        auth_secret=os.getenv("AUTH_SECRET", "change-me-in-production"),
        auth_algorithm=os.getenv("AUTH_ALGORITHM", "HS256"),
        access_token_exp_minutes=int(os.getenv("ACCESS_TOKEN_EXP_MINUTES", "120")),
        bootstrap_admin_enabled=_to_bool(os.getenv("BOOTSTRAP_ADMIN_ENABLED"), True),
        baseline_train_path=_resolve_path(os.getenv("BASELINE_TRAIN_PATH", str(base / "data" / "dreaddit-train.csv")), base),
        baseline_test_path=_resolve_path(os.getenv("BASELINE_TEST_PATH", str(base / "data" / "dreaddit-test.csv")), base),
        retraining_dir=_resolve_path(os.getenv("RETRAINING_DIR", str(base / "model" / "retraining_runs")), base),
        min_approved_feedback_for_retraining=int(os.getenv("MIN_APPROVED_FEEDBACK_FOR_RETRAINING", "5")),
        worker_poll_seconds=int(os.getenv("WORKER_POLL_SECONDS", "5")),
        worker_name=os.getenv("WORKER_NAME", "stress-worker-1"),
        max_job_attempts=int(os.getenv("MAX_JOB_ATTEMPTS", "3")),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        alert_webhook_secret=os.getenv("ALERT_WEBHOOK_SECRET"),
    )
