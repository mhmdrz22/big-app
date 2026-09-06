"""
AramNegar — lean, risk-focused test suite.

Philosophy (per product owner): few tests, high value, guarding the two
critical surfaces:
  1. USER PATH  — site loads, analyze returns fast, output displays, crisis
                  text always co-renders urgent support.
  2. ADMIN PATH — dashboard works, passwords never leak, no unauthorized
                  access, feedback flows correctly end-to-end, and the
                  database is never accidentally destroyed.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from stressguard.api import create_app
from stressguard.settings import AppSettings


def build_settings(tmp_path: Path) -> AppSettings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    train_csv.write_text(
        "text,label\nI feel overwhelmed and stressed,1\nI am anxious about deadlines,1\nI feel calm and rested today,0\nEverything is manageable and fine,0\n",
        encoding="utf-8",
    )
    test_csv.write_text(
        "text,label\nI cannot focus and feel overwhelmed,1\nI feel relaxed and okay,0\n",
        encoding="utf-8",
    )
    return AppSettings(
        app_name="AramNegar Test",
        app_env="test",
        base_dir=PROJECT_ROOT,
        model_dir=PROJECT_ROOT / "model",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        store_text_by_default=False,
        low_risk_threshold=0.35,
        high_risk_threshold=0.60,
        decision_threshold=0.48,
        analyze_rate_limit_per_min=1000,
        feedback_rate_limit_per_min=1000,
        auth_secret="test-secret-key-that-is-long-enough-32b",
        auth_algorithm="HS256",
        access_token_exp_minutes=120,
        bootstrap_admin_enabled=True,
        baseline_train_path=train_csv,
        baseline_test_path=test_csv,
        retraining_dir=tmp_path / "retraining_runs",
        min_approved_feedback_for_retraining=1,
    )


def make_client(tmp_path) -> TestClient:
    return TestClient(create_app(build_settings(tmp_path)))


def login(client: TestClient) -> str:
    r = client.post(
        "/api/v1/auth/bootstrap-admin",
        json={"full_name": "Root Admin", "email": "admin@example.com", "password": "StrongPass123!"},
    )
    assert r.status_code == 200
    return r.json()["access_token"]


# ═══════════════════════════════════════════════════════════════════
# 1. USER PATH — fast, loads, displays, safe
# ═══════════════════════════════════════════════════════════════════

def test_user_path_analyze_returns_score_and_display_fields(tmp_path):
    """The core user flow: type text -> click -> get stress_score + band fast."""
    client = make_client(tmp_path)
    r = client.post(
        "/api/v1/analyze",
        json={"text": "I feel overwhelmed and anxious about my exam tomorrow", "locale": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["analysis_mode"] == "model_inference"
    assert 0 <= body["stress_score"] <= 100
    assert body["stress_band"] in {"پایین", "متوسط", "بالا", "خیلی بالا"}
    assert body["triage_level"] in {"low", "uncertain", "elevated"}
    assert body["recommendations"]
    assert body["assessment_id"] >= 1


def test_crisis_text_always_co_renders_urgent_support(tmp_path):
    """Hard guardrail: crisis text sets urgent_support_flag WITH the result."""
    client = make_client(tmp_path)
    r = client.post(
        "/api/v1/analyze",
        json={"text": "I feel so hopeless I want to kill myself tonight", "locale": "en"},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["urgent_support_flag"] is True
    assert body["urgent_support_message"]


def test_persian_input_safely_falls_back_to_support(tmp_path):
    """Persian input must degrade safely until a Persian model exists."""
    client = make_client(tmp_path)
    r = client.post(
        "/api/v1/analyze",
        json={"text": "چند هفته است تمرکز ندارم و خوابم به‌هم ریخته و احساس فشار دارم.", "locale": "fa"},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["analysis_mode"] == "support_only"
    assert body["stress_score"] is None


# ═══════════════════════════════════════════════════════════════════
# 2. ADMIN PATH — dashboard, security, data integrity
# ═══════════════════════════════════════════════════════════════════

def test_admin_dashboard_works_after_login(tmp_path):
    """The dashboard must load and return KPIs for an authorized admin."""
    client = make_client(tmp_path)
    token = login(client)
    r = client.get("/api/v1/admin/summary", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["viewer"]["email"] == "admin@example.com"
    assert "assessment_count" in body["store"]
    assert "feedback_count" in body["store"]


def test_admin_requires_auth_and_role(tmp_path):
    """No anonymous access; auditor cannot review feedback."""
    client = make_client(tmp_path)
    assert client.get("/api/v1/admin/summary").status_code == 401

    admin_token = login(client)
    auditor = client.post(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"full_name": "Audit", "email": "audit@example.com", "password": "StrongPass456!", "roles": ["auditor"]},
    )
    assert auditor.status_code == 200
    auditor_token = client.post("/api/v1/auth/login", json={"email": "audit@example.com", "password": "StrongPass456!"}).json()["access_token"]

    # auditor can read summary but NOT review feedback
    assert client.get("/api/v1/admin/summary", headers={"Authorization": f"Bearer {auditor_token}"}).status_code == 200

    analyze = client.post("/api/v1/analyze", json={"text": "I feel overwhelmed and anxious about my exam tomorrow", "locale": "en"}).json()
    fb = client.post(
        "/api/v1/feedback",
        json={"assessment_id": analyze["assessment_id"], "text": "I feel overwhelmed and anxious about my exam tomorrow", "direction": "agree", "consent_to_research": True},
    ).json()
    forbidden = client.post(
        f"/api/v1/admin/feedback-queue/{fb['feedback_id']}/review",
        headers={"Authorization": f"Bearer {auditor_token}"},
        json={"review_action": "approve", "review_notes": "nope"},
    )
    assert forbidden.status_code == 403


def test_password_never_leaks_through_api(tmp_path):
    """SECURITY GUARD: no endpoint may serialize password_hash or raw password."""
    client = make_client(tmp_path)
    token = login(client)

    # /auth/me
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert "password" not in str(me.json()).lower()

    # /admin/users list
    users = client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert users.status_code == 200
    assert "password" not in str(users.json()).lower()


def test_feedback_flows_end_to_end_and_is_persisted(tmp_path):
    """Feedback: submit -> quarantine -> dual review -> approved -> retraining candidate."""
    client = make_client(tmp_path)
    admin_token = login(client)

    # two reviewers
    for name, email, pw in [("Rev One", "rev1@example.com", "StrongPass111!"), ("Rev Two", "rev2@example.com", "StrongPass222!")]:
        r = client.post(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"full_name": name, "email": email, "password": pw, "roles": ["reviewer"]},
        )
        assert r.status_code == 200
    t1 = client.post("/api/v1/auth/login", json={"email": "rev1@example.com", "password": "StrongPass111!"}).json()["access_token"]
    t2 = client.post("/api/v1/auth/login", json={"email": "rev2@example.com", "password": "StrongPass222!"}).json()["access_token"]

    analyze = client.post("/api/v1/analyze", json={"text": "I feel overwhelmed and anxious about my exam tomorrow", "locale": "en"}).json()
    fb = client.post(
        "/api/v1/feedback",
        json={"assessment_id": analyze["assessment_id"], "text": "I feel overwhelmed and anxious about my exam tomorrow", "direction": "under", "consent_to_research": True},
    ).json()
    assert fb["direction"] == "under"
    # model predicted stress=1 for this text; "under" on an already-high verdict
    # reads as strong confirmation -> bounded +0.10 (not the full +0.30)
    assert fb["soft_label_delta"] == 0.10

    # dual review
    r1 = client.post(f"/api/v1/admin/feedback-queue/{fb['feedback_id']}/review", headers={"Authorization": f"Bearer {t1}"}, json={"review_action": "approve", "review_notes": "ok"})
    assert r1.status_code == 200
    r2 = client.post(f"/api/v1/admin/feedback-queue/{fb['feedback_id']}/review", headers={"Authorization": f"Bearer {t2}"}, json={"review_action": "approve", "review_notes": "ok2"})
    assert r2.status_code == 200
    assert r2.json()["item"]["review_state"] == "approved"

    # approved feedback becomes a retraining candidate
    run = client.post(
        "/api/v1/admin/retraining/run-shadow",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"dataset_version": "dreaddit-phase1+approved-feedback", "notes": "shadow"},
    )
    assert run.status_code == 200
    assert run.json()["item"]["approved_feedback_count"] >= 1


def test_database_data_survives_app_restart(tmp_path):
    """DATA INTEGRITY GUARD: assessments + feedback persist across restart; no destructive endpoint."""
    settings = build_settings(tmp_path)
    client = TestClient(create_app(settings))
    token = login(client)
    analyze = client.post("/api/v1/analyze", json={"text": "I feel overwhelmed and anxious about my exam tomorrow", "locale": "en"}).json()
    client.post(
        "/api/v1/feedback",
        json={"assessment_id": analyze["assessment_id"], "text": "I feel overwhelmed and anxious about my exam tomorrow", "direction": "agree", "consent_to_research": True},
    )

    # restart the app on the SAME database file — admin persists, so LOGIN
    client2 = TestClient(create_app(settings))
    r = client2.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "StrongPass123!"})
    assert r.status_code == 200
    token2 = r.json()["access_token"]
    summary = client2.get("/api/v1/admin/summary", headers={"Authorization": f"Bearer {token2}"}).json()
    assert summary["store"]["assessment_count"] >= 1
    assert summary["store"]["feedback_count"] >= 1

    # no destructive HTTP endpoints exist
    routes = [r.path for r in create_app(settings).routes]
    assert not any("delete" in r.lower() for r in routes)
    assert not any("drop" in r.lower() for r in routes)
