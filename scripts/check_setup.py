"""Pre-flight smoke check: verifies model files, database schema and key endpoints
BEFORE starting the real server, so problems show up here instead of in the browser."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from fastapi.testclient import TestClient

    from stressguard.api import create_app
    from stressguard.settings import load_settings

    settings = load_settings(PROJECT_ROOT)
    ok = True

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal ok
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))

    report("model file exists", (settings.model_dir / "stress_model.joblib").exists())
    report("metrics file exists", (settings.model_dir / "metrics.json").exists())

    try:
        app = create_app(settings)
        report("app starts", True)
    except Exception as exc:
        report("app starts", False, str(exc))
        return 1

    client = TestClient(app)

    health = client.get("/api/v1/health")
    report("GET /api/v1/health", health.status_code == 200, f"status={health.status_code}")
    if health.status_code == 200:
        report("model loaded", health.json().get("model_loaded") is True)

    home = client.get("/")
    report("GET / (user page)", home.status_code == 200 and "html" in home.headers.get("content-type", ""), f"status={home.status_code}")

    admin = client.get("/admin")
    report("GET /admin (admin page)", admin.status_code == 200 and "html" in admin.headers.get("content-type", ""), f"status={admin.status_code}")

    for asset in ["styles.css", "app.js", "admin.js"]:
        res = client.get(f"/static/{asset}")
        report(f"GET /static/{asset}", res.status_code == 200, f"status={res.status_code}")

    analyze = client.post("/api/v1/analyze", json={"text": "I feel overwhelmed and cannot focus on work at all lately.", "locale": "en"})
    report("POST /api/v1/analyze", analyze.status_code == 200 and analyze.json().get("probability") is not None, f"status={analyze.status_code}")

    print("SMOKE CHECK:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
