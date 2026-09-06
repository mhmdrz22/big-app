"""Real-time drift alerting via webhook (Slack, Discord, Teams, generic)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx


def send_drift_alert(webhook_url: str, drift_report: dict, summary: dict, secret: str | None = None):
    if not webhook_url:
        return False
    level = drift_report.get("drift_level", "unknown")
    if level not in ("warning", "critical"):
        return False

    emoji = "🔴" if level == "critical" else "🟡"
    payload = {
        "text": f"{emoji} AramNegar Drift Alert: {level.upper()}",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} Drift Alert — {level.upper()}"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Level:*\n{level}"},
                    {"type": "mrkdwn", "text": f"*Population:*\n{drift_report.get('population_size', 0)}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{datetime.now(timezone.utc).isoformat()}"},
                    {"type": "mrkdwn", "text": f"*Model:*\n{summary.get('selected_model', 'unknown')}"},
                ],
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Alert-Secret"] = secret
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(webhook_url, json=payload, headers=headers)
            return resp.status_code < 300
    except Exception:
        return False
