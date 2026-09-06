"""Assessment CRUD + drift snapshots."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import desc, func, select

from stressguard.models import Assessment, DriftSnapshot
from stressguard.privacy import hash_text, safe_excerpt


def utc_now():
    return datetime.now(timezone.utc)


class AssessmentStore:
    def _assessment_to_dict(self, row: Assessment) -> dict:
        return {
            "id": row.id,
            "text_excerpt": row.text_excerpt,
            "text_hash": row.text_hash,
            "model_name": row.model_name,
            "predicted_label": row.predicted_label,
            "probability": row.probability,
            "triage_level": row.triage_level,
            "analysis_mode": row.analysis_mode,
            "char_length": row.char_length,
            "token_length": row.token_length,
            "arabic_char_count": row.arabic_char_count,
            "latin_char_count": row.latin_char_count,
            "created_at": row.created_at.isoformat(),
        }

    def save_assessment(self, text, text_consent, locale, model_name, predicted_label, probability, triage_level, analysis_mode):
        stored_text = text if text_consent else None
        with self.session_scope() as session:
            row = Assessment(
                text=stored_text,
                text_excerpt=safe_excerpt(text),
                text_hash=hash_text(text),
                locale=locale,
                model_name=model_name,
                predicted_label=predicted_label,
                probability=probability,
                triage_level=triage_level,
                analysis_mode=analysis_mode,
                char_length=len(text),
                token_length=len(text.split()),
                arabic_char_count=sum(1 for ch in text if "\u0600" <= ch <= "\u06FF"),
                latin_char_count=sum(1 for ch in text if "a" <= ch.lower() <= "z"),
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def get_assessment(self, assessment_id: int):
        with self.session_scope() as session:
            row = session.get(Assessment, assessment_id)
            return self._assessment_to_dict(row) if row else None

    def list_recent_assessments(self, limit: int = 200):
        with self.session_scope() as session:
            rows = session.execute(select(Assessment).order_by(desc(Assessment.id)).limit(limit)).scalars().all()
            return [self._assessment_to_dict(row) for row in rows]

    def list_assessment_series(self, limit: int = 60):
        with self.session_scope() as session:
            rows = session.execute(select(Assessment).order_by(Assessment.id.asc()).limit(limit)).scalars().all()
            return [self._assessment_to_dict(row) for row in rows]

    def save_drift_snapshot(self, report: dict) -> int:
        with self.session_scope() as session:
            row = DriftSnapshot(
                report_json=json.dumps(report, ensure_ascii=False),
                drift_level=report.get("drift_level", "unknown"),
                population_size=report.get("population_size", 0),
                psi_text_length=report.get("metrics", {}).get("psi_text_length"),
                psi_token_length=report.get("metrics", {}).get("psi_token_length"),
                jsd_script=report.get("metrics", {}).get("jsd_script"),
                quality_proxy_agreement=report.get("metrics", {}).get("quality_proxy_agreement"),
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def list_drift_snapshots(self, limit: int = 30):
        with self.session_scope() as session:
            rows = session.execute(select(DriftSnapshot).order_by(desc(DriftSnapshot.id)).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "drift_level": row.drift_level,
                    "population_size": row.population_size,
                    "psi_text_length": row.psi_text_length,
                    "psi_token_length": row.psi_token_length,
                    "jsd_script": row.jsd_script,
                    "quality_proxy_agreement": row.quality_proxy_agreement,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
