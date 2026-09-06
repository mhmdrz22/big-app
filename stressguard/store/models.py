"""Model registry + user management."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import desc, select

from stressguard.models import ModelVersion, RetrainingRun, User
from stressguard.security import hash_password, verify_password


def utc_now():
    return datetime.now(timezone.utc)


class UserStore:
    def _user_to_dict(self, row: User) -> dict:
        return {
            "id": row.id,
            "full_name": row.full_name,
            "email": row.email,
            "roles": json.loads(row.roles_json),
            "is_active": row.is_active,
            "created_at": row.created_at.isoformat(),
        }

    def bootstrap_admin(self, email: str, full_name: str, password: str) -> dict:
        with self.session_scope() as session:
            existing = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if existing:
                raise ValueError("admin_already_exists")
            row = User(
                full_name=full_name,
                email=email,
                hashed_password=hash_password(password),
                roles_json=json.dumps(["admin", "reviewer", "ml_engineer", "safety_reviewer", "auditor"]),
                is_active=True,
            )
            session.add(row)
            session.flush()
            return self._user_to_dict(row)

    def create_user(self, full_name: str, email: str, password: str, roles: list[str]) -> dict:
        with self.session_scope() as session:
            existing = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if existing:
                raise ValueError("user_already_exists")
            row = User(
                full_name=full_name,
                email=email,
                hashed_password=hash_password(password),
                roles_json=json.dumps(roles),
                is_active=True,
            )
            session.add(row)
            session.flush()
            return self._user_to_dict(row)

    def authenticate_user(self, email: str, password: str):
        with self.session_scope() as session:
            row = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if not row or not verify_password(password, row.hashed_password):
                return None
            return self._user_to_dict(row)

    def get_user(self, user_id: int):
        with self.session_scope() as session:
            row = session.get(User, user_id)
            return self._user_to_dict(row) if row else None

    def list_users(self, limit: int = 50):
        with self.session_scope() as session:
            rows = session.execute(select(User).order_by(User.id.asc()).limit(limit)).scalars().all()
            return [self._user_to_dict(row) for row in rows]


class ModelStore:
    def _model_to_dict(self, row: ModelVersion) -> dict:
        return {
            "id": row.id,
            "version_name": row.version_name,
            "f1": row.f1,
            "roc_auc": row.roc_auc,
            "accuracy": row.accuracy,
            "artifact_sha256": row.artifact_sha256,
            "dataset_version": row.dataset_version,
            "thresholds": json.loads(row.thresholds_json or "{}"),
            "release_gate_status": row.release_gate_status,
            "release_gate": json.loads(row.release_gate_json or "{}"),
            "lifecycle_stage": row.lifecycle_stage,
            "promoted_by_user_id": row.promoted_by_user_id,
            "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
            "notes": row.notes,
            "created_at": row.created_at.isoformat(),
        }

    def ensure_model_version(self, version_name: str, f1: float, roc_auc: float, accuracy: float = 0.0, artifact_sha256: str = "", dataset_version: str = "unknown", thresholds: dict | None = None, release_gate: dict | None = None, notes: str = ""):
        with self.session_scope() as session:
            existing = session.execute(select(ModelVersion).where(ModelVersion.version_name == version_name)).scalar_one_or_none()
            if existing:
                return self._model_to_dict(existing)
            row = ModelVersion(
                version_name=version_name,
                f1=f1,
                roc_auc=roc_auc,
                accuracy=accuracy,
                artifact_sha256=artifact_sha256,
                dataset_version=dataset_version,
                thresholds_json=json.dumps(thresholds or {}),
                release_gate_status=(release_gate or {}).get("gate_status", "unknown"),
                release_gate_json=json.dumps(release_gate or {}),
                lifecycle_stage="candidate",
                notes=notes,
            )
            session.add(row)
            session.flush()
            return self._model_to_dict(row)

    def list_model_versions(self, limit: int = 20):
        with self.session_scope() as session:
            rows = session.execute(select(ModelVersion).order_by(desc(ModelVersion.id)).limit(limit)).scalars().all()
            return [self._model_to_dict(row) for row in rows]

    def get_model_version(self, model_id: int):
        with self.session_scope() as session:
            row = session.get(ModelVersion, model_id)
            return self._model_to_dict(row) if row else None

    def promote_model(self, model_id: int, actor_user_id: int, target_stage: str, notes: str = ""):
        with self.session_scope() as session:
            row = session.get(ModelVersion, model_id)
            if not row:
                raise ValueError("model_not_found")
            row.lifecycle_stage = target_stage
            row.promoted_by_user_id = actor_user_id
            row.promoted_at = utc_now()
            row.notes = notes
            self._add_audit_log(session, actor_user_id=actor_user_id, action="model_promoted", target_type="model_version", target_id=str(model_id), details={"target_stage": target_stage})
            return self._model_to_dict(row)

    def _retraining_to_dict(self, row: RetrainingRun) -> dict:
        return {
            "id": row.id,
            "run_name": row.run_name,
            "status": row.status,
            "base_model_version": row.base_model_version,
            "candidate_model_name": row.candidate_model_name,
            "source_dataset_version": row.source_dataset_version,
            "approved_feedback_count": row.approved_feedback_count,
            "output_dir": row.output_dir,
            "metrics": json.loads(row.metrics_json or "{}"),
            "drift_profile": json.loads(row.drift_profile_json or "{}"),
            "improvements": json.loads(row.improvements_json or "{}"),
            "release_gate": json.loads(row.release_gate_json or "{}"),
            "recommendation": row.recommendation,
            "notes": row.notes,
            "created_at": row.created_at.isoformat(),
        }

    def save_retraining_run(self, run_name, status, base_model_version, candidate_model_name, source_dataset_version, approved_feedback_count, output_dir, metrics, drift_profile, improvements, release_gate, recommendation, notes, actor_user_id=None):
        with self.session_scope() as session:
            row = RetrainingRun(
                run_name=run_name,
                status=status,
                base_model_version=base_model_version,
                candidate_model_name=candidate_model_name,
                source_dataset_version=source_dataset_version,
                approved_feedback_count=approved_feedback_count,
                output_dir=output_dir,
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                drift_profile_json=json.dumps(drift_profile, ensure_ascii=False),
                improvements_json=json.dumps(improvements, ensure_ascii=False),
                release_gate_json=json.dumps(release_gate, ensure_ascii=False),
                recommendation=recommendation,
                notes=notes,
            )
            session.add(row)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="shadow_retraining_run", target_type="retraining_run", target_id=str(row.id), details={"recommendation": row.recommendation, "approved_feedback_count": row.approved_feedback_count})
            return self._retraining_to_dict(row)

    def list_retraining_runs(self, limit: int = 20):
        with self.session_scope() as session:
            rows = session.execute(select(RetrainingRun).order_by(desc(RetrainingRun.id)).limit(limit)).scalars().all()
            return [self._retraining_to_dict(row) for row in rows]
