from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, desc, func, inspect, select
from sqlalchemy.orm import Session, joinedload, sessionmaker

from stressguard.models import Assessment, AuditLog, BackgroundJob, Base, DriftSnapshot, FeedbackQuarantine, ModelVersion, RetrainingRun, Role, SystemNotification, User
from stressguard.privacy import hash_text, safe_excerpt
from stressguard.security import hash_password, verify_password


ROLE_DESCRIPTIONS = {
    "admin": "مدیریت کامل سامانه و کاربران",
    "reviewer": "بررسی صف بازخورد و داده‌های قرنطینه",
    "ml_engineer": "پایش مدل و مدیریت انتشار نسخه‌ها",
    "safety_reviewer": "بررسی موارد حساس و safety queue",
    "auditor": "مشاهده لاگ‌ها و ممیزی بدون ویرایش",
}

STAGE_ORDER = {"candidate": 0, "staging": 1, "canary": 2, "champion": 3, "archived": 4}


class DataStore:
    def __init__(self, database_url: str):
        # Normalize PostgreSQL URLs to the psycopg3 dialect (requirements installs
        # psycopg[binary] v3; the bare postgresql:// scheme would demand psycopg2).
        if database_url.startswith("postgresql://") and not database_url.startswith("postgresql+psycopg://"):
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.database_url = database_url
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)

    @contextmanager
    def session_scope(self):
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_db(self):
        Base.metadata.create_all(self.engine)
        self._heal_schema_drift()
        with self.session_scope() as session:
            self._ensure_roles(session)

    # Columns the application code depends on, mapped to the table that must have them.
    # If an older local SQLite file is missing any of these, we try to heal the schema
    # with Alembic migrations automatically instead of crashing with "no such column".
    _REQUIRED_COLUMNS = {
        "model_versions": ["accuracy", "artifact_sha256", "dataset_version", "lifecycle_stage", "release_gate_status"],
        "feedback_quarantine": ["second_reviewer_user_id", "second_review_notes", "second_reviewed_at", "direction", "soft_label_delta"],
        "assessments": ["char_length", "token_length", "arabic_char_count", "latin_char_count"],
    }

    def _schema_gaps(self) -> list[str]:
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        gaps = []
        for table, columns in self._REQUIRED_COLUMNS.items():
            if table not in existing_tables:
                continue
            existing_cols = {col["name"] for col in inspector.get_columns(table)}
            for col in columns:
                if col not in existing_cols:
                    gaps.append(f"{table}.{col}")
        return gaps

    # Column -> SQLite type for direct ALTER TABLE healing.
    # Kept in sync with the SQLAlchemy models so old local DBs (created by
    # create_all before Alembic was wired in) get upgraded in place.
    _COLUMN_SQL_TYPES = {
        "direction": "VARCHAR(16)",
        "soft_label_delta": "FLOAT",
        "second_reviewer_user_id": "INTEGER",
        "second_review_notes": "TEXT",
        "second_reviewed_at": "DATETIME",
        "accuracy": "FLOAT",
        "artifact_sha256": "VARCHAR(64)",
        "dataset_version": "VARCHAR(255)",
        "lifecycle_stage": "VARCHAR(32)",
        "release_gate_status": "VARCHAR(32)",
        "char_length": "INTEGER",
        "token_length": "INTEGER",
        "arabic_char_count": "INTEGER",
        "latin_char_count": "INTEGER",
    }

    def _heal_schema_drift(self):
        """Add missing columns in place.

        SQLite: direct ALTER TABLE ADD COLUMN — robust even when the DB was
        created by create_all and has no alembic_version table.
        PostgreSQL: run Alembic migrations (they are the source of truth).
        """
        gaps = self._schema_gaps()
        if not gaps:
            return

        if self.database_url.startswith("sqlite"):
            with self.engine.begin() as conn:
                for gap in gaps:
                    table, col = gap.split(".", 1)
                    sql_type = self._COLUMN_SQL_TYPES.get(col, "TEXT")
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")
            remaining = self._schema_gaps()
            if remaining:
                raise RuntimeError(
                    "database_schema_outdated: "
                    f"local DB still missing {remaining} after ALTER TABLE heal."
                )
            return

        try:
            from alembic import command as alembic_command
            from alembic.config import Config as AlembicConfig
            from stressguard.settings import load_settings

            settings = load_settings()
            cfg = AlembicConfig(str(settings.base_dir / "alembic.ini"))
            cfg.set_main_option("script_location", str(settings.base_dir / "alembic"))
            cfg.set_main_option("sqlalchemy.url", settings.database_url)
            alembic_command.upgrade(cfg, "head")
        except Exception as exc:
            raise RuntimeError(
                "database_schema_outdated: "
                f"the local database is missing columns {gaps} and automatic migration failed ({exc}). "
                "Fix: delete the old database file inside the 'instance' folder and start again."
            ) from exc

    def _ensure_roles(self, session: Session):
        existing = {role.name for role in session.execute(select(Role)).scalars().all()}
        for name, description in ROLE_DESCRIPTIONS.items():
            if name not in existing:
                session.add(Role(name=name, description=description))

    def _user_to_dict(self, user: User) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "roles": sorted(role.name for role in user.roles),
            "created_at": user.created_at.isoformat(),
        }

    def count_users(self) -> int:
        with self.session_scope() as session:
            return int(session.execute(select(func.count()).select_from(User)).scalar_one())

    def bootstrap_admin(self, email: str, full_name: str, password: str) -> dict:
        with self.session_scope() as session:
            if session.execute(select(func.count()).select_from(User)).scalar_one() > 0:
                raise ValueError("bootstrap_disabled_after_first_user")
            admin_role = session.execute(select(Role).where(Role.name == "admin")).scalar_one()
            user = User(email=email.lower().strip(), full_name=full_name.strip(), password_hash=hash_password(password), is_active=True)
            user.roles = [admin_role]
            session.add(user)
            session.flush()
            self._add_audit_log(session, actor_user_id=user.id, action="bootstrap_admin", target_type="user", target_id=str(user.id), details={"email": user.email, "roles": ["admin"]})
            return self._user_to_dict(user)

    def upsert_admin(self, email: str, full_name: str, password: str) -> dict:
        """Create the admin if missing; if the email already exists, reset its password
        and make sure it has the admin role. Safe to run repeatedly in local dev."""
        normalized_email = email.lower().strip()
        with self.session_scope() as session:
            admin_role = session.execute(select(Role).where(Role.name == "admin")).scalar_one()
            user = session.execute(
                select(User).options(joinedload(User.roles)).where(User.email == normalized_email)
            ).unique().scalar_one_or_none()
            if user:
                user.password_hash = hash_password(password)
                user.is_active = True
                user.full_name = full_name.strip() or user.full_name
                if admin_role not in user.roles:
                    user.roles = list(user.roles) + [admin_role]
                action = "reset_admin"
            else:
                user = User(email=normalized_email, full_name=full_name.strip(), password_hash=hash_password(password), is_active=True)
                user.roles = [admin_role]
                session.add(user)
                action = "create_admin"
            session.flush()
            self._add_audit_log(session, actor_user_id=user.id, action=action, target_type="user", target_id=str(user.id), details={"email": user.email})
            return self._user_to_dict(user)

    def list_assessment_series(self, limit: int = 60) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(Assessment).order_by(desc(Assessment.id)).limit(limit)).scalars().all()
            rows = list(reversed(rows))
            return [
                {
                    "id": row.id,
                    "probability": row.probability,
                    "triage_level": row.triage_level,
                    "analysis_mode": row.analysis_mode,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def authenticate_user(self, email: str, password: str) -> dict | None:
        with self.session_scope() as session:
            user = session.execute(
                select(User).options(joinedload(User.roles)).where(User.email == email.lower().strip())
            ).unique().scalar_one_or_none()
            if not user or not user.is_active or not verify_password(password, user.password_hash):
                return None
            return self._user_to_dict(user)

    def get_user_by_id(self, user_id: int) -> dict | None:
        with self.session_scope() as session:
            user = session.execute(select(User).options(joinedload(User.roles)).where(User.id == user_id)).unique().scalar_one_or_none()
            return self._user_to_dict(user) if user else None

    def create_user(self, actor_user_id: int, email: str, full_name: str, password: str, roles: list[str]) -> dict:
        normalized_email = email.lower().strip()
        with self.session_scope() as session:
            existing = session.execute(select(User).where(User.email == normalized_email)).scalar_one_or_none()
            if existing:
                raise ValueError("email_already_exists")
            db_roles = session.execute(select(Role).where(Role.name.in_(roles))).scalars().all()
            found_roles = sorted(role.name for role in db_roles)
            if sorted(set(roles)) != found_roles:
                raise ValueError("invalid_roles")
            user = User(email=normalized_email, full_name=full_name.strip(), password_hash=hash_password(password), is_active=True)
            user.roles = db_roles
            session.add(user)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="create_user", target_type="user", target_id=str(user.id), details={"email": user.email, "roles": found_roles})
            return self._user_to_dict(user)

    def list_users(self) -> list[dict]:
        with self.session_scope() as session:
            users = session.execute(select(User).options(joinedload(User.roles)).order_by(User.id.asc())).unique().scalars().all()
            return [self._user_to_dict(user) for user in users]

    def list_roles(self) -> list[dict]:
        with self.session_scope() as session:
            roles = session.execute(select(Role).order_by(Role.name.asc())).scalars().all()
            return [{"name": role.name, "description": role.description} for role in roles]

    def _add_audit_log(self, session: Session, actor_user_id: int | None, action: str, target_type: str, target_id: str | None, details: dict):
        session.add(
            AuditLog(
                actor_user_id=actor_user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                details_json=json.dumps(details, ensure_ascii=False),
            )
        )

    def list_audit_logs(self, limit: int = 50) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "actor_user_id": row.actor_user_id,
                    "action": row.action,
                    "target_type": row.target_type,
                    "target_id": row.target_id,
                    "details": json.loads(row.details_json),
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def _feedback_to_dict(self, row: FeedbackQuarantine) -> dict:
        return {
            "id": row.id,
            "assessment_id": row.assessment_id,
            "text_excerpt": row.text_excerpt,
            "proposed_label": row.proposed_label,
            "note": row.note,
            "model_label": row.model_label,
            "model_probability": row.model_probability,
            "trust_score": row.trust_score,
            "action": row.action,
            "sample_weight": row.sample_weight,
            "consent_to_research": row.consent_to_research,
            "user_agrees_with_result": row.user_agrees_with_result,
            "direction": row.direction,
            "soft_label_delta": row.soft_label_delta,
            "review_state": row.review_state,
            "reviewer_user_id": row.reviewer_user_id,
            "review_notes": row.review_notes,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
            "second_reviewer_user_id": row.second_reviewer_user_id,
            "second_review_notes": row.second_review_notes,
            "second_reviewed_at": row.second_reviewed_at.isoformat() if row.second_reviewed_at else None,
            "reasons": json.loads(row.reasons_json),
            "created_at": row.created_at.isoformat(),
        }

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

    def _retraining_to_dict(self, row: RetrainingRun) -> dict:
        return {
            "id": row.id,
            "trigger_user_id": row.trigger_user_id,
            "base_model_version": row.base_model_version,
            "candidate_model_name": row.candidate_model_name,
            "source_dataset_version": row.source_dataset_version,
            "approved_feedback_count": row.approved_feedback_count,
            "output_dir": row.output_dir,
            "status": row.status,
            "recommendation": row.recommendation,
            "report": json.loads(row.report_json or "{}"),
            "created_at": row.created_at.isoformat(),
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    def _job_to_dict(self, row: BackgroundJob) -> dict:
        return {
            "id": row.id,
            "trigger_user_id": row.trigger_user_id,
            "job_type": row.job_type,
            "status": row.status,
            "payload": json.loads(row.payload_json or "{}"),
            "result": json.loads(row.result_json or "{}"),
            "error_text": row.error_text,
            "worker_name": row.worker_name,
            "attempts": row.attempts,
            "created_at": row.created_at.isoformat(),
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    def _notification_to_dict(self, row: SystemNotification) -> dict:
        return {
            "id": row.id,
            "level": row.level,
            "category": row.category,
            "title": row.title,
            "message": row.message,
            "related_job_id": row.related_job_id,
            "related_model_id": row.related_model_id,
            "is_read": row.is_read,
            "created_at": row.created_at.isoformat(),
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save_assessment(self, text: str, text_consent: bool, locale: str, model_name: str, predicted_label, probability, triage_level: str, analysis_mode: str) -> int:
        text_hash = hash_text(text)
        stored_text = text if text_consent else None
        char_length = len(text)
        token_length = len(text.split())
        arabic_char_count = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
        latin_char_count = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        with self.session_scope() as session:
            row = Assessment(
                text=stored_text,
                text_hash=text_hash,
                text_excerpt=safe_excerpt(text),
                text_consent=bool(text_consent),
                locale=locale,
                model_name=model_name,
                predicted_label=predicted_label,
                probability=probability,
                char_length=char_length,
                token_length=token_length,
                arabic_char_count=arabic_char_count,
                latin_char_count=latin_char_count,
                triage_level=triage_level,
                analysis_mode=analysis_mode,
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def get_assessment(self, assessment_id: int | None):
        if not assessment_id:
            return None
        with self.session_scope() as session:
            row = session.get(Assessment, assessment_id)
            if not row:
                return None
            return {
                "id": row.id,
                "predicted_label": row.predicted_label,
                "probability": row.probability,
                "char_length": row.char_length,
                "token_length": row.token_length,
                "arabic_char_count": row.arabic_char_count,
                "latin_char_count": row.latin_char_count,
                "analysis_mode": row.analysis_mode,
                "triage_level": row.triage_level,
            }

    def save_feedback(self, assessment_id, text, proposed_label, note, model_label, model_probability, trust_score, action, sample_weight, reasons, consent_to_research, user_agrees_with_result, direction=None, soft_label_delta=None, actor_user_id=None) -> int:
        text_hash = hash_text(text)
        stored_text = text if consent_to_research else None
        with self.session_scope() as session:
            row = FeedbackQuarantine(
                assessment_id=assessment_id,
                text=stored_text,
                text_hash=text_hash,
                text_excerpt=safe_excerpt(text),
                proposed_label=proposed_label,
                note=note,
                model_label=model_label,
                model_probability=model_probability,
                trust_score=trust_score,
                action=action,
                sample_weight=sample_weight,
                consent_to_research=bool(consent_to_research),
                user_agrees_with_result=user_agrees_with_result,
                direction=direction,
                soft_label_delta=soft_label_delta,
                reasons_json=json.dumps(reasons, ensure_ascii=False),
            )
            session.add(row)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="feedback_submitted", target_type="feedback_quarantine", target_id=str(row.id), details={"action": action, "assessment_id": assessment_id, "consent_to_research": bool(consent_to_research)})
            return int(row.id)

    def transition_feedback_review(self, feedback_id: int, actor_user_id: int, review_action: str, review_notes: str, actor_roles: list[str] | None = None) -> dict:
        actor_roles = actor_roles or []
        is_admin = "admin" in actor_roles
        with self.session_scope() as session:
            row = session.get(FeedbackQuarantine, feedback_id)
            if not row:
                raise ValueError("feedback_not_found")

            if review_action == "start_review":
                if row.review_state not in {"queued", "needs_changes"}:
                    raise ValueError("invalid_review_transition")
                row.review_state = "in_review"
                row.reviewer_user_id = actor_user_id
                row.review_notes = review_notes.strip() or row.review_notes
                row.reviewed_at = datetime.now(timezone.utc)
            elif review_action == "approve":
                if row.review_state in {"queued", "in_review", "needs_changes"}:
                    if is_admin:
                        # Local/single-operator installs have exactly one reviewer; the admin may
                        # finalize directly. The override is recorded so the audit trail shows it.
                        row.review_state = "approved"
                        row.reviewer_user_id = actor_user_id
                        row.review_notes = ((review_notes.strip() + " " if review_notes.strip() else "") + "[admin_finalize]").strip()
                        row.reviewed_at = datetime.now(timezone.utc)
                    else:
                        row.review_state = "pending_second_review"
                        row.reviewer_user_id = actor_user_id
                        row.review_notes = review_notes.strip()
                        row.reviewed_at = datetime.now(timezone.utc)
                elif row.review_state == "pending_second_review":
                    if row.reviewer_user_id == actor_user_id and not is_admin:
                        raise ValueError("second_reviewer_must_be_distinct")
                    row.review_state = "approved"
                    row.second_reviewer_user_id = actor_user_id
                    row.second_review_notes = review_notes.strip() or ("[admin_finalize]" if is_admin and row.reviewer_user_id == actor_user_id else "")
                    row.second_reviewed_at = datetime.now(timezone.utc)
                else:
                    raise ValueError("invalid_review_transition")
            elif review_action == "reject":
                if row.review_state not in {"queued", "in_review", "needs_changes", "pending_second_review"}:
                    raise ValueError("invalid_review_transition")
                row.review_state = "rejected"
                if row.reviewer_user_id and row.reviewer_user_id != actor_user_id:
                    row.second_reviewer_user_id = actor_user_id
                    row.second_review_notes = review_notes.strip()
                    row.second_reviewed_at = datetime.now(timezone.utc)
                else:
                    row.reviewer_user_id = actor_user_id
                    row.review_notes = review_notes.strip()
                    row.reviewed_at = datetime.now(timezone.utc)
            elif review_action == "return_to_queue":
                if row.review_state not in {"in_review", "approved", "rejected", "pending_second_review"}:
                    raise ValueError("invalid_review_transition")
                row.review_state = "needs_changes"
                if row.reviewer_user_id and row.reviewer_user_id != actor_user_id:
                    row.second_reviewer_user_id = actor_user_id
                    row.second_review_notes = review_notes.strip()
                    row.second_reviewed_at = datetime.now(timezone.utc)
                else:
                    row.reviewer_user_id = actor_user_id
                    row.review_notes = review_notes.strip()
                    row.reviewed_at = datetime.now(timezone.utc)
            else:
                raise ValueError("unknown_review_action")

            session.flush()
            self._add_audit_log(
                session,
                actor_user_id=actor_user_id,
                action=f"feedback_{review_action}",
                target_type="feedback_quarantine",
                target_id=str(row.id),
                details={"new_state": row.review_state, "review_notes": row.review_notes},
            )
            return self._feedback_to_dict(row)

    def list_approved_feedback_candidates(self, limit: int = 500) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(
                select(FeedbackQuarantine)
                .where(
                    FeedbackQuarantine.review_state == "approved",
                    FeedbackQuarantine.consent_to_research.is_(True),
                    FeedbackQuarantine.text.is_not(None),
                )
                .order_by(desc(FeedbackQuarantine.id))
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "text": row.text,
                    "proposed_label": row.proposed_label,
                    "sample_weight": row.sample_weight,
                    "model_label": row.model_label,
                    "model_probability": row.model_probability,
                    "reviewer_user_id": row.reviewer_user_id,
                    "second_reviewer_user_id": row.second_reviewer_user_id,
                    "direction": row.direction,
                    "soft_label_delta": row.soft_label_delta,
                }
                for row in rows
                if row.second_reviewer_user_id is not None
            ]

    def list_feedback_series(self, limit: int = 120) -> list[dict]:
        """Chronological feedback events for the learning-from-feedback chart."""
        with self.session_scope() as session:
            rows = session.execute(select(FeedbackQuarantine).order_by(FeedbackQuarantine.id.asc()).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat(),
                    "review_state": row.review_state,
                    "action": row.action,
                    "user_agrees_with_result": row.user_agrees_with_result,
                    "model_probability": row.model_probability,
                }
                for row in rows
            ]

    def count_similar_feedback(self, text: str) -> int:
        text_hash = hash_text(text)
        with self.session_scope() as session:
            return int(session.execute(select(func.count()).select_from(FeedbackQuarantine).where(FeedbackQuarantine.text_hash == text_hash)).scalar_one())

    def ensure_model_version(self, version_name: str, f1: float, roc_auc: float, notes: str):
        with self.session_scope() as session:
            exists = session.execute(select(ModelVersion).where(ModelVersion.version_name == version_name)).scalar_one_or_none()
            if not exists:
                session.add(ModelVersion(version_name=version_name, f1=f1, roc_auc=roc_auc, notes=notes, release_gate_status="blocked", release_gate_json="{}", thresholds_json="{}", lifecycle_stage="candidate"))

    def register_model_version(self, actor_user_id: int, version_name: str, f1: float, roc_auc: float, accuracy: float | None, dataset_version: str, artifact_path: Path, thresholds: dict, release_gate: dict, notes: str) -> dict:
        artifact_sha = self._sha256_file(Path(artifact_path)) if Path(artifact_path).exists() else None
        with self.session_scope() as session:
            existing = session.execute(select(ModelVersion).where(ModelVersion.version_name == version_name)).scalar_one_or_none()
            if existing:
                existing.f1 = f1
                existing.roc_auc = roc_auc
                existing.accuracy = accuracy
                existing.dataset_version = dataset_version
                existing.artifact_sha256 = artifact_sha
                existing.thresholds_json = json.dumps(thresholds, ensure_ascii=False)
                existing.release_gate_status = release_gate.get("gate_status", "blocked")
                existing.release_gate_json = json.dumps(release_gate, ensure_ascii=False)
                existing.notes = notes or existing.notes
                row = existing
            else:
                row = ModelVersion(
                    version_name=version_name,
                    f1=f1,
                    roc_auc=roc_auc,
                    accuracy=accuracy,
                    dataset_version=dataset_version,
                    artifact_sha256=artifact_sha,
                    thresholds_json=json.dumps(thresholds, ensure_ascii=False),
                    release_gate_status=release_gate.get("gate_status", "blocked"),
                    release_gate_json=json.dumps(release_gate, ensure_ascii=False),
                    lifecycle_stage="candidate",
                    notes=notes,
                )
                session.add(row)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="register_model", target_type="model_version", target_id=str(row.id), details={"version_name": version_name, "dataset_version": dataset_version, "release_gate_status": row.release_gate_status})
            return self._model_to_dict(row)

    def update_model_release_gate(self, model_id: int, release_gate: dict) -> dict:
        with self.session_scope() as session:
            row = session.get(ModelVersion, model_id)
            if not row:
                raise ValueError("model_not_found")
            row.release_gate_status = release_gate.get("gate_status", row.release_gate_status)
            row.release_gate_json = json.dumps(release_gate, ensure_ascii=False)
            session.flush()
            return self._model_to_dict(row)

    def promote_model_version(self, model_id: int, actor_user_id: int, target_stage: str, notes: str = "") -> dict:
        with self.session_scope() as session:
            row = session.get(ModelVersion, model_id)
            if not row:
                raise ValueError("model_not_found")
            if row.release_gate_status != "passed" and target_stage in {"staging", "canary", "champion"}:
                raise ValueError("release_gate_blocked")
            current_rank = STAGE_ORDER.get(row.lifecycle_stage, -1)
            target_rank = STAGE_ORDER.get(target_stage, -1)
            if target_rank == -1:
                raise ValueError("invalid_target_stage")
            if target_stage != "archived" and target_rank < current_rank:
                raise ValueError("invalid_stage_regression")
            if row.lifecycle_stage == "champion" and target_stage == "canary":
                raise ValueError("invalid_stage_regression")

            if target_stage == "champion":
                champions = session.execute(select(ModelVersion).where(ModelVersion.lifecycle_stage == "champion")).scalars().all()
                for champion in champions:
                    if champion.id != row.id:
                        champion.lifecycle_stage = "archived"

            row.lifecycle_stage = target_stage
            row.promoted_by_user_id = actor_user_id
            row.promoted_at = datetime.now(timezone.utc)
            if notes:
                row.notes = (row.notes + "\n" if row.notes else "") + notes.strip()
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="promote_model", target_type="model_version", target_id=str(row.id), details={"target_stage": target_stage, "release_gate_status": row.release_gate_status})
            return self._model_to_dict(row)

    def list_model_versions(self) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(ModelVersion).order_by(ModelVersion.id.desc()).limit(10)).scalars().all()
            return [self._model_to_dict(row) for row in rows]

    def list_recent_assessments(self, limit: int = 200) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(Assessment).order_by(desc(Assessment.id)).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "probability": row.probability,
                    "char_length": row.char_length,
                    "token_length": row.token_length,
                    "arabic_char_count": row.arabic_char_count,
                    "latin_char_count": row.latin_char_count,
                    "triage_level": row.triage_level,
                    "analysis_mode": row.analysis_mode,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def list_reviewed_feedback_for_quality(self, limit: int = 200) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(
                select(FeedbackQuarantine)
                .where(FeedbackQuarantine.review_state == "approved")
                .order_by(desc(FeedbackQuarantine.id))
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "model_label": row.model_label,
                    "proposed_label": row.proposed_label,
                    "model_probability": row.model_probability,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def save_drift_snapshot(self, report: dict) -> int:
        with self.session_scope() as session:
            row = DriftSnapshot(
                window_size=report["window_size"],
                population_size=report["population_size"],
                psi_probability=report["metrics"].get("psi_probability"),
                psi_length=report["metrics"].get("psi_length"),
                jsd_triage=report["metrics"].get("jsd_triage"),
                jsd_script=report["metrics"].get("jsd_script"),
                quality_proxy_agreement=report["metrics"].get("quality_proxy_agreement"),
                drift_level=report["drift_level"],
                report_json=json.dumps(report, ensure_ascii=False),
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def latest_drift_snapshot(self):
        with self.session_scope() as session:
            row = session.execute(select(DriftSnapshot).order_by(desc(DriftSnapshot.id)).limit(1)).scalar_one_or_none()
            if not row:
                return None
            data = json.loads(row.report_json)
            data["snapshot_id"] = row.id
            data["created_at"] = row.created_at.isoformat()
            return data

    def save_retraining_run(self, actor_user_id: int, report: dict) -> dict:
        with self.session_scope() as session:
            row = RetrainingRun(
                trigger_user_id=actor_user_id,
                base_model_version=report.get("base_model_version", "unknown"),
                candidate_model_name=report.get("candidate_model_name"),
                source_dataset_version=report.get("source_dataset_version", "unknown"),
                approved_feedback_count=int(report.get("approved_feedback_count", 0)),
                output_dir=report.get("output_dir", ""),
                status=report.get("status", "completed"),
                recommendation=report.get("recommendation", "hold"),
                report_json=json.dumps(report, ensure_ascii=False),
                completed_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="shadow_retraining_run", target_type="retraining_run", target_id=str(row.id), details={"recommendation": row.recommendation, "approved_feedback_count": row.approved_feedback_count})
            return self._retraining_to_dict(row)

    def list_retraining_runs(self, limit: int = 20) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(RetrainingRun).order_by(desc(RetrainingRun.id)).limit(limit)).scalars().all()
            return [self._retraining_to_dict(row) for row in rows]

    def enqueue_job(self, actor_user_id: int, job_type: str, payload: dict) -> dict:
        with self.session_scope() as session:
            row = BackgroundJob(
                trigger_user_id=actor_user_id,
                job_type=job_type,
                status="queued",
                payload_json=json.dumps(payload, ensure_ascii=False),
                result_json="{}",
                attempts=0,
            )
            session.add(row)
            session.flush()
            self._create_notification(
                session,
                level="info",
                category="job",
                title="Job queued",
                message=f"Job {row.job_type} با شناسه {row.id} در صف قرار گرفت.",
                related_job_id=row.id,
            )
            self._add_audit_log(session, actor_user_id=actor_user_id, action="enqueue_job", target_type="background_job", target_id=str(row.id), details={"job_type": job_type})
            return self._job_to_dict(row)

    def claim_next_job(self, worker_name: str, job_type: str | None = None) -> dict | None:
        with self.session_scope() as session:
            query = select(BackgroundJob).where(BackgroundJob.status == "queued")
            if job_type:
                query = query.where(BackgroundJob.job_type == job_type)
            if self.engine.dialect.name != "sqlite":
                query = query.with_for_update(skip_locked=True)
            row = session.execute(query.order_by(BackgroundJob.id.asc()).limit(1)).scalar_one_or_none()
            if not row:
                return None
            row.status = "running"
            row.worker_name = worker_name
            row.attempts = int(row.attempts or 0) + 1
            row.started_at = datetime.now(timezone.utc)
            session.flush()
            self._add_audit_log(session, actor_user_id=row.trigger_user_id, action="claim_job", target_type="background_job", target_id=str(row.id), details={"worker_name": worker_name})
            return self._job_to_dict(row)

    def complete_job(self, job_id: int, result: dict) -> dict:
        with self.session_scope() as session:
            row = session.get(BackgroundJob, job_id)
            if not row:
                raise ValueError("job_not_found")
            row.status = "completed"
            row.result_json = json.dumps(result, ensure_ascii=False)
            row.error_text = None
            row.completed_at = datetime.now(timezone.utc)
            session.flush()
            self._create_notification(
                session,
                level="success",
                category="job",
                title="Job completed",
                message=f"Job {row.job_type} با شناسه {row.id} با موفقیت تکمیل شد.",
                related_job_id=row.id,
            )
            self._add_audit_log(session, actor_user_id=row.trigger_user_id, action="complete_job", target_type="background_job", target_id=str(row.id), details={"status": "completed"})
            return self._job_to_dict(row)

    def fail_job(self, job_id: int, error_text: str) -> dict:
        with self.session_scope() as session:
            row = session.get(BackgroundJob, job_id)
            if not row:
                raise ValueError("job_not_found")
            row.status = "failed"
            row.error_text = error_text
            row.completed_at = datetime.now(timezone.utc)
            session.flush()
            self._create_notification(
                session,
                level="error",
                category="job",
                title="Job failed",
                message=f"Job {row.job_type} با شناسه {row.id} شکست خورد: {error_text}",
                related_job_id=row.id,
            )
            self._add_audit_log(session, actor_user_id=row.trigger_user_id, action="fail_job", target_type="background_job", target_id=str(row.id), details={"error_text": error_text})
            return self._job_to_dict(row)

    def retry_job(self, job_id: int, actor_user_id: int, max_attempts: int, notes: str = "") -> dict:
        with self.session_scope() as session:
            row = session.get(BackgroundJob, job_id)
            if not row:
                raise ValueError("job_not_found")
            if row.status not in {"failed", "cancelled"}:
                raise ValueError("job_retry_not_allowed")
            if int(row.attempts or 0) >= max_attempts:
                raise ValueError("job_retry_attempts_exhausted")
            row.status = "queued"
            row.error_text = None
            row.worker_name = None
            row.started_at = None
            row.completed_at = None
            session.flush()
            self._create_notification(
                session,
                level="warning",
                category="job",
                title="Job requeued",
                message=f"Job {row.job_type} با شناسه {row.id} برای تلاش دوباره در صف قرار گرفت. {notes}".strip(),
                related_job_id=row.id,
            )
            self._add_audit_log(session, actor_user_id=actor_user_id, action="retry_job", target_type="background_job", target_id=str(row.id), details={"notes": notes, "attempts": row.attempts})
            return self._job_to_dict(row)

    def cancel_job(self, job_id: int, actor_user_id: int, notes: str = "") -> dict:
        with self.session_scope() as session:
            row = session.get(BackgroundJob, job_id)
            if not row:
                raise ValueError("job_not_found")
            if row.status != "queued":
                raise ValueError("job_cancel_not_allowed")
            row.status = "cancelled"
            row.completed_at = datetime.now(timezone.utc)
            session.flush()
            self._create_notification(
                session,
                level="warning",
                category="job",
                title="Job cancelled",
                message=f"Job {row.job_type} با شناسه {row.id} لغو شد. {notes}".strip(),
                related_job_id=row.id,
            )
            self._add_audit_log(session, actor_user_id=actor_user_id, action="cancel_job", target_type="background_job", target_id=str(row.id), details={"notes": notes})
            return self._job_to_dict(row)

    def get_job(self, job_id: int) -> dict | None:
        with self.session_scope() as session:
            row = session.get(BackgroundJob, job_id)
            return self._job_to_dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(BackgroundJob).order_by(desc(BackgroundJob.id)).limit(limit)).scalars().all()
            return [self._job_to_dict(row) for row in rows]

    def _create_notification(self, session: Session, level: str, category: str, title: str, message: str, related_job_id: int | None = None, related_model_id: int | None = None):
        session.add(
            SystemNotification(
                level=level,
                category=category,
                title=title,
                message=message,
                related_job_id=related_job_id,
                related_model_id=related_model_id,
            )
        )

    def list_notifications(self, limit: int = 20) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(SystemNotification).order_by(desc(SystemNotification.id)).limit(limit)).scalars().all()
            return [self._notification_to_dict(row) for row in rows]

    def summary(self):
        with self.session_scope() as session:
            assessment_count = int(session.execute(select(func.count()).select_from(Assessment)).scalar_one())
            elevated_count = int(session.execute(select(func.count()).select_from(Assessment).where(Assessment.triage_level == "elevated")).scalar_one())
            uncertain_count = int(session.execute(select(func.count()).select_from(Assessment).where(Assessment.triage_level == "uncertain")).scalar_one())
            feedback_count = int(session.execute(select(func.count()).select_from(FeedbackQuarantine)).scalar_one())
            consented_texts = int(session.execute(select(func.count()).select_from(Assessment).where(Assessment.text_consent.is_(True))).scalar_one())
            avg_probability = session.execute(select(func.avg(Assessment.probability)).where(Assessment.probability.is_not(None))).scalar_one()
            action_rows = session.execute(select(FeedbackQuarantine.action, func.count()).group_by(FeedbackQuarantine.action)).all()
            review_rows = session.execute(select(FeedbackQuarantine.review_state, func.count()).group_by(FeedbackQuarantine.review_state)).all()
            return {
                "assessment_count": assessment_count,
                "elevated_count": elevated_count,
                "uncertain_count": uncertain_count,
                "average_probability": round(float(avg_probability or 0.0), 4),
                "feedback_count": feedback_count,
                "consented_texts": consented_texts,
                "feedback_actions": {action: int(count) for action, count in action_rows},
                "review_states": {state: int(count) for state, count in review_rows},
                "model_versions": self.list_model_versions(),
                "retraining_runs": self.list_retraining_runs(limit=5),
                "jobs": self.list_jobs(limit=5),
                "notifications": self.list_notifications(limit=5),
            }

    def list_feedback_queue(self, limit: int = 10):
        with self.session_scope() as session:
            rows = session.execute(select(FeedbackQuarantine).order_by(FeedbackQuarantine.id.desc()).limit(limit)).scalars().all()
            return [self._feedback_to_dict(row) for row in rows]
