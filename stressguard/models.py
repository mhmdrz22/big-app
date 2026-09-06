from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc)

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("user_id", "role_id", name="uq_user_role"),
)


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    roles = relationship("Role", secondary=user_roles, lazy="joined")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=True)
    text_hash = Column(String(64), nullable=False, index=True)
    text_excerpt = Column(String(255), nullable=False)
    text_consent = Column(Boolean, default=False, nullable=False)
    locale = Column(String(16), nullable=True)
    model_name = Column(String(128), nullable=True)
    predicted_label = Column(Integer, nullable=True)
    probability = Column(Float, nullable=True)
    char_length = Column(Integer, nullable=False, default=0)
    token_length = Column(Integer, nullable=False, default=0)
    arabic_char_count = Column(Integer, nullable=False, default=0)
    latin_char_count = Column(Integer, nullable=False, default=0)
    triage_level = Column(String(64), nullable=False)
    analysis_mode = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class DriftSnapshot(Base):
    __tablename__ = "drift_snapshots"

    id = Column(Integer, primary_key=True)
    window_size = Column(Integer, nullable=False)
    population_size = Column(Integer, nullable=False)
    psi_probability = Column(Float, nullable=True)
    psi_length = Column(Float, nullable=True)
    jsd_triage = Column(Float, nullable=True)
    jsd_script = Column(Float, nullable=True)
    quality_proxy_agreement = Column(Float, nullable=True)
    drift_level = Column(String(32), nullable=False)
    report_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class FeedbackQuarantine(Base):
    __tablename__ = "feedback_quarantine"

    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id", ondelete="SET NULL"), nullable=True)
    text = Column(Text, nullable=True)
    text_hash = Column(String(64), nullable=False, index=True)
    text_excerpt = Column(String(255), nullable=False)
    proposed_label = Column(Integer, nullable=False)
    note = Column(Text, nullable=True)
    model_label = Column(Integer, nullable=False)
    model_probability = Column(Float, nullable=False)
    trust_score = Column(Float, nullable=False)
    action = Column(String(64), nullable=False)
    sample_weight = Column(Float, nullable=False)
    consent_to_research = Column(Boolean, default=False, nullable=False)
    user_agrees_with_result = Column(Boolean, nullable=True)
    direction = Column(String(16), nullable=True)  # agree | under | over | unsure
    soft_label_delta = Column(Float, nullable=True)  # bounded ±0.40 nudge
    review_state = Column(String(64), default="queued", nullable=False)
    reviewer_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    review_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    second_reviewer_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    second_review_notes = Column(Text, nullable=True)
    second_reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reasons_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    version_name = Column(String(255), unique=True, nullable=False)
    f1 = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    artifact_sha256 = Column(String(64), nullable=True, index=True)
    dataset_version = Column(String(255), nullable=True)
    thresholds_json = Column(Text, nullable=False, default="{}")
    release_gate_status = Column(String(32), nullable=False, default="blocked")
    release_gate_json = Column(Text, nullable=False, default="{}")
    lifecycle_stage = Column(String(32), nullable=False, default="candidate")
    promoted_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class RetrainingRun(Base):
    __tablename__ = "retraining_runs"

    id = Column(Integer, primary_key=True)
    trigger_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    base_model_version = Column(String(255), nullable=False)
    candidate_model_name = Column(String(255), nullable=True)
    source_dataset_version = Column(String(255), nullable=False)
    approved_feedback_count = Column(Integer, nullable=False, default=0)
    output_dir = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="queued")
    recommendation = Column(String(64), nullable=False, default="hold")
    report_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class BackgroundJob(Base):
    __tablename__ = "background_jobs"

    id = Column(Integer, primary_key=True)
    trigger_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    job_type = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="queued", index=True)
    payload_json = Column(Text, nullable=False, default="{}")
    result_json = Column(Text, nullable=False, default="{}")
    error_text = Column(Text, nullable=True)
    worker_name = Column(String(128), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SystemNotification(Base):
    __tablename__ = "system_notifications"

    id = Column(Integer, primary_key=True)
    level = Column(String(32), nullable=False, index=True)
    category = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    related_job_id = Column(Integer, ForeignKey("background_jobs.id", ondelete="SET NULL"), nullable=True)
    related_model_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    is_read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    actor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(128), nullable=False, index=True)
    target_type = Column(String(128), nullable=False)
    target_id = Column(String(128), nullable=True)
    details_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
