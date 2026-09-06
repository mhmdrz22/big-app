"""initial schema

Revision ID: 20260830_0001
Revises: 
Create Date: 2026-08-30 12:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260830_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
    )
    op.create_index("ix_roles_name", "roles", ["name"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_roles",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
        sa.UniqueConstraint("user_id", "role_id", name="uq_user_role"),
    )

    op.create_table(
        "assessments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("text_excerpt", sa.String(length=255), nullable=False),
        sa.Column("text_consent", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("locale", sa.String(length=16), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("predicted_label", sa.Integer(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("triage_level", sa.String(length=64), nullable=False),
        sa.Column("analysis_mode", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_assessments_text_hash", "assessments", ["text_hash"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version_name", sa.String(length=255), nullable=False),
        sa.Column("f1", sa.Float(), nullable=False),
        sa.Column("roc_auc", sa.Float(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_model_versions_version_name", "model_versions", ["version_name"], unique=True)

    op.create_table(
        "feedback_quarantine",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("assessment_id", sa.Integer(), sa.ForeignKey("assessments.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("text_excerpt", sa.String(length=255), nullable=False),
        sa.Column("proposed_label", sa.Integer(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("model_label", sa.Integer(), nullable=False),
        sa.Column("model_probability", sa.Float(), nullable=False),
        sa.Column("trust_score", sa.Float(), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("sample_weight", sa.Float(), nullable=False),
        sa.Column("consent_to_research", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("user_agrees_with_result", sa.Boolean(), nullable=True),
        sa.Column("review_state", sa.String(length=64), nullable=False, server_default="queued"),
        sa.Column("reviewer_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reasons_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feedback_quarantine_text_hash", "feedback_quarantine", ["text_hash"], unique=False)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("target_type", sa.String(length=128), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_feedback_quarantine_text_hash", table_name="feedback_quarantine")
    op.drop_table("feedback_quarantine")
    op.drop_index("ix_model_versions_version_name", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_assessments_text_hash", table_name="assessments")
    op.drop_table("assessments")
    op.drop_table("user_roles")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_roles_name", table_name="roles")
    op.drop_table("roles")
