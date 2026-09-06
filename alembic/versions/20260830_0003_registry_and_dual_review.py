"""registry and dual review

Revision ID: 20260830_0003
Revises: 20260830_0002
Create Date: 2026-08-30 13:05:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260830_0003"
down_revision = "20260830_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.add_column("feedback_quarantine", sa.Column("second_reviewer_user_id", sa.Integer(), nullable=True))
    op.add_column("feedback_quarantine", sa.Column("second_review_notes", sa.Text(), nullable=True))
    op.add_column("feedback_quarantine", sa.Column("second_reviewed_at", sa.DateTime(timezone=True), nullable=True))
    if not is_sqlite:
        op.create_foreign_key("fk_feedback_quarantine_second_reviewer_user", "feedback_quarantine", "users", ["second_reviewer_user_id"], ["id"], ondelete="SET NULL")

    op.add_column("model_versions", sa.Column("accuracy", sa.Float(), nullable=True))
    op.add_column("model_versions", sa.Column("artifact_sha256", sa.String(length=64), nullable=True))
    op.add_column("model_versions", sa.Column("dataset_version", sa.String(length=255), nullable=True))
    op.add_column("model_versions", sa.Column("thresholds_json", sa.Text(), nullable=False, server_default="{}"))
    op.add_column("model_versions", sa.Column("release_gate_status", sa.String(length=32), nullable=False, server_default="blocked"))
    op.add_column("model_versions", sa.Column("release_gate_json", sa.Text(), nullable=False, server_default="{}"))
    op.add_column("model_versions", sa.Column("lifecycle_stage", sa.String(length=32), nullable=False, server_default="candidate"))
    op.add_column("model_versions", sa.Column("promoted_by_user_id", sa.Integer(), nullable=True))
    op.add_column("model_versions", sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_model_versions_artifact_sha256", "model_versions", ["artifact_sha256"], unique=False)
    if not is_sqlite:
        op.create_foreign_key("fk_model_versions_promoted_by_user", "model_versions", "users", ["promoted_by_user_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if not is_sqlite:
        op.drop_constraint("fk_model_versions_promoted_by_user", "model_versions", type_="foreignkey")
    op.drop_index("ix_model_versions_artifact_sha256", table_name="model_versions")
    op.drop_column("model_versions", "promoted_at")
    op.drop_column("model_versions", "promoted_by_user_id")
    op.drop_column("model_versions", "lifecycle_stage")
    op.drop_column("model_versions", "release_gate_json")
    op.drop_column("model_versions", "release_gate_status")
    op.drop_column("model_versions", "thresholds_json")
    op.drop_column("model_versions", "dataset_version")
    op.drop_column("model_versions", "artifact_sha256")
    op.drop_column("model_versions", "accuracy")

    if not is_sqlite:
        op.drop_constraint("fk_feedback_quarantine_second_reviewer_user", "feedback_quarantine", type_="foreignkey")
    op.drop_column("feedback_quarantine", "second_reviewed_at")
    op.drop_column("feedback_quarantine", "second_review_notes")
    op.drop_column("feedback_quarantine", "second_reviewer_user_id")
