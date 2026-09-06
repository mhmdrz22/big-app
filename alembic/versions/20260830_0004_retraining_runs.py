"""retraining runs

Revision ID: 20260830_0004
Revises: 20260830_0003
Create Date: 2026-08-30 13:08:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260830_0004"
down_revision = "20260830_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retraining_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("trigger_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("base_model_version", sa.String(length=255), nullable=False),
        sa.Column("candidate_model_name", sa.String(length=255), nullable=True),
        sa.Column("source_dataset_version", sa.String(length=255), nullable=False),
        sa.Column("approved_feedback_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_dir", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("recommendation", sa.String(length=64), nullable=False, server_default="hold"),
        sa.Column("report_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("retraining_runs")
