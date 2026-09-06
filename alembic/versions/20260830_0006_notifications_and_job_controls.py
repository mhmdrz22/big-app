"""notifications and job controls

Revision ID: 20260830_0006
Revises: 20260830_0005
Create Date: 2026-08-30 13:16:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260830_0006"
down_revision = "20260830_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_notifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("level", sa.String(length=32), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("related_job_id", sa.Integer(), sa.ForeignKey("background_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("related_model_id", sa.Integer(), sa.ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_system_notifications_level", "system_notifications", ["level"], unique=False)
    op.create_index("ix_system_notifications_category", "system_notifications", ["category"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_system_notifications_category", table_name="system_notifications")
    op.drop_index("ix_system_notifications_level", table_name="system_notifications")
    op.drop_table("system_notifications")
