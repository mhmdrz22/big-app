"""drift monitoring

Revision ID: 20260830_0002
Revises: 20260830_0001
Create Date: 2026-08-30 12:54:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260830_0002"
down_revision = "20260830_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("assessments", sa.Column("char_length", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("assessments", sa.Column("token_length", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("assessments", sa.Column("arabic_char_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("assessments", sa.Column("latin_char_count", sa.Integer(), nullable=False, server_default="0"))

    op.create_table(
        "drift_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("population_size", sa.Integer(), nullable=False),
        sa.Column("psi_probability", sa.Float(), nullable=True),
        sa.Column("psi_length", sa.Float(), nullable=True),
        sa.Column("jsd_triage", sa.Float(), nullable=True),
        sa.Column("jsd_script", sa.Float(), nullable=True),
        sa.Column("quality_proxy_agreement", sa.Float(), nullable=True),
        sa.Column("drift_level", sa.String(length=32), nullable=False),
        sa.Column("report_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("drift_snapshots")
    op.drop_column("assessments", "latin_char_count")
    op.drop_column("assessments", "arabic_char_count")
    op.drop_column("assessments", "token_length")
    op.drop_column("assessments", "char_length")
