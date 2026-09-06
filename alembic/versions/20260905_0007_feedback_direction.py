"""add direction and soft_label_delta to feedback_quarantine."""
from alembic import op
import sqlalchemy as sa


revision = "20260905_0007"
down_revision = "20260830_0006"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feedback_quarantine") as batch:
        batch.add_column(sa.Column("direction", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("soft_label_delta", sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table("feedback_quarantine") as batch:
        batch.drop_column("soft_label_delta")
        batch.drop_column("direction")
