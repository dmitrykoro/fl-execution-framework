"""add local dashboard url to experiment

Revision ID: 6b7c9f8d2a10
Revises: 20505e671899
Create Date: 2026-03-25 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6b7c9f8d2a10'
down_revision: Union[str, Sequence[str], None] = '20505e671899'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('experiment_event', sa.Column('local_dashboard_url', sa.String(length=1000), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('experiment_event', 'local_dashboard_url')
