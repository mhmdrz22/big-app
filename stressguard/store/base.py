"""Base store with engine, session, and audit log."""
from __future__ import annotations

import json
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from stressguard.models import AuditLog, Base
from stressguard.settings import AppSettings


class BaseStore:
    def __init__(self, database_url: str, base_dir=None):
        self.database_url = database_url
        self.base_dir = base_dir
        self.engine = create_engine(database_url, echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._init_schema()

    def _init_schema(self):
        Base.metadata.create_all(self.engine)
        if self.database_url.startswith("sqlite"):
            self._run_alembic_migrations()

    def _run_alembic_migrations(self):
        from alembic import command as alembic_command
        from alembic.config import Config as AlembicConfig
        if not self.base_dir:
            return
        cfg = AlembicConfig(str(self.base_dir / "alembic.ini"))
        cfg.set_main_option("script_location", str(self.base_dir / "alembic"))
        cfg.set_main_option("sqlalchemy.url", self.database_url)
        try:
            alembic_command.upgrade(cfg, "head")
        except Exception:
            pass  # migrations may already be applied

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _add_audit_log(self, session, actor_user_id, action, target_type, target_id, details):
        row = AuditLog(
            actor_user_id=actor_user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details_json=json.dumps(details, ensure_ascii=False),
        )
        session.add(row)
