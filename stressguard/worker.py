from __future__ import annotations

import time

from stressguard.data_store import DataStore
from stressguard.jobs import execute_shadow_retraining_job
from stressguard.settings import load_settings


def process_next_job_once(store: DataStore, settings, worker_name: str | None = None):
    job = store.claim_next_job(worker_name or settings.worker_name)
    if not job:
        return None
    try:
        if job["job_type"] == "shadow_retraining":
            result = execute_shadow_retraining_job(settings, store, job["payload"], actor_user_id=job.get("trigger_user_id"))
        else:
            raise ValueError(f"unsupported_job_type:{job['job_type']}")
        store.complete_job(job["id"], result)
        return result
    except Exception as exc:
        store.fail_job(job["id"], str(exc))
        raise


def main():
    settings = load_settings()
    settings.retraining_dir.mkdir(parents=True, exist_ok=True)
    store = DataStore(settings.database_url)
    store.init_db()
    while True:
        processed = process_next_job_once(store, settings, settings.worker_name)
        if processed is None:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
