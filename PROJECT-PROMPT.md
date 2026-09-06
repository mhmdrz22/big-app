# AramNegar v6 — Project Context

## Current State
- **Model**: Hybrid ensemble (TF-IDF LogReg + TF-IDF CalibratedSVM + Numeric LIWC LogReg), soft voting
- **Output**: `stress_score` 0-100 + `stress_band` (پایین/متوسط/بالا/خیلی بالا)
- **Features**: 109 numeric columns (LIWC 94 + DAL 9 + Social 3 + Syntax 2) extracted offline via `stressguard/features.py`
- **Feedback**: 3-direction (agree/under/over/unsure) with `soft_label_delta` bounded ±0.40
- **Anti-overfit**: 20% feedback cap, sample_weight 0.10-0.35, locked test, champion/challenger gate
- **Dashboard**: Executive + Technical tabs (Chart.js local)
- **PDF**: `/api/v1/admin/report/stakeholder.pdf` (Amiri font)
- **Alerting**: Webhook on drift warning/critical
- **Docker**: `compose.prod.yaml` (Nginx + Gunicorn + PostgreSQL + Worker)
- **Persian**: `train_persian.py` scaffolding ready

## File Counts
- `stressguard/`: 21 Python modules
- `data/`: 2 CSVs + README + feedback-store-test.py
- `templates/`: 3 HTML
- `static/`: 4 JS/CSS
- `scripts/`: 4 Python
- `tests/`: 25 passing tests
- `alembic/versions/`: 7 migrations

## Known Issues
- `data_store.py` is monolithic (~900 lines). `stressguard/store/` has split modules but they are incomplete — not wired up.
- Persian model is a TF-IDF placeholder; needs ParsBERT fine-tune for production.
- Admin.js chart rendering is basic; no real-time updates.

## Next Steps
1. Complete `stressguard/store/` split (assessments, feedback, models, users)
2. ParsBERT fine-tune with real Persian stress data
3. Add SHAP explainability endpoint
4. Prometheus metrics endpoint
5. A/B testing between model versions
