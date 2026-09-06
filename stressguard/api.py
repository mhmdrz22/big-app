from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stressguard.agents import DEFAULT_AGENTS
from stressguard.data_store import DataStore
from stressguard.dependencies import get_current_user, require_roles
from stressguard.drift import DriftMonitor
from stressguard.jobs import execute_shadow_retraining_job, load_training_metrics
from stressguard.ml import StressModel
from stressguard.rate_limit import InMemoryRateLimiter
from stressguard.release_gate import build_release_gate_report
from stressguard.schemas import AnalyzeRequest, BootstrapAdminRequest, CreateUserRequest, DIRECTION_DELTA, EnqueueJobRequest, FeedbackRequest, JobActionRequest, LoginRequest, PromoteModelRequest, RegisterModelRequest, ReviewFeedbackRequest, RunShadowRetrainingRequest
from stressguard.safety import evaluate_feedback
from stressguard.alerts import send_drift_alert
from stressguard.security import create_access_token
from stressguard.settings import AppSettings, load_settings


def create_app(custom_settings: AppSettings | None = None) -> FastAPI:
    settings = custom_settings or load_settings()
    settings.model_dir.mkdir(exist_ok=True)
    settings.retraining_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title=settings.app_name, version="3.0.0")
    templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))
    app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")

    store = DataStore(settings.database_url)
    store.init_db()
    model = StressModel(
        settings.model_dir,
        default_low=settings.low_risk_threshold,
        default_high=settings.high_risk_threshold,
        default_decision=settings.decision_threshold,
    )
    drift_monitor = DriftMonitor(settings.model_dir)
    limiter = InMemoryRateLimiter()
    app.state.settings = settings
    app.state.store = store

    def ensure_model_version():
        metrics = load_training_metrics(settings.model_dir)
        if not metrics:
            return
        version_name = metrics.get("selected_model", "untrained")
        top = next((item for item in metrics.get("candidates", []) if item.get("model") == version_name), None)
        if top:
            store.ensure_model_version(version_name=version_name, f1=float(top.get("f1", 0.0)), roc_auc=float(top.get("roc_auc", 0.0) or 0.0), notes="Registered from current metrics artifact.")

    ensure_model_version()

    def client_key(request: Request, suffix: str) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
        return f"{suffix}:{ip}"

    def build_token(user: dict) -> dict:
        token = create_access_token(subject=str(user["id"]), email=user["email"], roles=user["roles"], secret=settings.auth_secret, algorithm=settings.auth_algorithm, expires_minutes=settings.access_token_exp_minutes)
        return {"access_token": token, "token_type": "bearer", "user": user}

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin(request: Request):
        return templates.TemplateResponse(request, "admin.html")

    @app.get("/api/v1/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": model.artifact is not None,
            "model_name": model.model_name,
            "thresholds": model.thresholds,
            "app_env": settings.app_env,
            "database_backend": "postgresql" if settings.database_url.startswith("postgresql") else "sqlite",
        }

    @app.post("/api/v1/auth/bootstrap-admin")
    async def bootstrap_admin(payload: BootstrapAdminRequest):
        if not settings.bootstrap_admin_enabled:
            raise HTTPException(status_code=403, detail="bootstrap_admin_disabled")
        try:
            user = store.bootstrap_admin(email=payload.email, full_name=payload.full_name, password=payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return build_token(user)

    @app.post("/api/v1/auth/login")
    async def login(payload: LoginRequest):
        user = store.authenticate_user(email=payload.email, password=payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="invalid_credentials")
        return build_token(user)

    @app.get("/api/v1/auth/me")
    async def me(user=Depends(get_current_user)):
        return user

    @app.post("/api/v1/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request):
        allowed, retry_after = limiter.check(client_key(request, "analyze"), settings.analyze_rate_limit_per_min)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"rate_limited:{retry_after}")
        text = payload.text.strip()
        result = model.predict_text(text)
        assessment_id = store.save_assessment(text=text, text_consent=payload.store_text_consent or settings.store_text_by_default, locale=payload.locale, model_name=result["model_name"], predicted_label=result.get("predicted_label"), probability=result.get("probability"), triage_level=result["triage_level"], analysis_mode=result["analysis_mode"])
        return {**result, "assessment_id": assessment_id}

    @app.post("/api/v1/feedback")
    async def feedback(payload: FeedbackRequest, request: Request):
        allowed, retry_after = limiter.check(client_key(request, "feedback"), settings.feedback_rate_limit_per_min)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"rate_limited:{retry_after}")
        assessment = store.get_assessment(payload.assessment_id)
        if not assessment:
            raise HTTPException(status_code=404, detail="assessment_not_found")

        model_label = assessment["predicted_label"]
        model_probability = float(assessment["probability"] or 0.5)
        analysis_mode = assessment["analysis_mode"]

        # The model's verdict is the anchor; the user reports the *direction*
        # of the mismatch (not the magnitude). Legacy `model_was_correct` is
        # mapped to direction for backward compatibility.
        base_model_label = 0 if model_label is None else int(model_label)
        direction = payload.direction
        if direction is None:
            if payload.model_was_correct is True:
                direction = "agree"
            elif payload.model_was_correct is False:
                direction = "over"
            else:
                direction = "unsure"

        # Hard label stays binary. Soft-target calibration comes from
        # soft_label_delta — never a stronger correction than a bounded nudge.
        if direction == "agree":
            proposed_label = base_model_label
        elif direction == "under":
            proposed_label = 1  # user said their stress was higher
        elif direction == "over":
            proposed_label = 0  # user said their stress was lower
        else:  # unsure
            proposed_label = base_model_label

        # Bounded soft delta — symmetric ±0.30 max from a vote; clamps at ±0.40.
        raw_delta = DIRECTION_DELTA[direction]
        if direction == "under" and base_model_label == 1:
            # Model already said stress; "under" reads as strong confirmation.
            raw_delta = +0.10
        elif direction == "over" and base_model_label == 0:
            raw_delta = +0.10
        soft_label_delta = float(max(-0.40, min(+0.40, raw_delta)))

        quality = evaluate_feedback(text=payload.text, proposed_label=proposed_label, model_label=base_model_label, model_probability=model_probability, note=payload.note, duplicate_count=store.count_similar_feedback(payload.text), consent_to_research=payload.consent_to_research, analysis_mode=analysis_mode)
        if direction == "unsure":
            quality["reasons"] = quality["reasons"] + ["کاربر از نتیجه مطمئن نبوده؛ این بازخورد با احتیاط بررسی می‌شود."]
            if quality["action"] == "approved_limited_weight":
                quality["action"] = "review_required"
                quality["sample_weight"] = 0.0

        feedback_id = store.save_feedback(
            assessment_id=payload.assessment_id,
            text=payload.text,
            proposed_label=proposed_label,
            note=payload.note,
            model_label=base_model_label,
            model_probability=model_probability,
            trust_score=quality["trust_score"],
            action=quality["action"],
            sample_weight=quality["sample_weight"],
            reasons=quality["reasons"],
            consent_to_research=payload.consent_to_research,
            user_agrees_with_result=(direction == "agree"),
            direction=direction,
            soft_label_delta=soft_label_delta,
        )
        messages = {
            "reject": "بازخورد ذخیره شد اما برای آموزش قابل استفاده نیست.",
            "review_required": "بازخوردت ثبت شد و وارد صف بازبینی مدیر شد؛ مدل به‌صورت خودکار تغییر نمی‌کند.",
            "approved_limited_weight": "بازخوردت ثبت شد و با وزن محدود کاندید بازآموزی شد؛ استفاده نهایی منوط به تأیید مدیر است.",
        }
        return {"feedback_id": feedback_id, **quality, "direction": direction, "soft_label_delta": soft_label_delta, "proposed_label": proposed_label, "message": messages[quality["action"]]}

    @app.get("/api/v1/admin/summary")
    async def admin_summary(user=Depends(require_roles("admin", "reviewer", "ml_engineer", "safety_reviewer", "auditor"))):
        drift_report = drift_monitor.build_report(store.list_recent_assessments(limit=200), store.list_reviewed_feedback_for_quality(limit=200))
        return {
            "store": store.summary(),
            "feedback_queue": store.list_feedback_queue(limit=10),
            "agents": DEFAULT_AGENTS,
            "training_metrics": load_training_metrics(settings.model_dir),
            "drift_report": drift_report,
            "policy": {"non_diagnostic": True, "quarantine_feedback": True, "default_text_storage": settings.store_text_by_default},
            "viewer": user,
        }

    @app.get("/api/v1/admin/report/stakeholder.pdf")
    async def stakeholder_pdf(user=Depends(require_roles("admin", "reviewer", "ml_engineer", "safety_reviewer", "auditor"))):
        from stressguard.reporting import build_stakeholder_pdf
        drift_report = drift_monitor.build_report(store.list_recent_assessments(limit=200), store.list_reviewed_feedback_for_quality(limit=200))
        metrics = load_training_metrics(settings.model_dir)
        pdf_bytes = build_stakeholder_pdf(store.summary(), drift_report, metrics)
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=aramnegar-stakeholder-report.pdf"})

    @app.get("/api/v1/admin/feedback-queue")
    async def feedback_queue(user=Depends(require_roles("admin", "reviewer", "safety_reviewer"))):
        return {"items": store.list_feedback_queue(limit=50), "viewer": user}

    @app.get("/api/v1/admin/assessment-series")
    async def assessment_series(user=Depends(require_roles("admin", "reviewer", "ml_engineer", "safety_reviewer", "auditor"))):
        return {"items": store.list_assessment_series(limit=60), "viewer": user}

    @app.post("/api/v1/admin/feedback-queue/{feedback_id}/review")
    async def review_feedback(feedback_id: int, payload: ReviewFeedbackRequest, user=Depends(require_roles("admin", "reviewer", "safety_reviewer"))):
        try:
            updated = store.transition_feedback_review(
                feedback_id=feedback_id,
                actor_user_id=user["id"],
                review_action=payload.review_action,
                review_notes=payload.review_notes,
                actor_roles=user["roles"],
            )
        except ValueError as exc:
            detail = str(exc)
            status = 404 if detail == "feedback_not_found" else 400
            raise HTTPException(status_code=status, detail=detail)
        return {"item": updated, "viewer": user}

    @app.get("/api/v1/admin/feedback-series")
    async def feedback_series(user=Depends(require_roles("admin", "reviewer", "ml_engineer", "safety_reviewer", "auditor"))):
        return {"items": store.list_feedback_series(limit=120), "viewer": user}

    @app.get("/api/v1/admin/model-registry")
    async def model_registry(user=Depends(require_roles("admin", "ml_engineer", "auditor"))):
        return {"items": store.list_model_versions(), "viewer": user}

    @app.post("/api/v1/admin/model-registry/register-current")
    async def register_current_model(payload: RegisterModelRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        metrics = load_training_metrics(settings.model_dir)
        if not metrics:
            raise HTTPException(status_code=404, detail="training_metrics_not_found")
        version_name = metrics.get("selected_model", "unknown")
        top = next((item for item in metrics.get("candidates", []) if item.get("model") == version_name), None)
        if not top:
            raise HTTPException(status_code=404, detail="selected_model_metrics_not_found")
        drift_report_data = drift_monitor.build_report(store.list_recent_assessments(limit=200), store.list_reviewed_feedback_for_quality(limit=200))
        artifact_path = settings.model_dir / "stress_model.joblib"
        gate = build_release_gate_report(top, drift_report_data, artifact_path.exists())
        item = store.register_model_version(
            actor_user_id=user["id"],
            version_name=version_name,
            f1=float(top.get("f1", 0.0)),
            roc_auc=float(top.get("roc_auc", 0.0)),
            accuracy=float(top.get("accuracy", 0.0)),
            dataset_version=payload.dataset_version,
            artifact_path=artifact_path,
            thresholds=metrics.get("thresholds", {}),
            release_gate=gate,
            notes=payload.notes or "Registered from current model artifacts.",
        )
        return {"item": item, "viewer": user}

    @app.post("/api/v1/admin/model-registry/{model_id}/promote")
    async def promote_model(model_id: int, payload: PromoteModelRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        try:
            item = store.promote_model_version(model_id=model_id, actor_user_id=user["id"], target_stage=payload.target_stage, notes=payload.notes)
        except ValueError as exc:
            detail = str(exc)
            status = 404 if detail == "model_not_found" else 400
            raise HTTPException(status_code=status, detail=detail)
        return {"item": item, "viewer": user}

    @app.get("/api/v1/admin/release-gate/current")
    async def current_release_gate(user=Depends(require_roles("admin", "ml_engineer", "auditor"))):
        metrics = load_training_metrics(settings.model_dir)
        if not metrics:
            raise HTTPException(status_code=404, detail="training_metrics_not_found")
        version_name = metrics.get("selected_model", "unknown")
        top = next((item for item in metrics.get("candidates", []) if item.get("model") == version_name), None)
        if not top:
            raise HTTPException(status_code=404, detail="selected_model_metrics_not_found")
        drift_report_data = drift_monitor.build_report(store.list_recent_assessments(limit=200), store.list_reviewed_feedback_for_quality(limit=200))
        gate = build_release_gate_report(top, drift_report_data, (settings.model_dir / "stress_model.joblib").exists())
        return {"item": gate, "viewer": user}

    @app.get("/api/v1/admin/drift-report")
    async def drift_report(user=Depends(require_roles("admin", "ml_engineer", "auditor", "safety_reviewer"))):
        report = drift_monitor.build_report(store.list_recent_assessments(limit=200), store.list_reviewed_feedback_for_quality(limit=200))
        snapshot_id = store.save_drift_snapshot(report)
        report["snapshot_id"] = snapshot_id
        if settings.alert_webhook_url and report.get("drift_level") in ("warning", "critical"):
            send_drift_alert(settings.alert_webhook_url, report, model.metrics, settings.alert_webhook_secret)
        return {"item": report, "viewer": user}

    @app.post("/api/v1/admin/retraining/run-shadow")
    async def run_shadow_retraining_endpoint(payload: RunShadowRetrainingRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        try:
            result = execute_shadow_retraining_job(settings, store, payload.model_dump(), actor_user_id=user["id"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"item": result["retraining_run"], "candidate_model": result.get("candidate_model"), "viewer": user}

    @app.post("/api/v1/admin/retraining/enqueue-shadow")
    async def enqueue_shadow_retraining(payload: EnqueueJobRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        job = store.enqueue_job(actor_user_id=user["id"], job_type="shadow_retraining", payload=payload.model_dump())
        return {"item": job, "viewer": user}

    @app.get("/api/v1/admin/retraining/runs")
    async def list_retraining_runs(user=Depends(require_roles("admin", "ml_engineer", "auditor"))):
        return {"items": store.list_retraining_runs(limit=20), "viewer": user}

    @app.get("/api/v1/admin/jobs")
    async def list_jobs(user=Depends(require_roles("admin", "ml_engineer", "auditor"))):
        return {"items": store.list_jobs(limit=50), "viewer": user}

    @app.get("/api/v1/admin/jobs/{job_id}")
    async def get_job(job_id: int, user=Depends(require_roles("admin", "ml_engineer", "auditor"))):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        return {"item": job, "viewer": user}

    @app.post("/api/v1/admin/jobs/{job_id}/retry")
    async def retry_job(job_id: int, payload: JobActionRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        try:
            item = store.retry_job(job_id=job_id, actor_user_id=user["id"], max_attempts=settings.max_job_attempts, notes=payload.notes)
        except ValueError as exc:
            detail = str(exc)
            status = 404 if detail == "job_not_found" else 400
            raise HTTPException(status_code=status, detail=detail)
        return {"item": item, "viewer": user}

    @app.post("/api/v1/admin/jobs/{job_id}/cancel")
    async def cancel_job(job_id: int, payload: JobActionRequest, user=Depends(require_roles("admin", "ml_engineer"))):
        try:
            item = store.cancel_job(job_id=job_id, actor_user_id=user["id"], notes=payload.notes)
        except ValueError as exc:
            detail = str(exc)
            status = 404 if detail == "job_not_found" else 400
            raise HTTPException(status_code=status, detail=detail)
        return {"item": item, "viewer": user}

    @app.get("/api/v1/admin/notifications")
    async def notifications(user=Depends(require_roles("admin", "ml_engineer", "auditor", "reviewer", "safety_reviewer"))):
        return {"items": store.list_notifications(limit=30), "viewer": user}

    @app.get("/api/v1/admin/audit-logs")
    async def audit_logs(user=Depends(require_roles("admin", "auditor"))):
        return {"items": store.list_audit_logs(limit=100), "viewer": user}

    @app.get("/api/v1/admin/roles")
    async def roles(user=Depends(require_roles("admin"))):
        return {"items": store.list_roles(), "viewer": user}

    @app.get("/api/v1/admin/users")
    async def users(user=Depends(require_roles("admin"))):
        return {"items": store.list_users(), "viewer": user}

    @app.post("/api/v1/admin/users")
    async def create_user(payload: CreateUserRequest, user=Depends(require_roles("admin"))):
        try:
            created = store.create_user(actor_user_id=user["id"], email=payload.email, full_name=payload.full_name, password=payload.password, roles=payload.roles)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return created

    return app
