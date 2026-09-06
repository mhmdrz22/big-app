"""Feedback quarantine CRUD + review transitions."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import desc, func, select

from stressguard.models import FeedbackQuarantine
from stressguard.privacy import hash_text, safe_excerpt


def utc_now():
    return datetime.now(timezone.utc)


class FeedbackStore:
    def _feedback_to_dict(self, row: FeedbackQuarantine) -> dict:
        return {
            "id": row.id,
            "assessment_id": row.assessment_id,
            "text_excerpt": row.text_excerpt,
            "proposed_label": row.proposed_label,
            "note": row.note,
            "model_label": row.model_label,
            "model_probability": row.model_probability,
            "trust_score": row.trust_score,
            "action": row.action,
            "sample_weight": row.sample_weight,
            "consent_to_research": row.consent_to_research,
            "user_agrees_with_result": row.user_agrees_with_result,
            "direction": row.direction,
            "soft_label_delta": row.soft_label_delta,
            "review_state": row.review_state,
            "reviewer_user_id": row.reviewer_user_id,
            "review_notes": row.review_notes,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
            "second_reviewer_user_id": row.second_reviewer_user_id,
            "second_review_notes": row.second_review_notes,
            "second_reviewed_at": row.second_reviewed_at.isoformat() if row.second_reviewed_at else None,
            "reasons": json.loads(row.reasons_json),
            "created_at": row.created_at.isoformat(),
        }

    def save_feedback(self, assessment_id, text, proposed_label, note, model_label, model_probability, trust_score, action, sample_weight, reasons, consent_to_research, user_agrees_with_result, direction=None, soft_label_delta=None, actor_user_id=None) -> int:
        text_hash = hash_text(text)
        stored_text = text if consent_to_research else None
        with self.session_scope() as session:
            row = FeedbackQuarantine(
                assessment_id=assessment_id,
                text=stored_text,
                text_hash=text_hash,
                text_excerpt=safe_excerpt(text),
                proposed_label=proposed_label,
                note=note,
                model_label=model_label,
                model_probability=model_probability,
                trust_score=trust_score,
                action=action,
                sample_weight=sample_weight,
                consent_to_research=bool(consent_to_research),
                user_agrees_with_result=user_agrees_with_result,
                direction=direction,
                soft_label_delta=soft_label_delta,
                reasons_json=json.dumps(reasons, ensure_ascii=False),
            )
            session.add(row)
            session.flush()
            self._add_audit_log(session, actor_user_id=actor_user_id, action="feedback_submitted", target_type="feedback_quarantine", target_id=str(row.id), details={"action": action, "assessment_id": assessment_id, "consent_to_research": bool(consent_to_research)})
            return int(row.id)

    def transition_feedback_review(self, feedback_id: int, actor_user_id: int, review_action: str, review_notes: str, actor_roles: list[str] | None = None) -> dict:
        actor_roles = actor_roles or []
        is_admin = "admin" in actor_roles
        with self.session_scope() as session:
            row = session.get(FeedbackQuarantine, feedback_id)
            if not row:
                raise ValueError("feedback_not_found")

            if review_action == "start_review":
                if row.review_state != "queued":
                    raise ValueError("feedback_not_in_queue")
                row.review_state = "in_review"
                row.reviewer_user_id = actor_user_id
                row.review_notes = review_notes
                row.reviewed_at = utc_now()
            elif review_action == "approve":
                if row.review_state == "queued":
                    row.review_state = "pending_second_review"
                    row.reviewer_user_id = actor_user_id
                    row.review_notes = review_notes
                    row.reviewed_at = utc_now()
                elif row.review_state == "pending_second_review":
                    if row.reviewer_user_id == actor_user_id and not is_admin:
                        raise ValueError("second_reviewer_must_be_distinct")
                    row.review_state = "approved"
                    row.second_reviewer_user_id = actor_user_id
                    row.second_review_notes = review_notes
                    row.second_reviewed_at = utc_now()
                elif row.review_state == "in_review":
                    row.review_state = "approved"
                    row.second_reviewer_user_id = actor_user_id
                    row.second_review_notes = review_notes
                    row.second_reviewed_at = utc_now()
                else:
                    raise ValueError("invalid_review_transition")
            elif review_action == "reject":
                row.review_state = "rejected"
                row.reviewer_user_id = actor_user_id
                row.review_notes = review_notes
                row.reviewed_at = utc_now()
            elif review_action == "return_to_queue":
                row.review_state = "queued"
                row.reviewer_user_id = None
                row.review_notes = None
                row.reviewed_at = None
            else:
                raise ValueError("invalid_review_action")

            self._add_audit_log(session, actor_user_id=actor_user_id, action=f"feedback_{review_action}", target_type="feedback_quarantine", target_id=str(row.id), details={"new_state": row.review_state, "review_notes": review_notes})
            return self._feedback_to_dict(row)

    def list_feedback_queue(self, limit: int = 10):
        with self.session_scope() as session:
            rows = session.execute(select(FeedbackQuarantine).order_by(desc(FeedbackQuarantine.id)).limit(limit)).scalars().all()
            return [self._feedback_to_dict(row) for row in rows]

    def list_approved_feedback_candidates(self, limit: int = 500) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(
                select(FeedbackQuarantine)
                .where(
                    FeedbackQuarantine.review_state == "approved",
                    FeedbackQuarantine.consent_to_research.is_(True),
                    FeedbackQuarantine.text.is_not(None),
                )
                .order_by(desc(FeedbackQuarantine.id))
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "text": row.text,
                    "proposed_label": row.proposed_label,
                    "sample_weight": row.sample_weight,
                    "model_label": row.model_label,
                    "model_probability": row.model_probability,
                    "reviewer_user_id": row.reviewer_user_id,
                    "second_reviewer_user_id": row.second_reviewer_user_id,
                    "direction": row.direction,
                    "soft_label_delta": row.soft_label_delta,
                }
                for row in rows
                if row.second_reviewer_user_id is not None
            ]

    def list_feedback_series(self, limit: int = 120) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(select(FeedbackQuarantine).order_by(FeedbackQuarantine.id.asc()).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "user_agrees_with_result": row.user_agrees_with_result,
                    "direction": row.direction,
                    "soft_label_delta": row.soft_label_delta,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def list_reviewed_feedback_for_quality(self, limit: int = 200) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(
                select(FeedbackQuarantine)
                .where(FeedbackQuarantine.review_state.in_(["approved", "rejected"]))
                .order_by(desc(FeedbackQuarantine.id))
                .limit(limit)
            ).scalars().all()
            return [self._feedback_to_dict(row) for row in rows]

    def count_similar_feedback(self, text: str) -> int:
        with self.session_scope() as session:
            return session.execute(select(func.count()).select_from(FeedbackQuarantine).where(FeedbackQuarantine.text_hash == hash_text(text))).scalar_one()
