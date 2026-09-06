from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=20, max_length=4000)
    store_text_consent: bool = False
    locale: str = "fa"


# Allowed feedback directions. The model output is anchored; the user only
# reports WHERE their true state sat relative to the model — never writes a
# raw training label.
#   agree  → verdict felt right
#   under  → "my stress was higher than the model said"
#   over   → "my stress was lower than the model said"
#   unsure → user cannot compare; admin-only review
FeedbackDirection = Literal["agree", "under", "over", "unsure"]

DIRECTION_LABELS_FA = {
    "agree":  "حدوداً درست بود",
    "under":  "استرس من بیشتر بود",
    "over":   "استرس من کمتر بود",
    "unsure": "مطمئن نیستم",
}

# How strongly each direction nudges the soft label, bounded so a single
# vote can never swing the model off the calibrated prior.
DIRECTION_DELTA = {
    "agree":  +0.05,   # gentle confirmation (calibration only)
    "under":  +0.30,   # pushes the next soft label up
    "over":   -0.30,   # pushes the next soft label down
    "unsure":  0.00,   # excluded from retraining — admin review only
}


class FeedbackRequest(BaseModel):
    """Feedback model: the model's verdict is the anchor; the user reports the
    *direction* of the mismatch. The server converts that to a bounded soft
    label delta — the user never writes a training label directly."""
    assessment_id: int
    text: str = Field(min_length=20, max_length=4000)
    # New primary field: directional feedback.
    direction: Optional[FeedbackDirection] = None
    # Legacy field kept for backward compatibility. If both are provided,
    # `direction` wins. When only the legacy is sent:
    #   True  → "agree"
    #   False → "over"  (the user's most common binary feedback)
    #   None  → "unsure"
    model_was_correct: Optional[bool] = None
    note: str = Field(default="", max_length=1000)
    consent_to_research: bool = False


class BootstrapAdminRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=10, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=10, max_length=128)


class CreateUserRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=10, max_length=128)
    roles: list[str] = Field(min_length=1)


class ReviewFeedbackRequest(BaseModel):
    review_action: str = Field(pattern="^(start_review|approve|reject|return_to_queue)$")
    review_notes: str = Field(default="", max_length=1500)


class RegisterModelRequest(BaseModel):
    dataset_version: str = Field(default="dreaddit-phase1", max_length=255)
    notes: str = Field(default="", max_length=2000)


class PromoteModelRequest(BaseModel):
    target_stage: str = Field(pattern="^(candidate|staging|canary|champion|archived)$")
    notes: str = Field(default="", max_length=2000)


class RunShadowRetrainingRequest(BaseModel):
    dataset_version: str = Field(default="dreaddit-phase1+approved-feedback", max_length=255)
    notes: str = Field(default="", max_length=2000)


class EnqueueJobRequest(BaseModel):
    dataset_version: str = Field(default="dreaddit-phase1+approved-feedback", max_length=255)
    notes: str = Field(default="", max_length=2000)


class JobActionRequest(BaseModel):
    notes: str = Field(default="", max_length=1000)
