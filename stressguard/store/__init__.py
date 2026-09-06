"""Data access layer — split from monolithic data_store.py."""
from stressguard.store.assessments import AssessmentStore
from stressguard.store.feedback import FeedbackStore
from stressguard.store.models import ModelStore, UserStore
from stressguard.store.base import BaseStore


class DataStore(BaseStore, AssessmentStore, FeedbackStore, ModelStore, UserStore):
    """Unified facade — backward compatible with existing imports."""
    pass
