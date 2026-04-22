"""UndertriAI package — exports for users."""
from .client import UndertriAIEnv
from .models import (
    BailAction, CaseObservation,
    RequestDocumentAction, FlagInconsistencyAction,
    CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
    AssessSuretyAction, ClassifyBailTypeAction, SubmitMemoAction,
)

__version__ = "1.0.0"
__all__ = [
    "UndertriAIEnv",
    "BailAction", "CaseObservation",
    "RequestDocumentAction", "FlagInconsistencyAction",
    "CrossReferencePrecedentAction", "ComputeStatutoryEligibilityAction",
    "AssessSuretyAction", "ClassifyBailTypeAction", "SubmitMemoAction",
]
