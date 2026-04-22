"""
UndertriAI — OpenEnv Client
Use this to connect to a running UndertriAI environment server.
"""

from typing import Optional

try:
    from openenv.core.env_client import EnvClient  # type: ignore
except ImportError:
    class EnvClient:
        """Stub when openenv-core is not installed."""
        def __init__(self, base_url: str): self.base_url = base_url

from .models import (
    BailAction, CaseObservation,
    RequestDocumentAction, FlagInconsistencyAction,
    CrossReferencePrecedentAction, ComputeStatutoryEligibilityAction,
    AssessSuretyAction, ClassifyBailTypeAction, SubmitMemoAction,
)


class UndertriAIEnv(EnvClient):
    """
    Client for the UndertriAI bail assessment environment.

    Usage (async):
        async with UndertriAIEnv(base_url="https://openenv-undertrial-ai.hf.space") as env:
            obs = await env.reset(stage=1)
            result = await env.step(ComputeStatutoryEligibilityAction(...))
            result = await env.step(SubmitMemoAction(...))
            print(result.reward)

    Usage (sync):
        with UndertriAIEnv(base_url="...").sync() as env:
            obs = env.reset(stage=1)
            result = env.step(SubmitMemoAction(...))
    """
    pass


# Convenience re-exports so users only need to import from undertrial_ai
__all__ = [
    "UndertriAIEnv",
    "BailAction",
    "CaseObservation",
    "RequestDocumentAction",
    "FlagInconsistencyAction",
    "CrossReferencePrecedentAction",
    "ComputeStatutoryEligibilityAction",
    "AssessSuretyAction",
    "ClassifyBailTypeAction",
    "SubmitMemoAction",
]
