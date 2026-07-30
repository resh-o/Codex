"""Stage 4: answer generation with mechanically-validated citations."""

from __future__ import annotations

from .answer_service import (
    MAX_ATTEMPTS,
    AnswerResult,
    AnswerService,
    NoContextError,
    SourceRef,
    StrippedClaim,
)
from .generator import AnswerGenerator, GenerationError
from .models import Answer, Citation, Claim, GeneratedAnswer
from .prompt import SYSTEM_PROMPT, build_retry_feedback, build_user_prompt
from .validator import (
    CitationValidation,
    ClaimValidation,
    InvalidReason,
    ValidationResult,
    validate_answer,
)

__all__ = [
    "Answer",
    "Citation",
    "Claim",
    "GeneratedAnswer",
    "AnswerGenerator",
    "GenerationError",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "build_retry_feedback",
    "validate_answer",
    "ValidationResult",
    "ClaimValidation",
    "CitationValidation",
    "InvalidReason",
    "AnswerService",
    "AnswerResult",
    "NoContextError",
    "StrippedClaim",
    "SourceRef",
    "MAX_ATTEMPTS",
]
