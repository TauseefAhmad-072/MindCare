"""
modules/text/preprocessing.py — Input validation & light preprocessing for
the Text Mental-Health Analysis module.

Note: the heavy-duty text cleaning (lowercasing, URL/mention stripping,
punctuation normalisation) is already implemented inside the reused
MindCarePredictor._clean(). This file only validates raw input BEFORE it
reaches that pipeline.
"""
from typing import List
from .config import MAX_TEXT_LENGTH_CHARS, MAX_BATCH_SIZE


class TextValidationError(ValueError):
    """Raised when input text fails validation before reaching the model."""


def validate_text(text: str) -> str:
    if text is None:
        raise TextValidationError("`text` field is required.")
    if not isinstance(text, str):
        raise TextValidationError("`text` must be a string.")
    stripped = text.strip()
    if not stripped:
        raise TextValidationError("`text` must not be empty.")
    if len(stripped) > MAX_TEXT_LENGTH_CHARS:
        raise TextValidationError(
            f"`text` exceeds max length of {MAX_TEXT_LENGTH_CHARS} characters."
        )
    return stripped


def validate_batch(texts: List[str]) -> List[str]:
    if texts is None:
        raise TextValidationError("`texts` field is required for batch prediction.")
    if not isinstance(texts, list):
        raise TextValidationError("`texts` must be a list of strings.")
    if len(texts) == 0:
        raise TextValidationError("`texts` must contain at least one item.")
    if len(texts) > MAX_BATCH_SIZE:
        raise TextValidationError(
            f"Batch size {len(texts)} exceeds maximum of {MAX_BATCH_SIZE}."
        )
    return [validate_text(t) for t in texts]
