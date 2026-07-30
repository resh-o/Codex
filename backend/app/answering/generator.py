"""Structured answer generation.

Model choice: **Gemini** (``gemini-3.6-flash`` by default, see
``Settings.answer_model``). Two reasons, in order:

* The project already authenticates to the Gemini API for embeddings, so the
  answering layer adds a model, not a second vendor, a second key, and a second
  failure mode.
* Its ``response_schema`` mode takes a Pydantic model directly and constrains
  decoding to that JSON schema, which is what lets the "every claim has a
  citation" rule be structural. ``Claim.citations`` is ``minItems: 1`` in the
  schema the API enforces, and pydantic re-checks it on parse -- so a citation-
  less claim is rejected at two independent points and never reaches the
  validator.

Mode: JSON schema-constrained output (``response_mime_type='application/json'``
plus ``response_schema``), not tool-calling. There is exactly one thing to
produce and no side effects to trigger, so a forced function call would be the
same constraint with an extra envelope to unwrap.

Everything that can go wrong with a response -- empty, non-JSON, JSON of the
wrong shape -- raises :class:`GenerationError`. Nothing here degrades a bad
response into a half-built ``Answer``: an ``Answer`` object exists only if the
model actually produced a schema-valid one.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Protocol, Sequence

from pydantic import ValidationError

from ..config import ConfigError, Settings
from .models import Answer, GeneratedAnswer
from .prompt import SYSTEM_PROMPT, ContextChunk, build_user_prompt

logger = logging.getLogger(__name__)

#: How much of a bad response to quote in an error message.
_ERROR_EXCERPT_CHARS = 300


class GenerationError(RuntimeError):
    """The model call failed, or returned something that is not a valid Answer."""


class GenerateFn(Protocol):
    """Low-level call: prompts in, raw JSON text out. Injectable for tests."""

    def __call__(self, *, system: str, user: str) -> str: ...


class AnswerGenerator:
    """Prompts a model for a schema-constrained :class:`Answer`."""

    def __init__(
        self,
        settings: Settings,
        generate_fn: Optional[GenerateFn] = None,
    ) -> None:
        self._settings = settings
        # Built lazily so constructing the service needs no API key; tests pass
        # their own and never touch the SDK.
        self._generate_fn = generate_fn

    @property
    def model(self) -> str:
        return self._settings.answer_model

    def generate(
        self,
        query: str,
        chunks: Sequence[ContextChunk],
        feedback: Optional[str] = None,
    ) -> Answer:
        """Generate one answer. ``feedback`` carries a prior attempt's failures."""
        user_prompt = build_user_prompt(query, chunks, feedback=feedback)
        fn = self._get_generate_fn()

        try:
            raw = fn(system=SYSTEM_PROMPT, user=user_prompt)
        except ConfigError:
            raise
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK/transport failure
            raise GenerationError(f"Answer generation request failed: {exc}") from exc

        generated = self._parse(raw)
        return Answer.from_generated(query, generated)

    # -- internals ----------------------------------------------------------

    def _get_generate_fn(self) -> GenerateFn:
        if self._generate_fn is None:
            self._generate_fn = _build_sdk_generate_fn(self._settings)
        return self._generate_fn

    def _parse(self, raw: str) -> GeneratedAnswer:
        if raw is None or not raw.strip():
            raise GenerationError(
                "The model returned an empty response instead of an answer."
            )
        text = _strip_code_fence(raw.strip())

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"The model returned malformed JSON ({exc}). "
                f"Response began: {_excerpt(text)}"
            ) from exc

        if not isinstance(payload, dict):
            raise GenerationError(
                "The model returned a JSON "
                f"{type(payload).__name__}, not an answer object. "
                f"Response began: {_excerpt(text)}"
            )

        try:
            return GeneratedAnswer.model_validate(payload)
        except ValidationError as exc:
            # The most likely violation is a claim with an empty `citations`
            # array -- schema-enforced upstream, re-checked here so a provider
            # that ignores minItems still cannot produce an uncited claim.
            raise GenerationError(
                f"The model's response did not match the Answer schema: {exc}"
            ) from exc


def _strip_code_fence(text: str) -> str:
    """Unwrap a ```json ...``` fence if the model added one despite JSON mode."""
    if not text.startswith("```"):
        return text
    body = text[3:]
    if body[:4].lower().startswith("json"):
        body = body[4:]
    closing = body.rfind("```")
    if closing != -1:
        body = body[:closing]
    return body.strip()


def _excerpt(text: str) -> str:
    if len(text) <= _ERROR_EXCERPT_CHARS:
        return repr(text)
    return repr(text[:_ERROR_EXCERPT_CHARS] + "…")


def _build_sdk_generate_fn(settings: Settings) -> GenerateFn:
    """Construct the real google-genai-backed generation function."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ConfigError(
            "google-genai is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    api_key = settings.require_gemini()
    client = genai.Client(api_key=api_key)

    def _generate(*, system: str, user: str) -> str:
        response = client.models.generate_content(
            model=settings.answer_model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                # The Pydantic model *is* the contract: the SDK converts it to
                # the JSON schema the decoder is constrained by.
                response_schema=GeneratedAnswer,
                # Grounded extraction, not prose: near-zero temperature keeps
                # the model copying line ranges rather than improvising them.
                temperature=settings.answer_temperature,
            ),
        )
        text = response.text
        if not text:
            # A blocked or truncated response has no .text; surface why.
            reason = getattr(response, "prompt_feedback", None)
            raise GenerationError(
                "The model returned no content"
                + (f" (prompt_feedback: {reason})" if reason else "")
                + "."
            )
        return text

    return _generate
