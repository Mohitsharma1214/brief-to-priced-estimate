"""
LLM-Based Structured Extraction Module for Brief to Priced Estimate.

Extracts structured jewelry specifications (product_type, material, dimensions_mm, stone_count)
from unstructured customer sales briefs using a configurable LLM provider.

Core Architecture:
1. Provider Interface (LLMProvider): Decoupled from validation logic.
2. OpenRouterProvider: Default provider communicating via OpenRouter API using environment variables.
3. Bounded Retry Mechanism: Automatically retries malformed JSON or validation errors up to max_retries.
4. Calibrated Confidence Scoring: Does not blindly trust LLM claims; validates confidence
   against textual evidence and explicit deterministic rules.
5. Safe Failure Fallback: Never crashes on provider errors or timeouts; returns a graceful
   fallback with empty specifications and requires_human_review=True.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

# Load environment variables from .env
load_dotenv()

from app.schemas import (
    ConfidenceLevel,
    ExtractionResult,
    ExtractionStatus,
    FieldConfidence,
    MaterialEnum,
    ProductSpecification,
    ProductTypeEnum,
)

# Environment configuration defaults
DEFAULT_MODEL: str = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
DEFAULT_FALLBACK_MODEL: str = os.getenv("OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct")
DEFAULT_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
DEFAULT_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

SYSTEM_PROMPT = """You are an expert jewelry product specification extraction assistant.
Given a customer sales brief, extract structured jewelry specifications strictly matching this schema:

Fields:
1. "product_type": string or null. Allowed values: "ring", "pendant", "necklace", "bracelet", "earrings".
2. "material": string or null. Allowed values: "gold", "silver", "platinum". (Map "18k gold", "white gold", "yellow gold", "rose gold" to "gold"; "sterling silver", "925" to "silver"; "pt950" to "platinum").
3. "dimensions_mm": positive number (float) or null. The primary dimension in millimeters (e.g. inner diameter for rings, drop height for pendants/earrings, length for necklaces/bracelets).
4. "stone_count": non-negative integer or null. Number of stones/gems (e.g. 0 for plain band, 1 for solitaire).

CRITICAL RULES:
- If an attribute is NOT mentioned or cannot be unambiguously determined from the brief, set its value to null.
- NEVER fabricate, invent, or guess specifications not supported by the brief.
- Return ONLY a valid JSON object. Do not include introductory or concluding conversational text.

Example response:
{
  "product_type": "ring",
  "material": "gold",
  "dimensions_mm": 17.5,
  "stone_count": 1
}
"""


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate response text for the given prompt."""
        ...


class OpenRouterProvider:
    """
    OpenRouter API client implementing the LLMProvider protocol.
    Reads credentials and configuration strictly from environment variables.
    Supports primary model and automatic fallback model failover.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.fallback_model = fallback_model or DEFAULT_FALLBACK_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured. Set the environment variable "
                "or pass api_key to the provider."
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://brief-to-priced-estimate.local",
            "X-Title": "Brief to Priced Estimate Assessment",
        }

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as primary_err:
            if self.fallback_model and self.fallback_model != self.model:
                payload["model"] = self.fallback_model
                try:
                    response = requests.post(
                        endpoint, headers=headers, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()
                except Exception:
                    raise primary_err
            else:
                raise primary_err

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Provider returned response with no choices.")

        content = choices[0].get("message", {}).get("content", "")
        return content


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON dictionary from raw model output.
    Handles markdown code blocks and surrounding whitespace.
    """
    cleaned = text.strip()

    # Case 1: Markdown code block (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Case 2: Find outermost JSON braces if wrapped in extraneous text
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Extracted JSON must be a dictionary, got {type(data).__name__}")
    return data


def calibrate_field_confidences(
    brief: str, specs: ProductSpecification
) -> Dict[str, FieldConfidence]:
    """
    Calibrate field confidences using explicit evidence rules rather than
    uncalibrated self-reported model claims.
    """
    brief_lower = brief.lower()
    confidences: Dict[str, FieldConfidence] = {}

    # 1. Product Type
    if specs.product_type is None:
        confidences["product_type"] = FieldConfidence(
            confidence=ConfidenceLevel.LOW,
            reason="Product type not specified or recognized in sales brief.",
            evidence=None,
        )
    else:
        pt_val = specs.product_type.value
        # Check for explicit keywords
        synonyms = {
            "ring": ["ring", "band", "solitaire ring", "eternity"],
            "pendant": ["pendant", "charm", "locket", "drop"],
            "necklace": ["necklace", "choker", "chain", "collar"],
            "bracelet": ["bracelet", "bangle", "cuff", "wristlet"],
            "earrings": ["earring", "earrings", "stud", "studs", "hoop", "hoops", "huggie"],
        }
        found_kw = next((kw for kw in synonyms.get(pt_val, [pt_val]) if kw in brief_lower), None)
        if found_kw:
            confidences["product_type"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason=f"Explicit product type keyword '{found_kw}' found in brief text.",
                evidence=found_kw,
            )
        else:
            confidences["product_type"] = FieldConfidence(
                confidence=ConfidenceLevel.MEDIUM,
                reason=f"Product type inferred as '{pt_val}' from context without explicit keyword.",
                evidence=None,
            )

    # 2. Material
    if specs.material is None:
        confidences["material"] = FieldConfidence(
            confidence=ConfidenceLevel.LOW,
            reason="Material not specified in sales brief.",
            evidence=None,
        )
    else:
        mat_val = specs.material.value
        mat_keywords = {
            "gold": ["gold", "18k", "14k", "22k", "24k", "yellow gold", "white gold", "rose gold"],
            "silver": ["silver", "sterling", "925", "argent"],
            "platinum": ["platinum", "pt950", "plat"],
        }
        found_mat = next((kw for kw in mat_keywords.get(mat_val, [mat_val]) if kw in brief_lower), None)
        if found_mat:
            confidences["material"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason=f"Explicit material keyword '{found_mat}' found in brief text.",
                evidence=found_mat,
            )
        else:
            confidences["material"] = FieldConfidence(
                confidence=ConfidenceLevel.MEDIUM,
                reason=f"Material inferred as '{mat_val}' from general phrasing.",
                evidence=None,
            )

    # 3. Dimensions
    if specs.dimensions_mm is None:
        confidences["dimensions_mm"] = FieldConfidence(
            confidence=ConfidenceLevel.LOW,
            reason="Dimension not specified in sales brief.",
            evidence=None,
        )
    else:
        dim_num = str(int(specs.dimensions_mm)) if specs.dimensions_mm.is_integer() else f"{specs.dimensions_mm:.1f}"
        has_num = dim_num in brief_lower or f"{specs.dimensions_mm}" in brief_lower
        has_unit = any(u in brief_lower for u in ("mm", "cm", "inch", "size", "gauge", "diameter", "length"))
        if has_num and has_unit:
            confidences["dimensions_mm"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason="Explicit numerical dimension with unit verified in brief text.",
                evidence=f"{specs.dimensions_mm}mm",
            )
        elif has_num:
            confidences["dimensions_mm"] = FieldConfidence(
                confidence=ConfidenceLevel.MEDIUM,
                reason="Numerical value detected in brief but unit context was implicit.",
                evidence=f"{specs.dimensions_mm}",
            )
        else:
            confidences["dimensions_mm"] = FieldConfidence(
                confidence=ConfidenceLevel.LOW,
                reason="Dimension derived or converted without direct verbatim numeric match.",
                evidence=None,
            )

    # 4. Stone Count
    if specs.stone_count is None:
        confidences["stone_count"] = FieldConfidence(
            confidence=ConfidenceLevel.LOW,
            reason="Stone count not specified in sales brief.",
            evidence=None,
        )
    else:
        sc = specs.stone_count
        if sc == 0 and any(w in brief_lower for w in ("plain", "no stone", "without stone", "no diamond", "band")):
            confidences["stone_count"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason="Explicit phrasing indicating plain/stone-free design verified.",
                evidence="plain/no stone",
            )
        elif sc == 1 and any(w in brief_lower for w in ("solitaire", "single stone", "one diamond", "1 diamond", "1 stone", "single diamond")):
            confidences["stone_count"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason="Explicit phrasing indicating single/solitaire stone verified.",
                evidence="solitaire/single stone",
            )
        elif str(sc) in brief_lower:
            confidences["stone_count"] = FieldConfidence(
                confidence=ConfidenceLevel.HIGH,
                reason=f"Explicit count '{sc}' verified in brief text.",
                evidence=str(sc),
            )
        else:
            confidences["stone_count"] = FieldConfidence(
                confidence=ConfidenceLevel.MEDIUM,
                reason=f"Stone count '{sc}' inferred from descriptive context.",
                evidence=None,
            )

    return confidences


def evaluate_review_flags(
    specs: ProductSpecification,
    retries_attempted: int,
    error_message: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Determine whether human review is required and enumerate reasons."""
    reasons: List[str] = []

    if error_message:
        reasons.append(f"Extraction error encountered: {error_message}")

    if retries_attempted > 0:
        reasons.append(f"Required {retries_attempted} retry attempt(s) to obtain valid output.")

    # Critical fields check
    if specs.product_type is None:
        reasons.append("Mandatory field 'product_type' is missing/unknown.")
    elif specs.field_confidences.get("product_type", FieldConfidence()).confidence == ConfidenceLevel.LOW:
        reasons.append("Product type confidence is low.")

    if specs.material is None:
        reasons.append("Mandatory field 'material' is missing/unknown.")
    elif specs.field_confidences.get("material", FieldConfidence()).confidence == ConfidenceLevel.LOW:
        reasons.append("Material confidence is low.")

    requires_review = len(reasons) > 0
    return requires_review, reasons


def extract_specifications(
    brief: str,
    provider: Optional[LLMProvider] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExtractionResult:
    """
    Extract structured jewelry specifications from a customer sales brief.

    Args:
        brief: Text of the customer sales brief.
        provider: An instance of LLMProvider. If None, instantiates OpenRouterProvider.
        max_retries: Maximum number of retry attempts for malformed/invalid output.

    Returns:
        ExtractionResult containing status, validated specifications, confidence,
        retry metrics, and human-review flags. Never raises uncaught provider exceptions.
    """
    if not brief or not brief.strip():
        empty_specs = ProductSpecification()
        empty_specs.field_confidences = calibrate_field_confidences("", empty_specs)
        return ExtractionResult(
            status=ExtractionStatus.FAILED,
            specifications=empty_specs,
            raw_response=None,
            retries_attempted=0,
            requires_human_review=True,
            review_reasons=["Empty or whitespace sales brief provided."],
            error_message="Empty sales brief.",
        )

    if provider is None:
        provider = OpenRouterProvider()

    current_prompt = f"Extract product specifications from this sales brief:\n\n\"\"\"\n{brief}\n\"\"\""
    last_raw_response: Optional[str] = None
    last_error: Optional[str] = None
    retries = 0

    while retries <= max_retries:
        try:
            raw_text = provider.generate(
                prompt=current_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
            )
            last_raw_response = raw_text

            # Parse JSON
            raw_data = extract_json_from_text(raw_text)

            # Validate with Pydantic
            specs = ProductSpecification.model_validate(raw_data)

            # Calibrate field confidences with textual evidence
            specs.field_confidences = calibrate_field_confidences(brief, specs)

            # Evaluate review flags
            requires_review, review_reasons = evaluate_review_flags(
                specs, retries, None
            )

            # Determine overall status
            if specs.product_type is not None and specs.material is not None:
                status = ExtractionStatus.SUCCESS
            else:
                status = ExtractionStatus.PARTIAL

            return ExtractionResult(
                status=status,
                specifications=specs,
                raw_response=last_raw_response,
                retries_attempted=retries,
                requires_human_review=requires_review,
                review_reasons=review_reasons,
                provider_info={"provider": type(provider).__name__},
                error_message=None,
            )

        except (json.JSONDecodeError, ValueError, ValidationError) as parse_err:
            last_error = f"{type(parse_err).__name__}: {str(parse_err)}"
            retries += 1
            if retries <= max_retries:
                # Prepare targeted correction prompt for retry
                current_prompt = (
                    f"Your previous response had a validation/formatting error: {last_error}\n\n"
                    f"Please output strictly valid JSON conforming to the schema for this brief:\n"
                    f"\"\"\"\n{brief}\n\"\"\""
                )
        except Exception as provider_err:
            # Network error, timeout, or external provider failure
            last_error = f"Provider Error: {str(provider_err)}"
            break

    # If retries exhausted or provider exception occurred, construct safe fallback
    fallback_specs = ProductSpecification()
    fallback_specs.field_confidences = calibrate_field_confidences(brief, fallback_specs)
    requires_review, review_reasons = evaluate_review_flags(
        fallback_specs, retries, last_error
    )

    return ExtractionResult(
        status=ExtractionStatus.FAILED,
        specifications=fallback_specs,
        raw_response=last_raw_response,
        retries_attempted=retries,
        requires_human_review=True,
        review_reasons=review_reasons,
        provider_info={"provider": type(provider).__name__},
        error_message=last_error,
    )
