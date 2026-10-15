"""
Unit tests for LLM-Based Structured Extraction Module.

Tests cover:
1. Valid structured output extraction and confidence calibration.
2. Missing optional fields handled explicitly with low confidence.
3. Malformed JSON handling with automatic retry success.
4. Invalid field values caught by Pydantic and recovered via retry.
5. Safe failure after retries are exhausted without crashing.
6. Provider exceptions and timeouts handled gracefully.
7. Unit tests run completely offline with mock providers (no API keys or network required).
8. Markdown-wrapped JSON response parsing.
9. Empty sales brief guard.
10. Integration with matcher specifications format.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import requests
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.extraction import (
    DEFAULT_MAX_RETRIES,
    LLMProvider,
    OpenRouterProvider,
    extract_json_from_text,
    extract_specifications,
)
from app.matcher import match_products
from app.schemas import (
    ConfidenceLevel,
    ExtractionResult,
    ExtractionStatus,
    MaterialEnum,
    ProductSpecification,
    ProductTypeEnum,
)


class MockLLMProvider:
    """Mock LLM provider for deterministic offline testing."""

    def __init__(self, responses: List[Union[str, Exception]]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: List[str] = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("MockLLMProvider exhausted all mocked responses.")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


# ============================================================================
# Test 1: Valid structured output
# ============================================================================

def test_valid_structured_output():
    """Extract valid structured specification with calibrated high confidence."""
    brief = "Customer requested an 18k gold ring with 1 solitaire diamond, inner diameter 17.5mm."
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 17.5, "stone_count": 1}'
    )
    mock_provider = MockLLMProvider([mock_json])

    result = extract_specifications(brief, provider=mock_provider)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.specifications.product_type == ProductTypeEnum.RING
    assert result.specifications.material == MaterialEnum.GOLD
    assert result.specifications.dimensions_mm == 17.5
    assert result.specifications.stone_count == 1
    assert result.retries_attempted == 0
    assert result.requires_human_review is False
    assert mock_provider.call_count == 1

    # Verify confidence calibrations
    confidences = result.specifications.field_confidences
    assert confidences["product_type"].confidence == ConfidenceLevel.HIGH
    assert confidences["material"].confidence == ConfidenceLevel.HIGH
    assert confidences["dimensions_mm"].confidence == ConfidenceLevel.HIGH
    assert confidences["stone_count"].confidence == ConfidenceLevel.HIGH


# ============================================================================
# Test 2: Missing optional fields
# ============================================================================

def test_missing_optional_fields():
    """Missing or unspecified attributes must be None and given low confidence."""
    brief = "Customer wants a sterling silver pendant. Size and stones not mentioned."
    mock_json = (
        '{"product_type": "pendant", "material": "silver", "dimensions_mm": null, "stone_count": null}'
    )
    mock_provider = MockLLMProvider([mock_json])

    result = extract_specifications(brief, provider=mock_provider)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.specifications.product_type == ProductTypeEnum.PENDANT
    assert result.specifications.material == MaterialEnum.SILVER
    assert result.specifications.dimensions_mm is None
    assert result.specifications.stone_count is None

    confidences = result.specifications.field_confidences
    assert confidences["dimensions_mm"].confidence == ConfidenceLevel.LOW
    assert confidences["stone_count"].confidence == ConfidenceLevel.LOW
    assert "not specified" in confidences["dimensions_mm"].reason.lower()


# ============================================================================
# Test 3: Markdown-wrapped JSON response
# ============================================================================

def test_markdown_wrapped_json():
    """Verify that models wrapping JSON in markdown fences are parsed seamlessly."""
    brief = "A plain silver hoop earring, 15mm drop."
    markdown_response = (
        "```json\n"
        "{\n"
        '  "product_type": "earrings",\n'
        '  "material": "silver",\n'
        '  "dimensions_mm": 15.0,\n'
        '  "stone_count": 0\n'
        "}\n"
        "```"
    )
    mock_provider = MockLLMProvider([markdown_response])

    result = extract_specifications(brief, provider=mock_provider)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.specifications.product_type == ProductTypeEnum.EARRINGS
    assert result.specifications.material == MaterialEnum.SILVER
    assert result.specifications.dimensions_mm == 15.0
    assert result.specifications.stone_count == 0
    assert result.retries_attempted == 0


# ============================================================================
# Test 4: Malformed JSON triggers retry and succeeds
# ============================================================================

def test_malformed_json_retry_success():
    """When model outputs malformed syntax initially, retry loop recovers."""
    brief = "Looking for a platinum necklace 450mm long without any stones."
    malformed_resp = "Here is the extraction: {product_type: 'necklace', material: broken..."
    valid_resp = (
        '{"product_type": "necklace", "material": "platinum", "dimensions_mm": 450.0, "stone_count": 0}'
    )
    mock_provider = MockLLMProvider([malformed_resp, valid_resp])

    result = extract_specifications(brief, provider=mock_provider, max_retries=2)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.specifications.product_type == ProductTypeEnum.NECKLACE
    assert result.specifications.material == MaterialEnum.PLATINUM
    assert result.retries_attempted == 1
    assert mock_provider.call_count == 2
    # Human review flagged because a retry was required
    assert result.requires_human_review is True
    assert any("retry" in r.lower() for r in result.review_reasons)


# ============================================================================
# Test 5: Invalid field values caught by Pydantic and recovered via retry
# ============================================================================

def test_invalid_field_values_triggers_retry():
    """Invalid enum values (e.g. wristwatch) or negative values are caught and corrected."""
    brief = "A 180mm gold bracelet with 6 accent stones."
    invalid_resp = (
        '{"product_type": "wristwatch", "material": "gold", "dimensions_mm": 180.0, "stone_count": 6}'
    )
    valid_resp = (
        '{"product_type": "bracelet", "material": "gold", "dimensions_mm": 180.0, "stone_count": 6}'
    )
    mock_provider = MockLLMProvider([invalid_resp, valid_resp])

    result = extract_specifications(brief, provider=mock_provider, max_retries=2)

    assert result.status == ExtractionStatus.SUCCESS
    assert result.specifications.product_type == ProductTypeEnum.BRACELET
    assert result.retries_attempted == 1
    assert mock_provider.call_count == 2


# ============================================================================
# Test 6: Safe failure after retries are exhausted
# ============================================================================

def test_safe_failure_after_retries_exhausted():
    """When model repeatedly returns garbage, system returns safe failure without crashing."""
    brief = "Custom jewelry request."
    garbage = "I am an AI and cannot process this."
    mock_provider = MockLLMProvider([garbage, garbage, garbage])

    result = extract_specifications(brief, provider=mock_provider, max_retries=2)

    assert result.status == ExtractionStatus.FAILED
    assert result.specifications.product_type is None
    assert result.specifications.material is None
    assert result.retries_attempted == 3  # initial + 2 retries
    assert result.requires_human_review is True
    assert result.error_message is not None
    assert mock_provider.call_count == 3


# ============================================================================
# Test 7: Provider exceptions and timeouts handled gracefully
# ============================================================================

def test_provider_timeout_handled_gracefully():
    """Network timeouts do not crash the application."""
    brief = "Gold ring brief."
    timeout_exc = requests.exceptions.Timeout("Request to OpenRouter timed out after 15s")
    mock_provider = MockLLMProvider([timeout_exc])

    result = extract_specifications(brief, provider=mock_provider)

    assert result.status == ExtractionStatus.FAILED
    assert result.requires_human_review is True
    assert "Provider Error" in result.error_message
    assert "timed out" in result.error_message
    assert result.specifications.product_type is None


def test_provider_http_error_handled_gracefully():
    """HTTP 500 or 401 errors from provider do not crash the application."""
    brief = "Gold ring brief."
    http_exc = requests.exceptions.HTTPError("500 Server Error: Internal Server Error")
    mock_provider = MockLLMProvider([http_exc])

    result = extract_specifications(brief, provider=mock_provider)

    assert result.status == ExtractionStatus.FAILED
    assert result.requires_human_review is True
    assert "500 Server Error" in result.error_message


# ============================================================================
# Test 8: No API key raises ValueError before network call
# ============================================================================

def test_openrouter_missing_api_key_raises_value_error(monkeypatch):
    """OpenRouterProvider fails fast with clear ValueError if api_key is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider(api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        provider.generate("test prompt")


# ============================================================================
# Test 9: Empty sales brief guard
# ============================================================================

def test_empty_sales_brief_guard():
    """Passing empty or whitespace brief immediately fails safely with 0 provider calls."""
    mock_provider = MockLLMProvider([])
    result = extract_specifications("   ", provider=mock_provider)

    assert result.status == ExtractionStatus.FAILED
    assert result.requires_human_review is True
    assert mock_provider.call_count == 0
    assert "Empty sales brief" in result.error_message


# ============================================================================
# Test 10: End-to-end integration: extracted specs feed directly into matcher
# ============================================================================

def test_extracted_specs_feed_into_matcher():
    """Ensure to_matcher_specifications() output seamlessly integrates with match_products()."""
    brief = "I would like a gold ring with 1 stone, 18.1mm diameter."
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 18.1, "stone_count": 1}'
    )
    mock_provider = MockLLMProvider([mock_json])

    extraction_res = extract_specifications(brief, provider=mock_provider)
    assert extraction_res.status == ExtractionStatus.SUCCESS

    # Convert to matcher dict and execute match
    matcher_specs = extraction_res.specifications.to_matcher_specifications()
    assert matcher_specs == {
        "product_type": "ring",
        "material": "gold",
        "dimensions_mm": 18.1,
        "stone_count": 1,
    }

    match_res = match_products(matcher_specs)
    assert match_res["is_reliable_match"] is True
    assert match_res["best_match"]["sku"] == "JWL-RNG-002"
    assert match_res["best_match"]["price_inr"] == 48000.0


# ============================================================================
# Test 11: Schema validation boundary unit tests
# ============================================================================

def test_schema_validations():
    """Test boundary conditions on ProductSpecification model."""
    # Unknown string keywords map to None
    spec = ProductSpecification.model_validate(
        {"product_type": "unknown", "material": "null", "dimensions_mm": "none", "stone_count": "unspecified"}
    )
    assert spec.product_type is None
    assert spec.material is None
    assert spec.dimensions_mm is None
    assert spec.stone_count is None

    # Negative dimension rejected
    with pytest.raises(ValidationError):
        ProductSpecification(dimensions_mm=-5.0)

    # Negative stone count rejected
    with pytest.raises(ValidationError):
        ProductSpecification(stone_count=-1)

    # Boolean stone count rejected
    with pytest.raises(ValidationError):
        ProductSpecification(stone_count=True)
