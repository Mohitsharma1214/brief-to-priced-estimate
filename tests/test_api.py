"""
Integration and API unit tests for FastAPI /analyze endpoint.

Tests cover:
1. Valid brief without an image.
2. Valid brief with an image (consistent stone count).
3. Empty brief and malformed requests (HTTP 400 and 422).
4. Invalid image upload (safe degradation without crashing).
5. Extraction failure (safe fallback response with human-review flag).
6. Vision failure with graceful fallback (text analysis preserved).
7. No acceptable inventory match (safe rejection and price=None).
8. Stone-count disagreement producing a review flag.
9. Response schema validation and inventory price provenance.
10. Upload size limits enforcement (HTTP 413).
11. System health check endpoint (/health).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.schemas import AnalyzeResponse, ConfidenceLevel, ProductTypeEnum
from tests.test_extraction import MockLLMProvider
from tests.test_vision import create_synthetic_jewelry_image

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_app_state():
    """Ensure app.state is clean before and after each test."""
    app.state.llm_provider = None
    app.state.inventory_path = None
    yield
    app.state.llm_provider = None
    app.state.inventory_path = None


# ============================================================================
# Test 1: Health check endpoint
# ============================================================================

def test_health_check_endpoint():
    """Verify /health reports healthy status and verifies the 40 inventory records."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["inventory_records"] == 40


# ============================================================================
# Test 2: Valid brief without an image
# ============================================================================

def test_valid_brief_without_image():
    """Valid text brief processes through extraction and matcher with no image."""
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 17.3, "stone_count": 0}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    response = client.post(
        "/analyze",
        data={"brief": "Looking for a simple 18k gold band ring, 17.3mm inner diameter, plain."},
    )

    assert response.status_code == 200
    data = response.json()

    # Schema validation
    validated = AnalyzeResponse.model_validate(data)
    assert validated.success is True
    assert validated.vision_estimate is None
    assert validated.extracted_spec.product_type == ProductTypeEnum.RING
    assert validated.best_match is not None
    assert validated.best_match["sku"] == "JWL-RNG-001"
    assert validated.estimated_price_inr == 24500.0
    assert validated.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
    assert validated.requires_human_review is False


# ============================================================================
# Test 3: Valid brief with an image
# ============================================================================

def test_valid_brief_with_image():
    """Valid brief and matching image estimate produce coordinated high-confidence output."""
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 18.1, "stone_count": 1}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    # Synthetic image with exactly 1 solitaire stone
    image_bytes = create_synthetic_jewelry_image([(150, 150, 18)])

    response = client.post(
        "/analyze",
        data={"brief": "18k gold ring with 1 solitaire diamond, 18.1mm diameter."},
        files={"image": ("ring.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.success is True
    assert validated.vision_estimate is not None
    assert validated.vision_estimate.estimated_visible_stones == 1
    assert validated.best_match is not None
    assert validated.best_match["sku"] == "JWL-RNG-002"
    assert validated.estimated_price_inr == 48000.0
    assert not any("mismatch" in f.lower() for f in validated.review_flags)


# ============================================================================
# Test 4: Empty brief and malformed requests
# ============================================================================

def test_empty_brief_returns_400():
    """Empty or whitespace-only sales brief returns HTTP 400."""
    res_empty = client.post("/analyze", data={"brief": ""})
    assert res_empty.status_code == 400
    assert "non-empty string" in res_empty.json()["detail"]

    res_spaces = client.post("/analyze", data={"brief": "    "})
    assert res_spaces.status_code == 400
    assert "non-empty string" in res_spaces.json()["detail"]


def test_missing_brief_parameter_returns_422():
    """Omitting the required brief parameter returns HTTP 422 Unprocessable Entity."""
    res = client.post("/analyze", data={})
    assert res.status_code == 422


# ============================================================================
# Test 5: Invalid image upload
# ============================================================================

def test_invalid_image_upload_degrades_gracefully():
    """Corrupted or non-image uploaded bytes must not crash the text analysis."""
    mock_json = (
        '{"product_type": "necklace", "material": "silver", "dimensions_mm": 400.0, "stone_count": 0}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    corrupted_bytes = b"CORRUPTED_NON_IMAGE_HEADER_XYZ"

    response = client.post(
        "/analyze",
        data={"brief": "Silver chain necklace 400mm long."},
        files={"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")},
    )

    # Must return 200 with degraded vision status and review flags
    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.success is True
    assert validated.vision_estimate is None
    assert validated.requires_human_review is True
    assert any("image" in e.lower() or "vision" in e.lower() for e in validated.errors)
    assert any("image" in f.lower() or "vision" in f.lower() for f in validated.review_flags)
    # Text analysis still succeeded!
    assert validated.best_match is not None
    assert validated.best_match["sku"] == "JWL-NCK-004"


# ============================================================================
# Test 6: Extraction failure
# ============================================================================

def test_extraction_failure_safe_handling():
    """When LLM provider fails or throws exception, API returns safe failure without crashing."""
    app.state.llm_provider = MockLLMProvider([RuntimeError("OpenRouter API 503 Service Unavailable")])

    response = client.post(
        "/analyze",
        data={"brief": "Gold bracelet request."},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.success is False
    assert validated.requires_human_review is True
    assert validated.estimated_price_inr is None
    assert validated.best_match is None
    assert len(validated.errors) > 0
    assert any("extraction" in e.lower() for e in validated.errors)


# ============================================================================
# Test 7: Vision failure with graceful fallback
# ============================================================================

def test_vision_failure_with_graceful_fallback(monkeypatch):
    """If computer vision estimation raises unexpected error, text pipeline still completes."""
    mock_json = (
        '{"product_type": "earrings", "material": "gold", "dimensions_mm": 12.0, "stone_count": 0}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    # Mock estimate_visible_stones to simulate unexpected processing error
    def mock_fail(image_bytes):
        raise ValueError("Simulated CV processing pipeline fault")

    monkeypatch.setattr("app.main.estimate_visible_stones", mock_fail)

    image_bytes = create_synthetic_jewelry_image([(100, 100, 10)])
    response = client.post(
        "/analyze",
        data={"brief": "Plain gold stud earrings 12mm."},
        files={"image": ("stud.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.success is True
    assert validated.vision_estimate is None
    assert validated.requires_human_review is True
    assert any("computer vision" in e.lower() for e in validated.errors)
    # Matching still successfully identified the item
    assert validated.best_match is not None
    assert validated.best_match["sku"] == "JWL-EAR-001"
    assert validated.estimated_price_inr == 18500.0


# ============================================================================
# Test 8: No acceptable inventory match
# ============================================================================

def test_no_acceptable_inventory_match():
    """When extracted attributes find no compatible inventory records, return safe no-price."""
    # Tiara is not in inventory
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 999.0, "stone_count": 500}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    response = client.post(
        "/analyze",
        data={"brief": "Enormous custom piece."},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.best_match is None
    assert validated.estimated_price_inr is None
    assert validated.requires_human_review is True
    assert any("inventory matching" in f.lower() for f in validated.review_flags)


# ============================================================================
# Test 9: Stone-count disagreement producing a review flag
# ============================================================================

def test_stone_count_disagreement_produces_review_flag():
    """When sales brief specifies X stones but vision detects Y visible stones, flag discrepancy."""
    # Brief claims 6 stones
    mock_json = (
        '{"product_type": "ring", "material": "gold", "dimensions_mm": 18.0, "stone_count": 6}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    # Reference image only displays 1 visible stone
    image_bytes = create_synthetic_jewelry_image([(150, 150, 18)])

    response = client.post(
        "/analyze",
        data={"brief": "Gold ring with 6 stones."},
        files={"image": ("ring.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.requires_human_review is True
    mismatch_flags = [f for f in validated.review_flags if "stone count mismatch" in f.lower()]
    assert len(mismatch_flags) > 0
    assert "specified 6" in mismatch_flags[0]
    assert "detected 1" in mismatch_flags[0]


# ============================================================================
# Test 10: Response schema and price provenance
# ============================================================================

def test_response_schema_and_price_provenance():
    """Verify that estimated_price_inr originates directly from products.json inventory records."""
    mock_json = (
        '{"product_type": "bracelet", "material": "platinum", "dimensions_mm": 180.0, "stone_count": 0}'
    )
    app.state.llm_provider = MockLLMProvider([mock_json])

    response = client.post(
        "/analyze",
        data={"brief": "Platinum bracelet 180mm plain."},
    )

    assert response.status_code == 200
    data = response.json()
    validated = AnalyzeResponse.model_validate(data)

    assert validated.best_match is not None
    assert validated.best_match["sku"] == "JWL-BRC-007"
    # Ground truth price in products.json for JWL-BRC-007 is 78000.0
    assert validated.estimated_price_inr == 78000.0
    assert validated.best_match["price_inr"] == 78000.0


# ============================================================================
# Test 11: Upload size limits
# ============================================================================

def test_upload_size_limit_enforced():
    """Payloads exceeding the 20 MB limit return HTTP 413 Request Entity Too Large."""
    huge_bytes = b"0" * (21 * 1024 * 1024)

    response = client.post(
        "/analyze",
        data={"brief": "Valid brief."},
        files={"image": ("huge.png", huge_bytes, "image/png")},
    )

    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]
