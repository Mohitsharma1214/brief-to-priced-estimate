"""
Unit tests for Deterministic Jewelry Inventory Matcher.

Tests cover:
1. An exact match ranks first.
2. An incompatible material is not selected.
3. An incompatible product type is not selected.
4. Missing attributes are handled correctly.
5. Identical inputs produce identical rankings.
6. Tied scores use SKU as a deterministic tie-breaker.
7. No compatible product produces a safe no-match result.
8. The returned price matches the inventory record.
9. Configurable acceptance thresholds.
10. Top-k parameter honoring.
11. Similarity function monotonicity and boundaries.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure application package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.matcher import (
    DEFAULT_ACCEPTANCE_THRESHOLD,
    DEFAULT_WEIGHTS,
    calculate_dimension_similarity,
    calculate_stone_similarity,
    match_products,
)

INVENTORY_PATH = PROJECT_ROOT / "inventory" / "products.json"


@pytest.fixture
def real_inventory() -> List[Dict[str, Any]]:
    """Load the 40-record production inventory."""
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def temp_inventory_file(tmp_path: Path):
    """Factory fixture to create temporary inventory files."""
    def _create_inventory(records: List[Dict[str, Any]]) -> Path:
        file_path = tmp_path / "custom_products.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        return file_path
    return _create_inventory


# ============================================================================
# Test 1: An exact match ranks first
# ============================================================================

def test_exact_match_ranks_first(real_inventory):
    """When query specifications exactly match an inventory record, it must rank first with score 1.0."""
    target_record = real_inventory[0]  # JWL-RNG-001 (gold ring, 17.3mm, 0 stones)
    specifications = {
        "product_type": target_record["product_type"],
        "material": target_record["material"],
        "dimensions_mm": target_record["dimensions_mm"],
        "stone_count": target_record["stone_count"],
    }

    result = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=3)

    assert result["is_reliable_match"] is True
    assert result["best_match"] is not None
    assert result["best_match"]["sku"] == target_record["sku"]
    assert pytest.approx(result["best_match"]["overall_score"], abs=1e-4) == 1.0
    assert result["matches"][0]["sku"] == target_record["sku"]
    assert result["best_match"]["price_inr"] == target_record["price_inr"]


# ============================================================================
# Test 2: An incompatible material is not selected
# ============================================================================

def test_incompatible_material_not_selected(temp_inventory_file):
    """Candidates with conflicting materials must be strictly excluded."""
    records = [
        {
            "sku": "JWL-SILVER-01",
            "product_type": "ring",
            "material": "silver",
            "dimensions_mm": 18.0,
            "stone_count": 1,
            "price_inr": 4500.0,
        },
        {
            "sku": "JWL-GOLD-01",
            "product_type": "ring",
            "material": "gold",
            "dimensions_mm": 18.0,
            "stone_count": 1,
            "price_inr": 48000.0,
        },
    ]
    inv_file = temp_inventory_file(records)

    # Request gold ring
    specifications = {
        "product_type": "ring",
        "material": "gold",
        "dimensions_mm": 18.0,
        "stone_count": 1,
    }

    result = match_products(specifications, inventory_path=inv_file)

    assert result["is_reliable_match"] is True
    assert len(result["matches"]) == 1
    assert result["best_match"]["sku"] == "JWL-GOLD-01"
    # Ensure silver item is never in matches
    matched_skus = [m["sku"] for m in result["matches"]]
    assert "JWL-SILVER-01" not in matched_skus


# ============================================================================
# Test 3: An incompatible product type is not selected
# ============================================================================

def test_incompatible_product_type_not_selected(real_inventory):
    """Candidates with conflicting product types must never be matched or compared."""
    specifications = {
        "product_type": "necklace",
        "material": "gold",
        "dimensions_mm": 450.0,
        "stone_count": 5,
    }

    result = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=10)

    assert result["is_reliable_match"] is True
    assert len(result["matches"]) > 0
    # Every returned match must strictly be a necklace
    for candidate in result["matches"]:
        assert candidate["product_type"] == "necklace"
        assert candidate["material"] == "gold"
        assert candidate["product_type"] not in ("ring", "pendant", "bracelet", "earrings")


# ============================================================================
# Test 4: Missing attributes are handled correctly
# ============================================================================

def test_missing_attributes_handled_correctly(real_inventory):
    """Unspecified attributes must be omitted from calculation and remaining weights renormalized."""
    # Query with stone_count missing
    specs_without_stones = {
        "product_type": "ring",
        "material": "gold",
        "dimensions_mm": 17.3,
    }

    result = match_products(specs_without_stones, inventory_path=INVENTORY_PATH)

    assert result["is_reliable_match"] is True
    best = result["best_match"]
    assert best is not None
    # stone_count should not have an evaluation score
    assert best["attribute_scores"]["stone_count"] is None
    assert best["attribute_scores"]["product_type"] == 1.0
    assert best["attribute_scores"]["material"] == 1.0
    assert best["attribute_scores"]["dimensions_mm"] is not None

    # Overall score should be normalized to 1.0 since remaining weights sum to 1.0
    # Base weights: pt=0.35, mat=0.30, dim=0.20 -> sum=0.85
    # JWL-RNG-001 has dim=17.3 (exact match)
    assert best["sku"] == "JWL-RNG-001"
    assert pytest.approx(best["overall_score"], abs=1e-4) == 1.0

    # Query with dimensions missing
    specs_without_dims = {
        "product_type": "ring",
        "material": "gold",
        "stone_count": 0,
    }
    res_no_dims = match_products(specs_without_dims, inventory_path=INVENTORY_PATH)
    assert res_no_dims["is_reliable_match"] is True
    assert res_no_dims["best_match"]["attribute_scores"]["dimensions_mm"] is None

    # Query with completely empty specifications
    empty_result = match_products({}, inventory_path=INVENTORY_PATH)
    assert empty_result["is_reliable_match"] is False
    assert empty_result["best_match"] is None
    assert len(empty_result["matches"]) == 0
    assert "No valid specifications" in empty_result["reason"]


# ============================================================================
# Test 5: Identical inputs produce identical rankings
# ============================================================================

def test_identical_inputs_produce_identical_rankings(real_inventory):
    """Rankings, scores, and candidate order must be strictly deterministic across repeated runs."""
    specifications = {
        "product_type": "bracelet",
        "material": "silver",
        "dimensions_mm": 180.0,
        "stone_count": 4,
    }

    initial_result = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=4)

    for _ in range(25):
        repeat_result = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=4)
        assert repeat_result["is_reliable_match"] == initial_result["is_reliable_match"]
        assert repeat_result["reason"] == initial_result["reason"]

        initial_skus = [m["sku"] for m in initial_result["matches"]]
        repeat_skus = [m["sku"] for m in repeat_result["matches"]]
        assert initial_skus == repeat_skus

        initial_scores = [m["overall_score"] for m in initial_result["matches"]]
        repeat_scores = [m["overall_score"] for m in repeat_result["matches"]]
        assert initial_scores == repeat_scores


# ============================================================================
# Test 6: Tied scores use SKU as a deterministic tie-breaker
# ============================================================================

def test_tied_scores_use_sku_tie_breaker(temp_inventory_file):
    """When candidates achieve identical scores, SKU must strictly resolve the tie in alphabetical order."""
    records = [
        {
            "sku": "JWL-ZZZ-999",
            "product_type": "pendant",
            "material": "platinum",
            "dimensions_mm": 20.0,
            "stone_count": 1,
            "price_inr": 50000.0,
        },
        {
            "sku": "JWL-AAA-111",
            "product_type": "pendant",
            "material": "platinum",
            "dimensions_mm": 20.0,
            "stone_count": 1,
            "price_inr": 50000.0,
        },
        {
            "sku": "JWL-MMM-555",
            "product_type": "pendant",
            "material": "platinum",
            "dimensions_mm": 20.0,
            "stone_count": 1,
            "price_inr": 50000.0,
        },
    ]

    inv_file = temp_inventory_file(records)
    specifications = {
        "product_type": "pendant",
        "material": "platinum",
        "dimensions_mm": 20.0,
        "stone_count": 1,
    }

    result = match_products(specifications, inventory_path=inv_file, top_k=3)

    assert result["is_reliable_match"] is True
    # All 3 have score 1.0, so tie breaker must sort: AAA, MMM, ZZZ
    ordered_skus = [m["sku"] for m in result["matches"]]
    assert ordered_skus == ["JWL-AAA-111", "JWL-MMM-555", "JWL-ZZZ-999"]


# ============================================================================
# Test 7: No compatible product produces a safe no-match result
# ============================================================================

def test_no_compatible_product_produces_safe_no_match(real_inventory):
    """When no compatible candidate exists or score fails threshold, return safe no-match."""
    # Case A: Nonexistent product type
    specs_alien = {
        "product_type": "tiara",
        "material": "gold",
        "dimensions_mm": 100.0,
        "stone_count": 50,
    }
    res_alien = match_products(specs_alien, inventory_path=INVENTORY_PATH)
    assert res_alien["is_reliable_match"] is False
    assert res_alien["best_match"] is None
    assert len(res_alien["matches"]) == 0
    assert "No compatible inventory records" in res_alien["reason"]

    # Case B: Compatible category, but score fails strict acceptance threshold
    specs_wild = {
        "product_type": "ring",
        "material": "gold",
        "dimensions_mm": 100.0,  # Ridiculous ring size
        "stone_count": 500,       # Wild stone count
    }
    res_wild = match_products(
        specs_wild, inventory_path=INVENTORY_PATH, acceptance_threshold=0.85
    )
    assert res_wild["is_reliable_match"] is False
    assert res_wild["best_match"] is None
    assert "below acceptance threshold" in res_wild["reason"]
    # Candidates are still retained for inspection
    assert len(res_wild["matches"]) > 0


# ============================================================================
# Test 8: The returned price matches the inventory record
# ============================================================================

def test_returned_price_matches_inventory_record(real_inventory):
    """The price in INR must strictly originate from the inventory record, never altered or hallucinated."""
    sku_price_lookup = {r["sku"]: r["price_inr"] for r in real_inventory}

    test_queries = [
        {"product_type": "ring", "material": "silver", "dimensions_mm": 18.5, "stone_count": 1},
        {"product_type": "necklace", "material": "platinum", "dimensions_mm": 450.0, "stone_count": 0},
        {"product_type": "earrings", "material": "gold", "dimensions_mm": 15.0, "stone_count": 2},
    ]

    for q in test_queries:
        res = match_products(q, inventory_path=INVENTORY_PATH, top_k=3)
        assert len(res["matches"]) > 0
        for match in res["matches"]:
            sku = match["sku"]
            expected_price = sku_price_lookup[sku]
            assert match["price_inr"] == expected_price
            assert match["price"] == expected_price


# ============================================================================
# Test 9: Acceptance threshold configurability
# ============================================================================

def test_acceptance_threshold_configurability(real_inventory):
    """Threshold can be tuned to accept or reject borderline candidate matches."""
    # A slightly off ring: requested 19.5mm and 3 stones
    specifications = {
        "product_type": "ring",
        "material": "silver",
        "dimensions_mm": 19.5,
        "stone_count": 3,
    }

    # Strict threshold 0.99 rejects it
    strict_res = match_products(
        specifications, inventory_path=INVENTORY_PATH, acceptance_threshold=0.99
    )
    assert strict_res["is_reliable_match"] is False
    assert strict_res["best_match"] is None

    # Lenient threshold 0.60 accepts it
    lenient_res = match_products(
        specifications, inventory_path=INVENTORY_PATH, acceptance_threshold=0.60
    )
    assert lenient_res["is_reliable_match"] is True
    assert lenient_res["best_match"] is not None


# ============================================================================
# Test 10: Top-k parameter is honored
# ============================================================================

def test_top_k_parameter_honored(real_inventory):
    """Matcher returns at most top_k items."""
    specifications = {"product_type": "ring", "material": "gold"}

    res_1 = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=1)
    assert len(res_1["matches"]) == 1

    res_3 = match_products(specifications, inventory_path=INVENTORY_PATH, top_k=3)
    assert len(res_3["matches"]) == 3


# ============================================================================
# Test 11: Similarity functions monotonicity and boundaries
# ============================================================================

def test_similarity_functions_monotonic_and_bounded():
    """Verify numeric similarity functions are bounded to (0, 1] and strictly monotonic."""
    # Dimensions
    assert pytest.approx(calculate_dimension_similarity(18.0, 18.0), abs=1e-6) == 1.0
    dim_close = calculate_dimension_similarity(18.1, 18.0)
    dim_mid = calculate_dimension_similarity(19.0, 18.0)
    dim_far = calculate_dimension_similarity(25.0, 18.0)

    assert 1.0 > dim_close > dim_mid > dim_far > 0.0

    # Stone count
    assert pytest.approx(calculate_stone_similarity(5, 5), abs=1e-6) == 1.0
    stone_close = calculate_stone_similarity(6, 5)
    stone_mid = calculate_stone_similarity(10, 5)
    stone_far = calculate_stone_similarity(30, 5)

    assert 1.0 > stone_close > stone_mid > stone_far > 0.0


# ============================================================================
# Test 12: Invalid top_k raises ValueError
# ============================================================================

@pytest.mark.parametrize("invalid_top_k", [0, -1, -5, "3", 2.5, False, None])
def test_invalid_top_k_raises_error(invalid_top_k):
    """Passing invalid top_k values must raise ValueError."""
    specs = {"product_type": "ring", "material": "gold"}
    with pytest.raises(ValueError):
        match_products(specs, inventory_path=INVENTORY_PATH, top_k=invalid_top_k)


# ============================================================================
# Test 13: Invalid acceptance_threshold raises ValueError
# ============================================================================

@pytest.mark.parametrize(
    "invalid_threshold",
    [-0.01, -1.0, 1.01, 2.0, float("nan"), float("inf"), float("-inf"), "0.7", True, False, None],
)
def test_invalid_acceptance_threshold_raises_error(invalid_threshold):
    """Passing invalid acceptance thresholds must raise ValueError."""
    specs = {"product_type": "ring", "material": "gold"}
    with pytest.raises(ValueError):
        match_products(
            specs, inventory_path=INVENTORY_PATH, acceptance_threshold=invalid_threshold
        )


# ============================================================================
# Test 14: Invalid custom_weights raises ValueError
# ============================================================================

@pytest.mark.parametrize(
    "invalid_weights",
    [
        {"dimensions_mm": -0.5},
        {"stone_count": float("nan")},
        {"material": float("inf")},
        {"product_type": float("-inf")},
        {"product_type": 0, "material": 0, "dimensions_mm": 0, "stone_count": 0},
        {"product_type": False},
        "not_a_dict",
        [0.35, 0.30, 0.20, 0.15],
    ],
)
def test_invalid_custom_weights_raises_error(invalid_weights):
    """Negative, non-finite, zero-sum, or non-dict custom weights must raise ValueError."""
    specs = {"product_type": "ring", "material": "gold"}
    with pytest.raises(ValueError):
        match_products(specs, inventory_path=INVENTORY_PATH, custom_weights=invalid_weights)


# ============================================================================
# Test 15: Negative and non-finite dimensions raise ValueError
# ============================================================================

@pytest.mark.parametrize(
    "invalid_dim",
    [-10.0, -1.0, 0.0, float("nan"), float("inf"), float("-inf"), False, True, "18.0"],
)
def test_negative_and_non_finite_dimensions_raise_error(invalid_dim):
    """Non-positive or non-finite dimensions must raise ValueError."""
    specs = {"product_type": "ring", "material": "gold", "dimensions_mm": invalid_dim}
    with pytest.raises(ValueError):
        match_products(specs, inventory_path=INVENTORY_PATH)


# ============================================================================
# Test 16: Negative, non-integer, and non-finite stone counts raise ValueError
# ============================================================================

@pytest.mark.parametrize(
    "invalid_stones",
    [-5, -1, 2.5, float("nan"), float("inf"), float("-inf"), True, False, "1"],
)
def test_negative_and_non_finite_stone_counts_raise_error(invalid_stones):
    """Negative, floating point, boolean, or non-finite stone counts must raise ValueError."""
    specs = {"product_type": "ring", "material": "gold", "stone_count": invalid_stones}
    with pytest.raises(ValueError):
        match_products(specs, inventory_path=INVENTORY_PATH)


# ============================================================================
# Test 17: Missing inventory attributes are handled and distinguished from known mismatch
# ============================================================================

def test_missing_inventory_attributes_distinguished_from_known_mismatch(temp_inventory_file):
    """
    Missing attributes in inventory records must be penalized (score 0.0 for that attribute)
    and explicitly distinguished in explanations as 'unknown' rather than '0.0mm' or a known mismatch.
    """
    records = [
        {
            "sku": "JWL-UNKNOWN-DIM",
            "product_type": "ring",
            "material": "gold",
            "dimensions_mm": None,
            "stone_count": 1,
            "price_inr": 35000.0,
        },
        {
            "sku": "JWL-UNKNOWN-STONES",
            "product_type": "ring",
            "material": "gold",
            "dimensions_mm": 18.0,
            "stone_count": None,
            "price_inr": 36000.0,
        },
        {
            "sku": "JWL-KNOWN-MISMATCH",
            "product_type": "ring",
            "material": "gold",
            "dimensions_mm": 99.0,
            "stone_count": 50,
            "price_inr": 37000.0,
        },
    ]
    inv_file = temp_inventory_file(records)

    specs = {
        "product_type": "ring",
        "material": "gold",
        "dimensions_mm": 18.0,
        "stone_count": 1,
    }

    result = match_products(specs, inventory_path=inv_file, top_k=3, acceptance_threshold=0.50)

    matches_by_sku = {m["sku"]: m for m in result["matches"]}

    # JWL-UNKNOWN-DIM: dimensions missing in inventory
    dim_unknown_match = matches_by_sku["JWL-UNKNOWN-DIM"]
    assert dim_unknown_match["attribute_scores"]["dimensions_mm"] == 0.0
    assert "inventory dimension unknown" in dim_unknown_match["explanation"]
    assert "0.0mm" not in dim_unknown_match["explanation"]

    # JWL-UNKNOWN-STONES: stone count missing in inventory
    stones_unknown_match = matches_by_sku["JWL-UNKNOWN-STONES"]
    assert stones_unknown_match["attribute_scores"]["stone_count"] == 0.0
    assert "inventory stone_count unknown" in stones_unknown_match["explanation"]

    # Test inventory record missing mandatory product_type or material is rejected
    invalid_rec_file = temp_inventory_file([
        {
            "sku": "JWL-NO-TYPE",
            "product_type": None,
            "material": "gold",
            "dimensions_mm": 18.0,
            "stone_count": 1,
            "price_inr": 30000.0,
        },
        {
            "sku": "JWL-NO-MAT",
            "product_type": "ring",
            "material": None,
            "dimensions_mm": 18.0,
            "stone_count": 1,
            "price_inr": 30000.0,
        },
    ])
    res_missing_cat = match_products(specs, inventory_path=invalid_rec_file)
    assert res_missing_cat["is_reliable_match"] is False
    assert len(res_missing_cat["matches"]) == 0


# ============================================================================
# Test 18: Material and product-type conflicts are strictly rejected
# ============================================================================

def test_material_and_product_type_conflicts_strictly_rejected(temp_inventory_file):
    """Conflicting materials or product types must be completely excluded from candidates."""
    records = [
        {"sku": "JWL-GOLD-RING", "product_type": "ring", "material": "gold", "dimensions_mm": 18.0, "stone_count": 1, "price_inr": 40000.0},
        {"sku": "JWL-SILVER-RING", "product_type": "ring", "material": "silver", "dimensions_mm": 18.0, "stone_count": 1, "price_inr": 4000.0},
        {"sku": "JWL-PLAT-RING", "product_type": "ring", "material": "platinum", "dimensions_mm": 18.0, "stone_count": 1, "price_inr": 70000.0},
        {"sku": "JWL-GOLD-NECK", "product_type": "necklace", "material": "gold", "dimensions_mm": 450.0, "stone_count": 1, "price_inr": 80000.0},
        {"sku": "JWL-GOLD-PEND", "product_type": "pendant", "material": "gold", "dimensions_mm": 20.0, "stone_count": 1, "price_inr": 30000.0},
    ]
    inv_file = temp_inventory_file(records)

    specs = {"product_type": "ring", "material": "gold", "dimensions_mm": 18.0, "stone_count": 1}
    result = match_products(specs, inventory_path=inv_file, top_k=10)

    assert result["is_reliable_match"] is True
    assert len(result["matches"]) == 1
    assert result["matches"][0]["sku"] == "JWL-GOLD-RING"

    returned_skus = {m["sku"] for m in result["matches"]}
    assert "JWL-SILVER-RING" not in returned_skus
    assert "JWL-PLAT-RING" not in returned_skus
    assert "JWL-GOLD-NECK" not in returned_skus
    assert "JWL-GOLD-PEND" not in returned_skus
