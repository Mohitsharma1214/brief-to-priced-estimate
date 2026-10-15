"""
Deterministic Jewelry Inventory Matcher for Brief to Priced Estimate.

This module matches extracted customer specifications (product_type, material,
dimensions_mm, stone_count) against an inventory dataset of jewelry records.

Key Architecture Principles:
1. Strict Mandatory Compatibility:
   - Product type and material act as hard constraints. Conflicting candidates
     are excluded to prevent cross-category contamination (e.g., matching a ring
     against a necklace dimension).
2. Documented Attribute Similarity Functions:
   - Dimensions & stone counts use normalized exponential decay kernels that
     yield 1.0 on exact match and decrease monotonically as differences grow.
3. Explicit Missing Attribute Handling:
   - Unspecified attributes in query specifications are omitted from the scoring
     denominator and remaining weights are renormalized. Missing data is never
     silently treated as a perfect match.
4. Determinism & Traceability:
   - Strict tie-breaking by SKU ensures repeatable, deterministic ranking.
   - Prices are sourced directly from the inventory record (no LLM hallucinations).
5. Configurable Acceptance Threshold:
   - Configurable minimum score required to declare a reliable match.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Base attribute weights
DEFAULT_WEIGHTS: Dict[str, float] = {
    "product_type": 0.35,
    "material": 0.30,
    "dimensions_mm": 0.20,
    "stone_count": 0.15,
}

# Default acceptance threshold for declaring a reliable match
DEFAULT_ACCEPTANCE_THRESHOLD: float = 0.70

# Default number of top matches to return
DEFAULT_TOP_K: int = 3

# Note on scoring semantics: The overall match score is an engineering similarity
# heuristic (bounded in [0.0, 1.0]), NOT a statistical probability or Bayesian likelihood.
# It indicates empirical closeness under the defined weighted distance kernels.


def get_default_inventory_path() -> Path:
    """Resolve the default path to products.json across project layouts."""
    env_path = os.getenv("INVENTORY_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()

    candidates = [
        Path(__file__).resolve().parent.parent / "inventory" / "products.json",
        Path.cwd() / "brief-to-priced-estimate" / "inventory" / "products.json",
        Path.cwd() / "inventory" / "products.json",
        Path.cwd() / "products.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def load_inventory(
    inventory_source: Optional[Union[str, Path, Sequence[Dict[str, Any]]]] = None
) -> List[Dict[str, Any]]:
    """
    Load inventory records from a file path, Path object, or in-memory list.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If JSON data is malformed or not a list of dictionaries.
    """
    if inventory_source is None:
        inventory_path = get_default_inventory_path()
        if not inventory_path.is_file():
            raise FileNotFoundError(
                f"Default inventory file not found at: {inventory_path}"
            )
        with open(inventory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(inventory_source, (str, Path)):
        path = Path(inventory_source)
        if not path.is_file():
            raise FileNotFoundError(f"Inventory file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(inventory_source, list):
        data = inventory_source
    else:
        raise TypeError(
            f"Unsupported inventory_source type: {type(inventory_source).__name__}"
        )

    if not isinstance(data, list):
        raise ValueError(
            f"Inventory must contain a list of records, got {type(data).__name__}"
        )

    return data


def calculate_dimension_similarity(
    candidate_dim: float, target_dim: float
) -> float:
    """
    Compute dimension similarity in range (0.0, 1.0].

    Formula: exp(-|candidate_dim - target_dim| / sigma_dim)
    where sigma_dim = max(0.15 * target_dim, 1.5 mm).

    Properties:
    - Exactly 1.0 when candidate_dim == target_dim.
    - Decreases monotonically as |candidate_dim - target_dim| increases.
    - Scales proportionally with target dimension size.
    """
    diff = abs(float(candidate_dim) - float(target_dim))
    sigma = max(0.15 * float(target_dim), 1.5)
    return math.exp(-diff / sigma)


def calculate_stone_similarity(
    candidate_stones: int, target_stones: int
) -> float:
    """
    Compute stone count similarity in range (0.0, 1.0].

    Formula: exp(-|candidate_stones - target_stones| / sigma_stones)
    where sigma_stones = max(0.25 * target_stones, 3.0).

    Properties:
    - Exactly 1.0 when candidate_stones == target_stones.
    - Decreases monotonically as count discrepancy increases.
    - Tolerates small absolute differences for low counts, and scales for high counts.
    """
    diff = abs(int(candidate_stones) - int(target_stones))
    sigma = max(0.25 * float(target_stones), 3.0)
    return math.exp(-diff / sigma)


def check_mandatory_compatibility(
    specifications: Dict[str, Any], record: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Verify hard compatibility constraints.

    Mandatory Rules:
    1. If product_type is requested, record['product_type'] must match exactly (case-insensitive).
       Comparing dimensions across distinct product types is invalid.
    2. If material is requested, record['material'] must match exactly (case-insensitive).
       Material mismatch is a critical rejection criterion.

    Returns:
        (is_compatible, failure_reason)
    """
    req_type = specifications.get("product_type")
    if req_type is not None and str(req_type).strip():
        req_type_clean = str(req_type).strip().lower()
        rec_type = record.get("product_type")
        if rec_type is None or not str(rec_type).strip():
            return (
                False,
                f"Incompatible product_type: requested '{req_type_clean}', record has unknown/missing product_type",
            )
        rec_type_clean = str(rec_type).strip().lower()
        if req_type_clean != rec_type_clean:
            return (
                False,
                f"Incompatible product_type: requested '{req_type_clean}', record has '{rec_type_clean}'",
            )

    req_mat = specifications.get("material")
    if req_mat is not None and str(req_mat).strip():
        req_mat_clean = str(req_mat).strip().lower()
        rec_mat = record.get("material")
        if rec_mat is None or not str(rec_mat).strip():
            return (
                False,
                f"Incompatible material: requested '{req_mat_clean}', record has unknown/missing material",
            )
        rec_mat_clean = str(rec_mat).strip().lower()
        if req_mat_clean != rec_mat_clean:
            return (
                False,
                f"Incompatible material: requested '{req_mat_clean}', record has '{rec_mat_clean}'",
            )

    return True, None


def score_candidate(
    specifications: Dict[str, Any],
    record: Dict[str, Any],
    base_weights: Dict[str, float],
) -> Tuple[float, Dict[str, Optional[float]], str]:
    """
    Score a compatible candidate against query specifications.

    Missing Attribute Handling:
    - If an attribute is missing/None in specifications, it is excluded from
      the calculation, and the weights of remaining present attributes are
      renormalized to sum to 1.0.
    - Missing attributes are assigned None in attribute_scores rather than 1.0.

    Returns:
        (overall_score, attribute_scores, explanation)
    """
    # Identify which attributes were requested in specifications
    active_attrs: List[str] = []
    for attr in ("product_type", "material", "dimensions_mm", "stone_count"):
        val = specifications.get(attr)
        if val is not None and (not isinstance(val, str) or val.strip()):
            active_attrs.append(attr)

    if not active_attrs:
        return 0.0, {k: None for k in base_weights}, "No active attributes specified in query."

    # Renormalize weights among active attributes
    active_weight_sum = sum(base_weights.get(attr, 0.0) for attr in active_attrs)
    if active_weight_sum <= 0:
        return 0.0, {k: None for k in base_weights}, "Zero active weight sum."

    renormalized_weights = {
        attr: base_weights.get(attr, 0.0) / active_weight_sum for attr in active_attrs
    }

    attribute_scores: Dict[str, Optional[float]] = {
        "product_type": None,
        "material": None,
        "dimensions_mm": None,
        "stone_count": None,
    }
    explanation_parts: List[str] = []

    # Score product_type
    if "product_type" in active_attrs:
        req_type = str(specifications["product_type"]).strip().lower()
        rec_type = str(record.get("product_type", "")).strip().lower()
        score = 1.0 if req_type == rec_type else 0.0
        attribute_scores["product_type"] = round(score, 4)
        explanation_parts.append(f"product_type: {score:.2f} (requested '{req_type}')")

    # Score material
    if "material" in active_attrs:
        req_mat = str(specifications["material"]).strip().lower()
        rec_mat = str(record.get("material", "")).strip().lower()
        score = 1.0 if req_mat == rec_mat else 0.0
        attribute_scores["material"] = round(score, 4)
        explanation_parts.append(f"material: {score:.2f} (requested '{req_mat}')")

    # Score dimensions_mm
    if "dimensions_mm" in active_attrs:
        target_dim = float(specifications["dimensions_mm"])
        rec_dim = record.get("dimensions_mm")
        if rec_dim is None:
            score = 0.0
            explanation_parts.append(
                f"dimensions: 0.00 (target {target_dim:.1f}mm, inventory dimension unknown)"
            )
        else:
            score = calculate_dimension_similarity(float(rec_dim), target_dim)
            explanation_parts.append(
                f"dimensions: {score:.2f} (target {target_dim:.1f}mm, record {float(rec_dim):.1f}mm)"
            )
        attribute_scores["dimensions_mm"] = round(score, 4)

    # Score stone_count
    if "stone_count" in active_attrs:
        target_stones = int(specifications["stone_count"])
        rec_stones = record.get("stone_count")
        if rec_stones is None:
            score = 0.0
            explanation_parts.append(
                f"stone_count: 0.00 (target {target_stones}, inventory stone_count unknown)"
            )
        else:
            score = calculate_stone_similarity(int(rec_stones), target_stones)
            explanation_parts.append(
                f"stone_count: {score:.2f} (target {target_stones}, record {int(rec_stones)})"
            )
        attribute_scores["stone_count"] = round(score, 4)

    # Compute overall score
    total_score = sum(
        renormalized_weights[attr] * (attribute_scores[attr] or 0.0)
        for attr in active_attrs
    )
    overall_score = round(max(0.0, min(1.0, total_score)), 4)

    # Build human-readable explanation
    explanation = f"Score: {overall_score:.4f} [Weights: " + ", ".join(
        f"{attr}={renormalized_weights[attr]:.2f}" for attr in active_attrs
    ) + "] | " + "; ".join(explanation_parts)

    return overall_score, attribute_scores, explanation


def match_products(
    specifications: Dict[str, Any],
    inventory_path: Optional[Union[str, Path, Sequence[Dict[str, Any]]]] = None,
    top_k: int = DEFAULT_TOP_K,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Deterministically match extracted specifications against the inventory.

    Args:
        specifications: Dictionary of query attributes (e.g., product_type,
                        material, dimensions_mm, stone_count).
        inventory_path: Path to products.json, Path object, or pre-loaded records list.
        top_k: Number of top candidate matches to return.
        acceptance_threshold: Minimum overall score (0.0 - 1.0) required to consider
                              the top match reliable.
        custom_weights: Optional dictionary overriding attribute weights.

    Returns:
        A dictionary containing:
        - is_reliable_match: bool indicating if a compatible candidate met the threshold.
        - best_match: The top candidate dictionary if reliable, else None.
        - matches: Top-k ranked compatible candidates with attribute scores and price.
        - acceptance_threshold: The configured acceptance threshold.
        - total_inventory_evaluated: Total records loaded and checked.
        - compatible_candidates_count: Number of candidates passing mandatory rules.
        - reason: Explanation of the matching result.
    """
    # 1. Validate top_k
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer (>= 1), got {top_k}")

    # 2. Validate acceptance_threshold
    if (
        isinstance(acceptance_threshold, bool)
        or not isinstance(acceptance_threshold, (int, float))
        or not math.isfinite(acceptance_threshold)
        or not (0.0 <= acceptance_threshold <= 1.0)
    ):
        raise ValueError(
            f"acceptance_threshold must be a finite float between 0.0 and 1.0, got {acceptance_threshold}"
        )

    # 3. Validate custom_weights
    weights = dict(DEFAULT_WEIGHTS)
    if custom_weights is not None:
        if not isinstance(custom_weights, dict):
            raise ValueError(
                f"custom_weights must be a dictionary, got {type(custom_weights).__name__}"
            )
        for attr, weight in custom_weights.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0.0
            ):
                raise ValueError(
                    f"Weight for '{attr}' must be a non-negative finite number, got {weight}"
                )
        weights.update(custom_weights)
        if sum(weights.values()) <= 0:
            raise ValueError("Sum of attribute weights must be positive (> 0).")

    # 4. Validate specifications dictionary and numeric inputs
    if specifications is None or not isinstance(specifications, dict):
        raise ValueError(
            f"specifications must be a dictionary, got {type(specifications).__name__}"
        )

    req_dim = specifications.get("dimensions_mm")
    if req_dim is not None:
        if (
            isinstance(req_dim, bool)
            or not isinstance(req_dim, (int, float))
            or not math.isfinite(req_dim)
            or req_dim <= 0
        ):
            raise ValueError(
                f"dimensions_mm must be a positive finite number (> 0), got {req_dim}"
            )

    req_stones = specifications.get("stone_count")
    if req_stones is not None:
        if (
            isinstance(req_stones, bool)
            or not isinstance(req_stones, int)
            or req_stones < 0
        ):
            raise ValueError(
                f"stone_count must be a non-negative integer (>= 0), got {req_stones}"
            )

    records = load_inventory(inventory_path)

    # Validate that at least one specification attribute was supplied
    has_spec = any(
        specifications.get(attr) is not None and (not isinstance(specifications[attr], str) or specifications[attr].strip())
        for attr in ("product_type", "material", "dimensions_mm", "stone_count")
    )
    if not has_spec:
        return {
            "is_reliable_match": False,
            "best_match": None,
            "matches": [],
            "acceptance_threshold": acceptance_threshold,
            "total_inventory_evaluated": len(records),
            "compatible_candidates_count": 0,
            "reason": "No valid specifications provided in query (all supported attributes missing or null).",
        }

    # Filter by mandatory compatibility rules and score eligible candidates
    compatible_candidates: List[Dict[str, Any]] = []

    for record in records:
        is_compatible, _ = check_mandatory_compatibility(specifications, record)
        if not is_compatible:
            continue

        score, attr_scores, explanation = score_candidate(specifications, record, weights)

        # Build candidate entry, preserving inventory price strictly
        candidate_entry = {
            "sku": str(record.get("sku", "")),
            "product_type": record.get("product_type"),
            "material": record.get("material"),
            "dimensions_mm": record.get("dimensions_mm"),
            "dimensions": record.get("dimensions_mm"),  # Convenience alias
            "stone_count": record.get("stone_count"),
            "price_inr": record.get("price_inr"),
            "price": record.get("price_inr"),            # Convenience alias
            "overall_score": score,
            "attribute_scores": attr_scores,
            "explanation": explanation,
        }
        compatible_candidates.append(candidate_entry)

    if not compatible_candidates:
        return {
            "is_reliable_match": False,
            "best_match": None,
            "matches": [],
            "acceptance_threshold": acceptance_threshold,
            "total_inventory_evaluated": len(records),
            "compatible_candidates_count": 0,
            "reason": (
                f"No compatible inventory records found matching mandatory attributes: "
                f"product_type='{specifications.get('product_type')}', "
                f"material='{specifications.get('material')}'."
            ),
        }

    # Deterministic ranking:
    # Sort primarily by overall_score DESC (rounded to 6 decimals to avoid precision anomalies),
    # and secondarily by SKU ASC as a strict deterministic tie-breaker.
    compatible_candidates.sort(
        key=lambda c: (-round(c["overall_score"], 6), c["sku"])
    )

    top_matches = compatible_candidates[:top_k]
    top_candidate = top_matches[0]

    is_reliable = top_candidate["overall_score"] >= acceptance_threshold

    if is_reliable:
        reason = (
            f"Reliable match found: Candidate '{top_candidate['sku']}' achieved overall score "
            f"{top_candidate['overall_score']:.4f} >= acceptance threshold {acceptance_threshold:.4f}."
        )
        best_match = top_candidate
    else:
        reason = (
            f"No reliable match: Best candidate '{top_candidate['sku']}' scored "
            f"{top_candidate['overall_score']:.4f}, below acceptance threshold {acceptance_threshold:.4f}."
        )
        best_match = None

    return {
        "is_reliable_match": is_reliable,
        "best_match": best_match,
        "top_candidate": top_candidate,  # Retain top candidate reference for inspection even if rejected
        "matches": top_matches,
        "acceptance_threshold": acceptance_threshold,
        "total_inventory_evaluated": len(records),
        "compatible_candidates_count": len(compatible_candidates),
        "reason": reason,
    }
