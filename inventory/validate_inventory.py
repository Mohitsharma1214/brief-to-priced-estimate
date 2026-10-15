#!/usr/bin/env python3
"""
Inventory Dataset Validator for Brief to Priced Estimate.

Validates that products.json meets all required criteria:
1. The file contains exactly 40 records.
2. All required fields exist: sku, product_type, material, dimensions_mm, stone_count, price_inr.
3. Every SKU is unique and non-empty.
4. Product types and materials belong to allowed values.
5. Dimensions are positive numbers.
6. Stone counts are non-negative integers.
7. Prices are positive numbers.
8. No required field contains a missing or invalid value.

Exits with non-zero status if validation fails, printing detailed error messages.
Exits with 0 if validation succeeds.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ALLOWED_PRODUCT_TYPES: Set[str] = {
    "ring",
    "pendant",
    "necklace",
    "bracelet",
    "earrings",
}

ALLOWED_MATERIALS: Set[str] = {
    "gold",
    "silver",
    "platinum",
}

REQUIRED_FIELDS: Tuple[str, ...] = (
    "sku",
    "product_type",
    "material",
    "dimensions_mm",
    "stone_count",
    "price_inr",
)

EXPECTED_RECORD_COUNT: int = 40


def find_dataset_file(target_path: str | None = None) -> Path:
    """Locate the products.json file based on argument or standard locations."""
    if target_path:
        p = Path(target_path)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"Specified inventory file not found: {target_path}")

    # Standard candidate locations
    candidates = [
        Path(__file__).resolve().parent / "products.json",
        Path.cwd() / "inventory" / "products.json",
        Path.cwd() / "brief-to-priced-estimate" / "inventory" / "products.json",
        Path.cwd() / "products.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find 'products.json'. Checked locations:\n"
        + "\n".join(f" - {c}" for c in candidates)
    )


def validate_record(index: int, record: Any, seen_skus: Set[str]) -> List[str]:
    """Validate a single record and return a list of error descriptions."""
    errors: List[str] = []
    record_label = f"Record #{index + 1}"

    if not isinstance(record, dict):
        return [f"{record_label}: Expected JSON object (dict), got {type(record).__name__}"]

    sku = record.get("sku")
    if sku is not None:
        record_label = f"Record #{index + 1} (SKU: {sku})"

    # Check 1: Missing required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"{record_label}: Missing required field '{field}'")
        elif record[field] is None:
            errors.append(f"{record_label}: Field '{field}' cannot be null/missing")

    # If basic fields are missing, return early for this record
    if errors:
        return errors

    # Check 2: SKU uniqueness and format
    if not isinstance(sku, str) or not sku.strip():
        errors.append(f"{record_label}: 'sku' must be a non-empty string, got {repr(sku)}")
    else:
        trimmed_sku = sku.strip()
        if trimmed_sku in seen_skus:
            errors.append(f"{record_label}: Duplicate SKU detected: '{trimmed_sku}'")
        else:
            seen_skus.add(trimmed_sku)

    # Check 3: Product type allowed values
    product_type = record.get("product_type")
    if not isinstance(product_type, str) or product_type.lower() not in ALLOWED_PRODUCT_TYPES:
        errors.append(
            f"{record_label}: Invalid 'product_type' {repr(product_type)}. "
            f"Allowed values: {sorted(ALLOWED_PRODUCT_TYPES)}"
        )

    # Check 4: Material allowed values
    material = record.get("material")
    if not isinstance(material, str) or material.lower() not in ALLOWED_MATERIALS:
        errors.append(
            f"{record_label}: Invalid 'material' {repr(material)}. "
            f"Allowed values: {sorted(ALLOWED_MATERIALS)}"
        )

    # Check 5: Dimensions positive number
    dimensions_mm = record.get("dimensions_mm")
    if (
        isinstance(dimensions_mm, bool)
        or not isinstance(dimensions_mm, (int, float))
        or not math.isfinite(dimensions_mm)
        or dimensions_mm <= 0
    ):
        errors.append(
            f"{record_label}: 'dimensions_mm' must be a positive number, got {repr(dimensions_mm)}"
        )

    # Check 6: Stone count non-negative integer
    stone_count = record.get("stone_count")
    if (
        isinstance(stone_count, bool)
        or not isinstance(stone_count, int)
        or stone_count < 0
    ):
        errors.append(
            f"{record_label}: 'stone_count' must be a non-negative integer, got {repr(stone_count)}"
        )

    # Check 7: Price positive number
    price_inr = record.get("price_inr")
    if (
        isinstance(price_inr, bool)
        or not isinstance(price_inr, (int, float))
        or not math.isfinite(price_inr)
        or price_inr <= 0
    ):
        errors.append(
            f"{record_label}: 'price_inr' must be a positive number, got {repr(price_inr)}"
        )

    return errors


def validate_inventory_file(filepath: Path) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Validate the inventory JSON file.
    Returns: (is_valid, error_list, records)
    """
    errors: List[str] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return False, [f"JSON Parse Error in {filepath}: {exc}"], []
    except Exception as exc:
        return False, [f"Failed to read {filepath}: {exc}"], []

    if not isinstance(data, list):
        return (
            False,
            [f"Top-level structure must be a JSON array (list), found {type(data).__name__}"],
            [],
        )

    actual_count = len(data)
    if actual_count != EXPECTED_RECORD_COUNT:
        errors.append(
            f"Record Count Mismatch: Expected exactly {EXPECTED_RECORD_COUNT} records, found {actual_count} records."
        )

    seen_skus: Set[str] = set()
    for idx, record in enumerate(data):
        record_errors = validate_record(idx, record, seen_skus)
        errors.extend(record_errors)

    is_valid = len(errors) == 0
    return is_valid, errors, data


def main() -> int:
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        file_path = find_dataset_file(target_arg)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Validating inventory dataset at: {file_path}")
    is_valid, errors, records = validate_inventory_file(file_path)

    if not is_valid:
        print(f"\n[VALIDATION FAILED] Found {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # If valid, print dataset summary
    print("\n[VALIDATION PASSED] All 8 criteria successfully verified!")
    print(f"  - Total records: {len(records)} (strictly matches required {EXPECTED_RECORD_COUNT})")
    print(f"  - Unique SKUs: {len(records)} / {EXPECTED_RECORD_COUNT}")

    # Compute summary breakdowns
    type_counts: Dict[str, int] = {}
    material_counts: Dict[str, int] = {}
    prices: List[float] = []
    stone_counts: List[int] = []
    dimensions: List[float] = []

    for r in records:
        ptype = r["product_type"]
        mat = r["material"]
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        material_counts[mat] = material_counts.get(mat, 0) + 1
        prices.append(float(r["price_inr"]))
        stone_counts.append(int(r["stone_count"]))
        dimensions.append(float(r["dimensions_mm"]))

    print("\nBreakdown by Product Type:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  * {ptype:<10}: {count} records")

    print("\nBreakdown by Material:")
    for mat, count in sorted(material_counts.items()):
        print(f"  * {mat:<10}: {count} records")

    print("\nValue Ranges:")
    print(f"  * Price (INR)  : min ₹{min(prices):,.2f} | max ₹{max(prices):,.2f} | avg ₹{sum(prices)/len(prices):,.2f}")
    print(f"  * Stone Count  : min {min(stone_counts)} | max {max(stone_counts)}")
    print(f"  * Dimensions   : min {min(dimensions):.1f} mm | max {max(dimensions):.1f} mm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
