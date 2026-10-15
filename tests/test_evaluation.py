"""
Unit tests for the Evaluation Harness.

Verifies:
1. Dataset completeness (>= 15 labeled cases, all 5 categories, 3 materials, negative cases).
2. Image fixtures validity (>= 5 image cases, valid decoded image bytes, documented ground-truth counts).
3. End-to-end evaluation execution produces valid metrics within bounded ranges.
4. Deterministic reproducibility across consecutive runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pytest

from evaluation.run_evaluation import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_cases.json"
IMAGES_DIR = PROJECT_ROOT / "evaluation" / "images"


def test_evaluation_dataset_completeness():
    """Verify test_cases.json meets milestone requirements."""
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)

    # At least 15 cases required (we provide 20)
    assert len(cases) >= 15

    # Check categories
    categories = {c.get("category") for c in cases}
    expected_categories = {"ring", "necklace", "bracelet", "earrings", "pendant"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    # Check materials
    materials = {c["ground_truth"].get("material") for c in cases if c["ground_truth"].get("material")}
    assert {"gold", "silver", "platinum"}.issubset(materials)

    # Check negative / unmatchable cases
    negative_cases = [c for c in cases if not c["ground_truth"].get("expected_match_accepted", True)]
    assert len(negative_cases) >= 2, "Must include at least 2 negative/unmatchable evaluation cases"

    # Check incomplete cases
    incomplete_cases = [
        c for c in cases
        if c["ground_truth"].get("dimensions_mm") is None or c["ground_truth"].get("stone_count") is None
    ]
    assert len(incomplete_cases) >= 2, "Must include incomplete brief test cases"


def test_evaluation_image_fixtures():
    """Verify at least 5 image cases exist and contain valid image bytes."""
    assert IMAGES_DIR.exists(), f"Missing images dir: {IMAGES_DIR}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)

    image_cases = [c for c in cases if c.get("image_file") is not None]
    # At least 5 image cases required (we provide 6)
    assert len(image_cases) >= 5

    for c in image_cases:
        img_name = c["image_file"]
        img_path = IMAGES_DIR / img_name
        assert img_path.exists(), f"Image fixture {img_name} does not exist"

        # Verify decodable by OpenCV
        mat = cv2.imread(str(img_path))
        assert mat is not None, f"Failed to decode image fixture: {img_path}"
        assert mat.shape[0] >= 50 and mat.shape[1] >= 50

        # Verify ground-truth count is documented
        gt_stones = c["ground_truth"].get("visible_stone_count")
        assert gt_stones is not None and isinstance(gt_stones, int) and gt_stones >= 0


def test_evaluation_run_execution(tmp_path):
    """Verify run_evaluation executes successfully and outputs valid metrics."""
    out_json = tmp_path / "test_results.json"
    out_csv = tmp_path / "test_results.csv"

    report = run_evaluation(
        dataset_path=DATASET_PATH,
        images_dir=IMAGES_DIR,
        output_json_path=out_json,
        output_csv_path=out_csv,
        use_live_llm=False,
        verbose=False,
    )

    assert out_json.exists()
    assert out_csv.exists()

    summary = report["summary"]
    ext = summary["extraction_accuracy"]
    match = summary["inventory_matching"]
    cv_met = summary["computer_vision"]
    rev = summary["human_review"]

    # Verify metric bounds
    assert 0.0 <= ext["mean_field_accuracy"] <= 1.0
    assert 0.0 <= ext["full_specification_exact_match"] <= 1.0
    assert 0.0 <= match["top1_hit_rate"] <= 1.0
    assert 0.0 <= match["top3_hit_rate"] <= 1.0
    assert cv_met["stone_count_mae"] >= 0.0
    assert 0.0 <= rev["review_flag_rate"] <= 1.0


def test_evaluation_reproducibility():
    """Verify consecutive evaluation runs produce deterministic results."""
    run1 = run_evaluation(
        dataset_path=DATASET_PATH,
        images_dir=IMAGES_DIR,
        use_live_llm=False,
    )
    run2 = run_evaluation(
        dataset_path=DATASET_PATH,
        images_dir=IMAGES_DIR,
        use_live_llm=False,
    )

    s1 = run1["summary"]
    s2 = run2["summary"]

    assert s1["extraction_accuracy"] == s2["extraction_accuracy"]
    assert s1["inventory_matching"] == s2["inventory_matching"]
    assert s1["computer_vision"]["stone_count_mae"] == s2["computer_vision"]["stone_count_mae"]
    assert s1["human_review"] == s2["human_review"]
