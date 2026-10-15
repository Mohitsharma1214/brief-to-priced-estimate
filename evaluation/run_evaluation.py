#!/usr/bin/env python3
"""
Evaluation Harness for Brief to Priced Estimate Pipeline.

Evaluates:
1. Extraction field accuracy (per-field and overall).
2. Inventory matching Top-1 and Top-3 hit rates.
3. Stone-count Mean Absolute Error (MAE) on reference images.
4. Human review trigger rate.
5. Extraction failure and unmatched catalog query counts.

Outputs:
- Machine-readable JSON summary and per-case logs.
- Machine-readable CSV log table.
- Formatted command-line performance report.

Can be run 100% offline and reproducibly without requiring a live LLM API key.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.vision import estimate_visible_stones
from tests.test_extraction import MockLLMProvider


def run_evaluation(
    dataset_path: Path,
    images_dir: Path,
    output_json_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    use_live_llm: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation over all test cases in dataset."""
    if use_live_llm:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required to run evaluation in live mode (--live-llm).\n"
                "Please configure OPENROUTER_API_KEY or omit --live-llm to execute the deterministic offline benchmark."
            )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    client = TestClient(app)
    case_results: List[Dict[str, Any]] = []

    # Counters
    field_matches = {
        "product_type": 0,
        "material": 0,
        "dimensions_mm": 0,
        "stone_count": 0,
    }
    full_spec_exact_matches = 0
    top1_hits = 0
    top3_hits = 0
    positive_top1_hits = 0
    positive_top3_hits = 0
    positive_cases_count = 0
    negative_cases_count = 0
    negative_rejections_count = 0
    review_flagged_count = 0
    extraction_failures_count = 0
    unmatched_cases_count = 0
    vision_absolute_errors: List[float] = []

    start_time = time.time()

    for idx, case in enumerate(test_cases, 1):
        cid = case["id"]
        cname = case["name"]
        brief = case["brief"]
        img_filename = case.get("image_file")
        gt = case["ground_truth"]
        mock_ext = case.get("mock_extraction", {})

        # Prepare LLM provider
        if not use_live_llm:
            mock_json_str = json.dumps(mock_ext)
            # Provide multiple identical responses for any retry attempts
            app.state.llm_provider = MockLLMProvider([mock_json_str, mock_json_str, mock_json_str])
        else:
            app.state.llm_provider = None

        # Prepare image file if present
        files: Dict[str, Any] = {}
        if img_filename:
            img_path = images_dir / img_filename
            if img_path.exists():
                with open(img_path, "rb") as img_f:
                    files["image"] = (img_filename, img_f.read(), "image/png")
            else:
                print(f"[WARN] Image file {img_filename} not found at {img_path}")

        case_start = time.time()
        response = client.post("/analyze", data={"brief": brief}, files=files)
        latency_ms = (time.time() - case_start) * 1000.0

        status_code = response.status_code
        payload = response.json() if status_code == 200 else {"error": response.text}

        # 1. Evaluate Extraction Accuracy
        extracted_spec = payload.get("extracted_spec") or {}
        pred_pt = extracted_spec.get("product_type")
        pred_mat = extracted_spec.get("material")
        pred_dim = extracted_spec.get("dimensions_mm")
        pred_stone = extracted_spec.get("stone_count")

        gt_pt = gt.get("product_type")
        gt_mat = gt.get("material")
        gt_dim = gt.get("dimensions_mm")
        gt_stone = gt.get("stone_count")

        pt_match = pred_pt == gt_pt
        mat_match = pred_mat == gt_mat
        dim_match = (
            (pred_dim is None and gt_dim is None)
            or (
                pred_dim is not None
                and gt_dim is not None
                and math.isclose(float(pred_dim), float(gt_dim), abs_tol=0.2)
            )
        )
        stone_match = pred_stone == gt_stone

        if pt_match:
            field_matches["product_type"] += 1
        if mat_match:
            field_matches["material"] += 1
        if dim_match:
            field_matches["dimensions_mm"] += 1
        if stone_match:
            field_matches["stone_count"] += 1

        all_fields_match = pt_match and mat_match and dim_match and stone_match
        if all_fields_match:
            full_spec_exact_matches += 1

        # 2. Evaluate Matching Accuracy & Pricing
        best_match = payload.get("best_match")
        best_sku = best_match.get("sku") if best_match else None
        matches = [m.get("sku") for m in payload.get("inventory_matches", [])]
        expected_sku = gt.get("target_sku")
        expected_accepted = gt.get("expected_match_accepted", True)

        case_top1 = False
        case_top3 = False

        if expected_accepted and expected_sku is not None:
            positive_cases_count += 1
            if best_sku == expected_sku:
                top1_hits += 1
                positive_top1_hits += 1
                case_top1 = True
            if expected_sku in matches[:3]:
                top3_hits += 1
                positive_top3_hits += 1
                case_top3 = True
        else:
            negative_cases_count += 1
            # For negative / unmatchable cases, success means rejecting the match (best_match is None)
            if best_sku is None:
                top1_hits += 1
                top3_hits += 1
                negative_rejections_count += 1
                case_top1 = True
                case_top3 = True

        # Unmatched catalog query check
        if best_match is None or payload.get("estimated_price_inr") is None:
            unmatched_cases_count += 1

        # Extraction failure check
        if not payload.get("success", False):
            extraction_failures_count += 1

        # Review flag check
        requires_review = payload.get("requires_human_review", False)
        if requires_review:
            review_flagged_count += 1

        # 3. Vision Stone Count Accuracy
        gt_vis_stones = gt.get("visible_stone_count")
        vis_estimate = payload.get("vision_estimate")
        pred_vis_stones = vis_estimate.get("estimated_visible_stones") if vis_estimate else None
        vis_err = None

        if gt_vis_stones is not None and pred_vis_stones is not None:
            vis_err = abs(pred_vis_stones - gt_vis_stones)
            vision_absolute_errors.append(vis_err)

        # Log case record
        case_log = {
            "id": cid,
            "name": cname,
            "category": case.get("category"),
            "brief": brief,
            "has_image": bool(img_filename),
            "status_code": status_code,
            "success": payload.get("success", False),
            "extraction": {
                "predicted": {
                    "product_type": pred_pt,
                    "material": pred_mat,
                    "dimensions_mm": pred_dim,
                    "stone_count": pred_stone,
                },
                "ground_truth": {
                    "product_type": gt_pt,
                    "material": gt_mat,
                    "dimensions_mm": gt_dim,
                    "stone_count": gt_stone,
                },
                "matches": {
                    "product_type": pt_match,
                    "material": mat_match,
                    "dimensions_mm": dim_match,
                    "stone_count": stone_match,
                    "all": all_fields_match,
                },
            },
            "matching": {
                "expected_sku": expected_sku,
                "predicted_best_sku": best_sku,
                "matches_top3": matches[:3],
                "top1_hit": case_top1,
                "top3_hit": case_top3,
                "estimated_price_inr": payload.get("estimated_price_inr"),
            },
            "vision": {
                "image_file": img_filename,
                "ground_truth_stones": gt_vis_stones,
                "predicted_stones": pred_vis_stones,
                "absolute_error": vis_err,
                "confidence": vis_estimate.get("confidence") if vis_estimate else None,
            },
            "review": {
                "requires_human_review": requires_review,
                "flags": payload.get("review_flags", []),
            },
            "errors": payload.get("errors", []),
            "latency_ms": round(latency_ms, 2),
        }
        case_results.append(case_log)

        if verbose:
            flag_info = f" [REVIEW: {len(payload.get('review_flags', []))} flags]" if requires_review else ""
            print(
                f"[{cid}] {cname} -> HTTP {status_code} | Top1: {case_top1} | "
                f"Pred SKU: {best_sku} (GT: {expected_sku}) | Latency: {latency_ms:.1f}ms{flag_info}"
            )

    total_cases = len(test_cases)
    total_time = time.time() - start_time

    # Calculate aggregate summary metrics
    pt_acc = field_matches["product_type"] / total_cases
    mat_acc = field_matches["material"] / total_cases
    dim_acc = field_matches["dimensions_mm"] / total_cases
    stone_acc = field_matches["stone_count"] / total_cases
    mean_field_acc = (pt_acc + mat_acc + dim_acc + stone_acc) / 4.0
    full_spec_acc = full_spec_exact_matches / total_cases

    top1_rate = top1_hits / total_cases
    top3_rate = top3_hits / total_cases
    pos_top1_rate = (
        positive_top1_hits / positive_cases_count if positive_cases_count > 0 else 0.0
    )
    pos_top3_rate = (
        positive_top3_hits / positive_cases_count if positive_cases_count > 0 else 0.0
    )

    neg_rejection_rate = (
        negative_rejections_count / negative_cases_count
        if negative_cases_count > 0
        else 0.0
    )

    stone_mae = (
        sum(vision_absolute_errors) / len(vision_absolute_errors)
        if vision_absolute_errors
        else 0.0
    )
    review_rate = review_flagged_count / total_cases

    summary_metrics = {
        "dataset_metadata": {
            "total_cases": total_cases,
            "positive_cases": positive_cases_count,
            "negative_cases": negative_cases_count,
            "image_cases": len(vision_absolute_errors),
            "execution_mode": "live_llm" if use_live_llm else "offline_recorded_mock",
            "total_duration_sec": round(total_time, 2),
        },
        "extraction_accuracy": {
            "product_type": round(pt_acc, 4),
            "material": round(mat_acc, 4),
            "dimensions_mm": round(dim_acc, 4),
            "stone_count": round(stone_acc, 4),
            "mean_field_accuracy": round(mean_field_acc, 4),
            "full_specification_exact_match": round(full_spec_acc, 4),
            "extraction_failures": extraction_failures_count,
        },
        "inventory_matching": {
            "top1_hit_rate": round(top1_rate, 4),
            "top3_hit_rate": round(top3_rate, 4),
            "positive_cases_top1_hit_rate": round(pos_top1_rate, 4),
            "positive_cases_top3_hit_rate": round(pos_top3_rate, 4),
            "negative_cases_rejection_rate": round(neg_rejection_rate, 4),
            "positive_cases_count": positive_cases_count,
            "positive_cases_top1_hits": positive_top1_hits,
            "positive_cases_top3_hits": positive_top3_hits,
            "negative_cases_count": negative_cases_count,
            "negative_rejections_count": negative_rejections_count,
            "unmatched_cases_count": unmatched_cases_count,
        },
        "computer_vision": {
            "image_cases_count": len(vision_absolute_errors),
            "stone_count_mae": round(stone_mae, 4),
            "disclaimer": (
                "Evaluated on controlled synthetic geometric fixtures. "
                "Does not establish empirical accuracy on real jewelry photographs."
            ),
        },
        "human_review": {
            "review_flagged_cases": review_flagged_count,
            "review_flag_rate": round(review_rate, 4),
        },
    }

    full_report = {
        "summary": summary_metrics,
        "cases": case_results,
    }

    # Save JSON if requested
    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"Saved machine-readable JSON results to: {output_json_path.resolve()}")

    # Save CSV if requested
    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for c in case_results:
            csv_rows.append(
                {
                    "case_id": c["id"],
                    "case_name": c["name"],
                    "category": c["category"],
                    "has_image": c["has_image"],
                    "gt_product_type": c["extraction"]["ground_truth"]["product_type"],
                    "pred_product_type": c["extraction"]["predicted"]["product_type"],
                    "match_product_type": c["extraction"]["matches"]["product_type"],
                    "gt_material": c["extraction"]["ground_truth"]["material"],
                    "pred_material": c["extraction"]["predicted"]["material"],
                    "match_material": c["extraction"]["matches"]["material"],
                    "gt_dimensions_mm": c["extraction"]["ground_truth"]["dimensions_mm"],
                    "pred_dimensions_mm": c["extraction"]["predicted"]["dimensions_mm"],
                    "match_dimensions": c["extraction"]["matches"]["dimensions_mm"],
                    "gt_stone_count": c["extraction"]["ground_truth"]["stone_count"],
                    "pred_stone_count": c["extraction"]["predicted"]["stone_count"],
                    "match_stone_count": c["extraction"]["matches"]["stone_count"],
                    "full_spec_match": c["extraction"]["matches"]["all"],
                    "gt_target_sku": c["matching"]["expected_sku"],
                    "pred_best_sku": c["matching"]["predicted_best_sku"],
                    "top1_hit": c["matching"]["top1_hit"],
                    "top3_hit": c["matching"]["top3_hit"],
                    "estimated_price_inr": c["matching"]["estimated_price_inr"],
                    "gt_visible_stones": c["vision"]["ground_truth_stones"],
                    "pred_visible_stones": c["vision"]["predicted_stones"],
                    "vision_error": c["vision"]["absolute_error"],
                    "requires_review": c["review"]["requires_human_review"],
                    "review_flags_count": len(c["review"]["flags"]),
                    "latency_ms": c["latency_ms"],
                }
            )

        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved machine-readable CSV results to: {output_csv_path.resolve()}")

    # Print Formatted Report to Console
    print_report_table(summary_metrics, case_results)
    return full_report


def print_report_table(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> None:
    """Print ASCII summary table to stdout."""
    ext = summary["extraction_accuracy"]
    match = summary["inventory_matching"]
    cv_met = summary["computer_vision"]
    rev = summary["human_review"]
    meta = summary["dataset_metadata"]

    print("\n" + "=" * 78)
    print("           BRIEF TO PRICED ESTIMATE - EVALUATION REPORT")
    print("=" * 78)
    print(f"Execution Mode:       {meta['execution_mode']}")
    print(f"Total Test Cases:     {meta['total_cases']} (Positive: {meta['positive_cases']}, Negative: {meta['negative_cases']})")
    print(f"Image Test Cases:     {meta['image_cases']}")
    print(f"Total Duration:       {meta['total_duration_sec']}s")
    print("-" * 78)
    print("1. EXTRACTION ACCURACY")
    print(f"  * Product Type:     {ext['product_type'] * 100:.1f}%")
    print(f"  * Material:         {ext['material'] * 100:.1f}%")
    print(f"  * Dimensions (mm):  {ext['dimensions_mm'] * 100:.1f}%")
    print(f"  * Stone Count:      {ext['stone_count'] * 100:.1f}%")
    print(f"  * Mean Field Acc:   {ext['mean_field_accuracy'] * 100:.1f}%")
    print(f"  * Full Spec Match:  {ext['full_specification_exact_match'] * 100:.1f}%")
    print(f"  * Extr. Failures:   {ext['extraction_failures']}")
    print("-" * 78)
    print("2. INVENTORY MATCHING & PRICING")
    print(f"  * Positive Top-1 Hit Rate: {match['positive_cases_top1_hit_rate'] * 100:.1f}% ({match['positive_cases_top1_hits']}/{match['positive_cases_count']})")
    print(f"  * Positive Top-3 Hit Rate: {match['positive_cases_top3_hit_rate'] * 100:.1f}% ({match['positive_cases_top3_hits']}/{match['positive_cases_count']})")
    print(f"  * Negative Rejection Rate: {match['negative_cases_rejection_rate'] * 100:.1f}% ({match['negative_rejections_count']}/{match['negative_cases_count']})")
    print(f"  * Overall Top-1 Accuracy:  {match['top1_hit_rate'] * 100:.1f}% ({match['positive_cases_top1_hits'] + match['negative_rejections_count']}/{meta['total_cases']})")
    print(f"  * Overall Top-3 Accuracy:  {match['top3_hit_rate'] * 100:.1f}% ({match['positive_cases_top3_hits'] + match['negative_rejections_count']}/{meta['total_cases']})")
    print(f"  * Unmatched Catalog Queries: {match['unmatched_cases_count']}")
    print("-" * 78)
    print("3. COMPUTER VISION (VISIBLE STONE COUNT)")
    print(f"  * Evaluated Images: {cv_met['image_cases_count']}")
    print(f"  * Stone Count MAE:  {cv_met['stone_count_mae']:.3f} stones (denominator: n={cv_met['image_cases_count']})")
    print("-" * 78)
    print("4. RELIABILITY & HUMAN REVIEW")
    print(f"  * Flagged for Review: {rev['review_flagged_cases']}/{meta['total_cases']} ({rev['review_flag_rate'] * 100:.1f}%)")
    print("=" * 78)


BASELINE_PREDICTIONS: Dict[str, int] = {
    "ring_plain_0.png": 0,
    "ring_solitaire_1.png": 1,
    "stud_earrings_light_2.png": 16,
    "necklace_station_3.png": 3,
    "trilogy_ring_light_3.png": 8,
    "necklace_five_stones_5.png": 5,
    "gold_ring_five_diamonds.jpeg": 111,
    "ring_eight_stones_8.png": 8,
    "pendant_twelve_clustered_12.png": 10,
    "ring_plain_glare_light_0.png": 0,
    "pendant_pear_halo_diamond.jpeg": 5,
    "halo_pave_ambiguous_19.png": 12,
}


def run_vision_benchmark(
    manifest_path: Path,
    images_dir: Path,
    output_json_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute reproducible computer vision visible gemstone counting benchmark
    across all fixtures in manifest.json, comparing improved pipeline against baseline.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Vision manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_cases: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []
    eval_errors: List[float] = []
    baseline_eval_errors: List[float] = []
    exact_count_hits = 0
    within_one_hits = 0
    overcount_count = 0
    undercount_count = 0
    review_flagged_count = 0
    evaluated_cases_count = 0
    ambiguous_excluded_count = 0

    for case in manifest_cases:
        cid = case["id"]
        filename = case["name"]
        gt_stones = case["ground_truth_visible_stones"]
        is_ambiguous = case.get("ambiguous", False)
        setting_type = case.get("setting_type", "unknown")
        baseline_pred = BASELINE_PREDICTIONS.get(filename, 0)
        baseline_err = abs(baseline_pred - gt_stones)

        img_path = images_dir / filename
        if not img_path.exists():
            print(f"[WARN] Benchmark image {filename} not found at {img_path}")
            continue

        with open(img_path, "rb") as img_f:
            image_bytes = img_f.read()

        vis_result = estimate_visible_stones(image_bytes)
        pred_stones = vis_result.estimated_visible_stones
        err = abs(pred_stones - gt_stones)
        requires_review = vis_result.requires_human_review
        confidence = vis_result.confidence.value
        method = vis_result.method
        review_reasons = vis_result.review_reasons

        if requires_review:
            review_flagged_count += 1

        if is_ambiguous:
            ambiguous_excluded_count += 1
        else:
            evaluated_cases_count += 1
            eval_errors.append(err)
            baseline_eval_errors.append(baseline_err)
            if pred_stones == gt_stones:
                exact_count_hits += 1
            if err <= 1:
                within_one_hits += 1
            if pred_stones > gt_stones:
                overcount_count += 1
            elif pred_stones < gt_stones:
                undercount_count += 1

        case_record = {
            "id": cid,
            "filename": filename,
            "setting_type": setting_type,
            "background": case.get("background"),
            "ground_truth_visible_stones": gt_stones,
            "baseline_predicted_stones": baseline_pred,
            "baseline_absolute_error": baseline_err,
            "predicted_visible_stones": pred_stones,
            "absolute_error": err,
            "confidence": confidence,
            "requires_human_review": requires_review,
            "review_reasons": review_reasons,
            "method": method,
            "ambiguous": is_ambiguous,
            "exclusion_reason": case.get("exclusion_reason"),
            "diagnostics": vis_result.diagnostics,
        }
        results.append(case_record)

    mae = float(np.mean(eval_errors)) if eval_errors else 0.0
    baseline_mae = float(np.mean(baseline_eval_errors)) if baseline_eval_errors else 0.0
    exact_acc = (exact_count_hits / evaluated_cases_count) if evaluated_cases_count else 0.0
    within_one_acc = (within_one_hits / evaluated_cases_count) if evaluated_cases_count else 0.0
    overcount_rate = (overcount_count / evaluated_cases_count) if evaluated_cases_count else 0.0
    undercount_rate = (undercount_count / evaluated_cases_count) if evaluated_cases_count else 0.0
    review_rate = (review_flagged_count / len(results)) if results else 0.0

    summary = {
        "total_images": len(results),
        "evaluated_images": evaluated_cases_count,
        "ambiguous_excluded_images": ambiguous_excluded_count,
        "improved_pipeline": {
            "mae_stones": round(mae, 3),
            "exact_count_accuracy": round(exact_acc, 4),
            "exact_count_hits": exact_count_hits,
            "within_one_stone_accuracy": round(within_one_acc, 4),
            "within_one_stone_hits": within_one_hits,
            "overcount_rate": round(overcount_rate, 4),
            "undercount_rate": round(undercount_rate, 4),
            "review_flag_rate": round(review_rate, 4),
            "review_flagged_images": review_flagged_count,
        },
        "baseline_pipeline": {
            "mae_stones": round(baseline_mae, 3),
        },
    }

    full_output = {
        "summary": summary,
        "cases": results,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
        print(f"Saved machine-readable vision JSON results to: {output_json_path.resolve()}")

    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for r in results:
            csv_rows.append(
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "setting_type": r["setting_type"],
                    "background": r["background"],
                    "ground_truth": r["ground_truth_visible_stones"],
                    "baseline_pred": r["baseline_predicted_stones"],
                    "baseline_error": r["baseline_absolute_error"],
                    "improved_pred": r["predicted_visible_stones"],
                    "improved_error": r["absolute_error"],
                    "confidence": r["confidence"],
                    "requires_review": r["requires_human_review"],
                    "method": r["method"],
                    "ambiguous": r["ambiguous"],
                }
            )
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved machine-readable vision CSV results to: {output_csv_path.resolve()}")

    print_vision_report_table(summary, results)
    return full_output


def print_vision_report_table(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> None:
    """Print ASCII comparison table for computer vision gemstone benchmark."""
    imp = summary["improved_pipeline"]
    base = summary["baseline_pipeline"]

    print("\n" + "=" * 94)
    print("      AURAGEM STUDIO - INDIVIDUAL GEMSTONE COUNTING BENCHMARK (CV)")
    print("=" * 94)
    print(f"Evaluated Fixtures (Primary MAE) : {summary['evaluated_images']}")
    print(f"Ambiguous / Excluded Fixtures   : {summary['ambiguous_excluded_images']}")
    print("-" * 94)
    print(f"{'Image Fixture':32s} | {'GT':2s} | {'Base (Err)':10s} | {'Imp (Err)':10s} | {'Conf':6s} | {'Rev':5s} | {'Status'}")
    print("-" * 94)

    for c in cases:
        fname = c["filename"]
        gt = c["ground_truth_visible_stones"]
        b_pred = c["baseline_predicted_stones"]
        b_err = c["baseline_absolute_error"]
        b_sign = "+" if b_pred >= gt else "-"
        b_str = f"{b_pred:2d} ({b_sign}{b_err:2d})" if b_err != 0 else f"{b_pred:2d} ( 0)"

        i_pred = c["predicted_visible_stones"]
        i_err = c["absolute_error"]
        i_sign = "+" if i_pred >= gt else "-"
        i_str = f"{i_pred:2d} ({i_sign}{i_err:2d})" if i_err != 0 else f"{i_pred:2d} ( 0)"

        status = "Ambiguous (Excluded)" if c["ambiguous"] else ("Exact" if i_err == 0 else f"Off by {i_err}")
        conf_str = c["confidence"].upper()
        rev_str = "YES" if c["requires_human_review"] else "no"

        print(f"{fname:32s} | {gt:2d} | {b_str:10s} | {i_str:10s} | {conf_str:6s} | {rev_str:5s} | {status}")

    print("-" * 94)
    print("AGGREGATE METRICS COMPARISON:")
    print(f"  * Baseline MAE                 : {base['mae_stones']:.2f} stones")
    print(f"  * Improved Pipeline MAE        : {imp['mae_stones']:.2f} stones (improvement: {base['mae_stones'] - imp['mae_stones']:.2f} stones lower error)")
    print(f"  * Exact-Count Accuracy         : {imp['exact_count_accuracy'] * 100:.1f}% ({imp['exact_count_hits']}/{summary['evaluated_images']})")
    print(f"  * Within-1-Stone Accuracy      : {imp['within_one_stone_accuracy'] * 100:.1f}% ({imp['within_one_stone_hits']}/{summary['evaluated_images']})")
    print(f"  * Overcount Rate               : {imp['overcount_rate'] * 100:.1f}%")
    print(f"  * Undercount Rate              : {imp['undercount_rate'] * 100:.1f}%")
    print(f"  * Review-Flag Rate             : {imp['review_flag_rate'] * 100:.1f}% ({imp['review_flagged_images']}/{summary['total_images']})")
    print("=" * 94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Brief to Priced Estimate & Gemstone Vision Evaluation Harness")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "test_cases.json",
        help="Path to evaluation test_cases.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifest.json",
        help="Path to gemstone vision evaluation manifest.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "images",
        help="Path to evaluation images directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results.json",
        help="Path to write machine-readable JSON output",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results.csv",
        help="Path to write machine-readable CSV output",
    )
    parser.add_argument(
        "--vision-output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "vision_results.json",
        help="Path to write machine-readable vision JSON benchmark",
    )
    parser.add_argument(
        "--vision-output-csv",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "vision_results.csv",
        help="Path to write machine-readable vision CSV benchmark",
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Run against live LLM API instead of offline recorded mocks",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Execute only the computer vision gemstone benchmark",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-case execution logs",
    )

    args = parser.parse_args()
    try:
        if args.vision_only:
            run_vision_benchmark(
                manifest_path=args.manifest,
                images_dir=args.images_dir,
                output_json_path=args.vision_output_json,
                output_csv_path=args.vision_output_csv,
                verbose=args.verbose,
            )
        else:
            run_evaluation(
                dataset_path=args.dataset,
                images_dir=args.images_dir,
                output_json_path=args.output_json,
                output_csv_path=args.output_csv,
                use_live_llm=args.live_llm,
                verbose=args.verbose,
            )
            # Also run the full gemstone vision benchmark
            run_vision_benchmark(
                manifest_path=args.manifest,
                images_dir=args.images_dir,
                output_json_path=args.vision_output_json,
                output_csv_path=args.vision_output_csv,
                verbose=args.verbose,
            )
    except ValueError as ve:
        print(f"\n[ERROR] {ve}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

