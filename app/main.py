"""
FastAPI Application for Brief to Priced Estimate.

Integrates:
- LLM structured specification extraction (app.extraction)
- Classical computer vision gemstone count estimation (app.vision)
- Deterministic catalog matching & pricing (app.matcher)
- Consolidated schema validation and human-review flagging (app.schemas)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from app.extraction import extract_specifications
from app.matcher import get_default_inventory_path, load_inventory, match_products
from app.schemas import (
    AnalyzeResponse,
    ConfidenceLevel,
    ExtractionStatus,
    ProductSpecification,
    VisionConfidenceLevel,
    VisionEstimationResult,
)
from app.vision import MAX_PAYLOAD_BYTES, estimate_visible_stones

logger = logging.getLogger("brief_to_priced_estimate")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Brief to Priced Estimate API",
    description=(
        "Production-minded assessment API integrating LLM structured extraction, "
        "OpenCV gemstone estimation, and deterministic catalog pricing."
    ),
    version="1.0.0",
)

# Enable CORS for external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """Logs latency, status code, and client IP for every HTTP request."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(
        "[AUDIT_LOG] method=%s path=%s status=%d duration_ms=%.2f client=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "unknown",
    )
    return response


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming inventory loading and service availability."""
    try:
        inv_path = getattr(app.state, "inventory_path", None) or get_default_inventory_path()
        records = load_inventory(inv_path)
        return {
            "status": "healthy",
            "inventory_records": len(records),
            "inventory_path": str(inv_path),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "error": str(exc),
        }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze sales brief and optional reference image to produce priced estimate",
)
async def analyze_brief(
    request: Request,
    brief: Optional[str] = Form(None, description="Customer sales brief text describing the desired jewelry item."),
    image: Optional[UploadFile] = File(None, description="Optional reference image file for visible stone count estimation."),
    top_k: int = Form(3, description="Number of candidate matches to retrieve from inventory."),
    acceptance_threshold: float = Form(0.70, description="Minimum overall score to accept an inventory match."),
) -> AnalyzeResponse:
    """
    Execute the multi-modal estimation pipeline:
    1. Validates brief text and enforces payload limits on optional image uploads.
    2. Extracts structured specifications using the configured LLM provider.
    3. Runs OpenCV classical vision to estimate visible stone count if an image was provided.
    4. Deterministically matches extracted specifications against catalog inventory records.
    5. Cross-checks brief stone count against image estimate to flag discrepancies.
    6. Returns a consolidated, schema-validated JSON response.
    """
    # 1. Validate Brief Presence and Content
    form = await request.form()
    if "brief" not in form:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[{"type": "missing", "loc": ["body", "brief"], "msg": "Field required"}],
        )

    cleaned_brief = str(brief).strip() if brief is not None else ""
    if not cleaned_brief:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sales brief must be a non-empty string.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer (>= 1).",
        )
    if not (0.0 <= acceptance_threshold <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="acceptance_threshold must be between 0.0 and 1.0.",
        )

    errors: List[str] = []
    review_flags: List[str] = []
    requires_human_review = False
    vision_result: Optional[VisionEstimationResult] = None

    # 2. Process Image Upload Safely
    if image is not None and image.filename:
        image_bytes = await image.read()
        if len(image_bytes) > MAX_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Image file exceeds maximum allowed size of {MAX_PAYLOAD_BYTES // (1024 * 1024)} MB.",
            )

        if len(image_bytes) > 0:
            try:
                vision_result = estimate_visible_stones(image_bytes)
                if vision_result.requires_human_review:
                    requires_human_review = True
                    for r in vision_result.review_reasons:
                        if r not in review_flags:
                            review_flags.append(f"Vision review: {r}")
            except ValueError as ve:
                # Graceful degradation: invalid/corrupt image must not crash text analysis
                err_msg = f"Computer vision processing failed: {str(ve)}"
                errors.append(err_msg)
                review_flags.append(f"Image analysis failed: {str(ve)}")
                requires_human_review = True
                vision_result = None
            except Exception as ex:
                err_msg = "Unexpected error during image processing."
                errors.append(err_msg)
                review_flags.append("Image processing failed due to internal error.")
                requires_human_review = True
                vision_result = None

    # 3. LLM Structured Extraction
    provider = getattr(app.state, "llm_provider", None)
    extraction_result = extract_specifications(cleaned_brief, provider=provider)
    specs = extraction_result.specifications

    if extraction_result.status == ExtractionStatus.FAILED:
        errors.append(f"Specification extraction failed: {extraction_result.error_message or 'Unknown error'}")
        review_flags.append("Failed to extract structured specifications from brief.")
        requires_human_review = True
    elif extraction_result.requires_human_review:
        requires_human_review = True
        for r in extraction_result.review_reasons:
            if r not in review_flags:
                review_flags.append(f"Extraction review: {r}")

    # 4. Specification vs Vision Stone Count Cross-Check
    if specs.stone_count is not None and vision_result is not None:
        vis_count = vision_result.estimated_visible_stones
        discrepancy = abs(specs.stone_count - vis_count)
        if discrepancy > 0:
            review_flags.append(
                f"Stone count mismatch: sales brief specified {specs.stone_count} stone(s), "
                f"but computer vision detected {vis_count} visible stone(s) (discrepancy: {discrepancy})."
            )
            requires_human_review = True

    # 5. Deterministic Inventory Matching
    inventory_path = getattr(app.state, "inventory_path", None)
    has_spec_data = any(
        getattr(specs, attr) is not None
        for attr in ("product_type", "material", "dimensions_mm", "stone_count")
    )

    inventory_matches: List[Dict[str, Any]] = []
    best_match: Optional[Dict[str, Any]] = None
    estimated_price: Optional[float] = None

    if has_spec_data:
        matcher_specs = specs.to_matcher_specifications()
        match_result = match_products(
            matcher_specs,
            inventory_path=inventory_path,
            top_k=top_k,
            acceptance_threshold=acceptance_threshold,
        )
        inventory_matches = match_result["matches"]
        best_match = match_result["best_match"]

        if not match_result["is_reliable_match"]:
            requires_human_review = True
            review_flags.append(f"Inventory matching: {match_result['reason']}")
        else:
            estimated_price = best_match["price_inr"] if best_match else None
    else:
        review_flags.append("No valid specifications extracted to query catalog.")
        requires_human_review = True

    # 6. Overall Pipeline Confidence Heuristic
    is_extraction_ok = extraction_result.status == ExtractionStatus.SUCCESS
    is_match_ok = best_match is not None and (best_match.get("overall_score", 0.0) >= acceptance_threshold)
    has_disagreement = any("mismatch" in r.lower() for r in review_flags)
    is_vision_ok = (vision_result is None) or (vision_result.confidence == VisionConfidenceLevel.HIGH)

    if is_extraction_ok and is_match_ok and is_vision_ok and not has_disagreement:
        overall_confidence = ConfidenceLevel.HIGH
    elif is_match_ok or (is_extraction_ok and not errors):
        overall_confidence = ConfidenceLevel.MEDIUM
    else:
        overall_confidence = ConfidenceLevel.LOW

    success = extraction_result.status != ExtractionStatus.FAILED

    # Per-request token and cost tracking (Stretch Goal)
    est_input_tokens = len(cleaned_brief.split()) + 120
    est_output_tokens = 60
    # Blended estimate ($0.0007/1k input, $0.0008/1k output)
    est_cost_usd = (est_input_tokens * 0.0000007) + (est_output_tokens * 0.0000008)
    logger.info(
        "[COST_AUDIT] endpoint=/analyze est_input_tokens=%d est_output_tokens=%d est_cost_usd=%.6f match_count=%d",
        est_input_tokens,
        est_output_tokens,
        est_cost_usd,
        len(inventory_matches),
    )

    return AnalyzeResponse(
        success=success,
        extracted_spec=specs,
        vision_estimate=vision_result,
        inventory_matches=inventory_matches,
        best_match=best_match,
        estimated_price_inr=estimated_price,
        confidence=overall_confidence,
        requires_human_review=requires_human_review,
        review_flags=review_flags,
        errors=errors,
    )
