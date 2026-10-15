"""
Pydantic schemas for LLM-Based Structured Extraction.

Defines models for:
- Product specifications (product_type, material, dimensions_mm, stone_count).
- Explicit field-level confidence and evidence reasoning.
- Extraction status, retry tracking, and human-review flags.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductTypeEnum(str, Enum):
    RING = "ring"
    PENDANT = "pendant"
    NECKLACE = "necklace"
    BRACELET = "bracelet"
    EARRINGS = "earrings"


class MaterialEnum(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    PLATINUM = "platinum"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FieldConfidence(BaseModel):
    """Field-level confidence assessment and evidence trail."""
    model_config = ConfigDict(extra="ignore")

    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    reason: str = "Unspecified in sales brief."
    evidence: Optional[str] = None


class ProductSpecification(BaseModel):
    """
    Extracted physical jewelry specifications.

    Unknown or unmentioned attributes are explicitly represented as None.
    Never silently fabulates unknown values into facts.
    """
    model_config = ConfigDict(extra="ignore")

    product_type: Optional[ProductTypeEnum] = None
    material: Optional[MaterialEnum] = None
    dimensions_mm: Optional[float] = None
    stone_count: Optional[int] = None
    field_confidences: Dict[str, FieldConfidence] = Field(default_factory=dict)

    @field_validator("product_type", mode="before")
    @classmethod
    def validate_product_type(cls, v: Any) -> Optional[ProductTypeEnum]:
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip().lower()
            if not clean or clean in ("null", "none", "unknown", "unspecified"):
                return None
            for member in ProductTypeEnum:
                if member.value == clean:
                    return member
            raise ValueError(
                f"Invalid product_type '{v}'. Allowed values: {[e.value for e in ProductTypeEnum]}"
            )
        if isinstance(v, ProductTypeEnum):
            return v
        raise ValueError(f"Unsupported type for product_type: {type(v).__name__}")

    @field_validator("material", mode="before")
    @classmethod
    def validate_material(cls, v: Any) -> Optional[MaterialEnum]:
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip().lower()
            if not clean or clean in ("null", "none", "unknown", "unspecified"):
                return None
            for member in MaterialEnum:
                if member.value == clean:
                    return member
            raise ValueError(
                f"Invalid material '{v}'. Allowed values: {[e.value for e in MaterialEnum]}"
            )
        if isinstance(v, MaterialEnum):
            return v
        raise ValueError(f"Unsupported type for material: {type(v).__name__}")

    @field_validator("dimensions_mm", mode="before")
    @classmethod
    def validate_dimensions(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("null", "none", "unknown", "unspecified"):
                return None
            try:
                v = float(clean)
            except (ValueError, TypeError):
                raise ValueError(f"dimensions_mm must be a valid numeric value, got '{clean}'")

        if isinstance(v, bool):
            raise ValueError("dimensions_mm cannot be a boolean")

        if not isinstance(v, (int, float)):
            raise ValueError(f"dimensions_mm must be a number, got {type(v).__name__}")

        val = float(v)
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"dimensions_mm must be a positive finite number (> 0), got {val}")
        if val > 2000.0:
            raise ValueError(f"dimensions_mm exceeds realistic jewelry range (max 2000mm), got {val}")

        return round(val, 2)

    @field_validator("stone_count", mode="before")
    @classmethod
    def validate_stone_count(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip()
            if not clean or clean.lower() in ("null", "none", "unknown", "unspecified"):
                return None
            try:
                v = float(clean)
            except (ValueError, TypeError):
                raise ValueError(f"stone_count must be an integer, got '{clean}'")

        if isinstance(v, bool):
            raise ValueError("stone_count cannot be a boolean")

        if not isinstance(v, (int, float)):
            raise ValueError(f"stone_count must be an integer, got {type(v).__name__}")

        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"stone_count must be a whole integer, got {v}")

        val = int(v)
        if val < 0:
            raise ValueError(f"stone_count must be a non-negative integer (>= 0), got {val}")
        if val > 1000:
            raise ValueError(f"stone_count exceeds realistic threshold (max 1000), got {val}")

        return val

    def to_matcher_specifications(self) -> Dict[str, Any]:
        """Convert to dict matching app.matcher.match_products parameter specifications."""
        return {
            "product_type": self.product_type.value if self.product_type else None,
            "material": self.material.value if self.material else None,
            "dimensions_mm": self.dimensions_mm,
            "stone_count": self.stone_count,
        }


class ExtractionResult(BaseModel):
    """Encapsulates the complete result of an extraction attempt."""
    model_config = ConfigDict(extra="ignore")

    status: ExtractionStatus
    specifications: ProductSpecification
    raw_response: Optional[str] = None
    retries_attempted: int = 0
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    provider_info: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to JSON-serializable dictionary."""
        return self.model_dump(mode="json")


# ============================================================================
# Computer Vision Schemas
# ============================================================================

class VisionConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DetectedComponent(BaseModel):
    """Details of a single candidate repeated component (e.g. visible gemstone)."""
    model_config = ConfigDict(extra="ignore")

    centroid_x: float
    centroid_y: float
    radius: float
    area: float
    circularity: float
    solidity: float


class VisionEstimationResult(BaseModel):
    """
    Diagnostic result of classical computer vision visible stone estimation.

    Note on uncertainty:
    Visible count represents foreground components detectable under classical
    filtering, NOT the total gemstone count in the intended 3D design.
    """
    model_config = ConfigDict(extra="ignore")

    estimated_visible_stones: int
    method: str = "classical_morphological_contour_analysis"
    confidence: VisionConfidenceLevel
    confidence_reason: str
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    detected_components: List[DetectedComponent] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    limitations_disclaimer: str = (
        "Visible stone estimation reflects 2D surface components detected by classical "
        "filters. Hidden, occluded, pavilion, or underside stones in the 3D design cannot "
        "be detected and require human or CAD verification."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert vision result to JSON-serializable dictionary."""
        return self.model_dump(mode="json")


# ============================================================================
# API Response Schemas
# ============================================================================

class AnalyzeResponse(BaseModel):
    """
    Consolidated response schema for POST /analyze.

    Integrates LLM structured extraction, computer vision stone count,
    deterministic inventory matching, and human-review flags.
    """
    model_config = ConfigDict(extra="ignore")

    success: bool
    extracted_spec: Optional[ProductSpecification] = None
    vision_estimate: Optional[VisionEstimationResult] = None
    inventory_matches: List[Dict[str, Any]] = Field(default_factory=list)
    best_match: Optional[Dict[str, Any]] = None
    estimated_price_inr: Optional[float] = None
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    requires_human_review: bool = False
    review_flags: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    pricing_disclaimer: str = (
        "Estimated price reflects matching catalog items from current inventory. "
        "This is an illustrative benchmark, not a custom fabrication quote."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to JSON-serializable dictionary."""
        return self.model_dump(mode="json")
