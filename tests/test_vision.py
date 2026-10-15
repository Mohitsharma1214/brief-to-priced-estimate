"""
Unit tests for Classical Computer Vision Baseline (Visible Stone Estimation).

Tests cover:
1. Invalid image bytes rejection.
2. Empty input bytes rejection.
3. Synthetic images with known counts of separated circular objects (1, 3, 5, 8 circles).
4. Images with no detectable objects (flat solid backgrounds).
5. Noise speck rejection and duplicate-detection handling (Spatial NMS).
6. Resource limits enforcement (minimum and maximum dimension bounds).
7. Determinism across repeated executions on identical inputs.
8. Human-review flags for ambiguous or low-quality cases.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import VisionConfidenceLevel, VisionEstimationResult
from app.vision import (
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    estimate_visible_stones,
)


def create_synthetic_jewelry_image(
    circles: List[Tuple[int, int, int]],
    width: int = 300,
    height: int = 300,
    bg_intensity: int = 35,
    circle_intensity: int = 245,
    noise_specks: int = 0,
) -> bytes:
    """
    Generate synthetic in-memory image bytes containing circular objects (stones).

    Args:
        circles: List of (cx, cy, radius) tuples.
        width: Image width in pixels.
        height: Image height in pixels.
        bg_intensity: Grayscale intensity for background (e.g. dark setting).
        circle_intensity: Grayscale intensity for stones (e.g. bright facet/gem).
        noise_specks: Number of 1-pixel noise specks to scatter.

    Returns:
        Encoded PNG bytes.
    """
    gray = np.full((height, width), bg_intensity, dtype=np.uint8)

    for cx, cy, r in circles:
        cv2.circle(gray, (cx, cy), r, circle_intensity, -1)

    if noise_specks > 0:
        # Seeded deterministic noise
        rng = np.random.RandomState(42)
        rand_x = rng.randint(0, width, size=noise_specks)
        rand_y = rng.randint(0, height, size=noise_specks)
        gray[rand_y, rand_x] = 255

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    success, buffer = cv2.imencode(".png", bgr)
    assert success, "Failed to encode synthetic test image."
    return buffer.tobytes()


# ============================================================================
# Test 1: Invalid image bytes
# ============================================================================

def test_invalid_image_bytes_raises_error():
    """Corrupted or non-image bytes must raise ValueError."""
    corrupted_data = b"NOT_A_VALID_IMAGE_PAYLOAD_123456789"
    with pytest.raises(ValueError, match="Could not decode image bytes"):
        estimate_visible_stones(corrupted_data)

    # Non-bytes type
    with pytest.raises(ValueError, match="Expected image_bytes to be bytes"):
        estimate_visible_stones("string_instead_of_bytes")  # type: ignore


# ============================================================================
# Test 2: Empty input
# ============================================================================

def test_empty_input_raises_error():
    """Empty byte payload must raise ValueError."""
    with pytest.raises(ValueError, match="Image bytes payload is empty"):
        estimate_visible_stones(b"")


# ============================================================================
# Test 3: Synthetic images with known counts of separated circular objects
# ============================================================================

@pytest.mark.parametrize(
    "circle_specs, expected_count",
    [
        # Solitaire stone (1 circle)
        ([(150, 150, 18)], 1),
        # 3 stone arrangement
        ([(80, 150, 14), (150, 150, 18), (220, 150, 14)], 3),
        # 5 stone band
        ([(50, 150, 12), (100, 150, 12), (150, 150, 12), (200, 150, 12), (250, 150, 12)], 5),
        # 8 stone halo layout
        (
            [
                (150, 60, 10),
                (214, 86, 10),
                (240, 150, 10),
                (214, 214, 10),
                (150, 240, 10),
                (86, 214, 10),
                (60, 150, 10),
                (86, 86, 10),
            ],
            8,
        ),
    ],
)
def test_synthetic_images_known_counts(circle_specs, expected_count):
    """Detection accurately counts isolated, well-spaced circular gemstones."""
    image_bytes = create_synthetic_jewelry_image(circle_specs, width=300, height=300)
    result = estimate_visible_stones(image_bytes)

    assert result.estimated_visible_stones == expected_count
    assert len(result.detected_components) == expected_count
    assert result.confidence == VisionConfidenceLevel.HIGH
    assert result.requires_human_review is False
    assert result.diagnostics["estimated_visible_count"] == expected_count


# ============================================================================
# Test 4: Images with no detectable objects
# ============================================================================

def test_image_with_no_detectable_objects():
    """Uniform or plain background produces count 0 with human-review flag."""
    # Plain solid background (e.g. plain metal band or empty background)
    flat_image = np.full((200, 200, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", flat_image)
    image_bytes = buf.tobytes()

    result = estimate_visible_stones(image_bytes)

    assert result.estimated_visible_stones == 0
    assert result.confidence == VisionConfidenceLevel.LOW
    assert result.requires_human_review is True
    assert len(result.detected_components) == 0


# ============================================================================
# Test 5: Noise specks and duplicate detection handling
# ============================================================================

def test_noise_specks_filtered():
    """Tiny salt-and-pepper noise specks are filtered out without affecting stone count."""
    # 3 legitimate stones surrounded by 50 isolated 1-pixel noise specks
    stones = [(80, 150, 15), (150, 150, 15), (220, 150, 15)]
    noisy_bytes = create_synthetic_jewelry_image(stones, noise_specks=50)

    result = estimate_visible_stones(noisy_bytes)

    assert result.estimated_visible_stones == 3
    assert result.diagnostics["noise_specks_filtered"] >= 20
    assert result.confidence == VisionConfidenceLevel.HIGH


def test_duplicate_detection_suppressed():
    """Overlapping/duplicate detections on the same stone are suppressed via Spatial NMS."""
    from app.schemas import DetectedComponent
    from app.vision import suppress_duplicates

    # Candidate 1: Primary stone candidate at (150, 150)
    comp1 = DetectedComponent(
        centroid_x=150.0,
        centroid_y=150.0,
        radius=16.0,
        area=800.0,
        circularity=0.95,
        solidity=0.98,
    )
    box1 = (134, 134, 32, 32)

    # Candidate 2: Overlapping secondary detection (e.g. inner facet highlight) at (152, 151)
    comp2 = DetectedComponent(
        centroid_x=152.0,
        centroid_y=151.0,
        radius=12.0,
        area=452.0,
        circularity=0.88,
        solidity=0.96,
    )
    box2 = (140, 139, 24, 24)

    # Candidate 3: Distinct second stone at (230, 150)
    comp3 = DetectedComponent(
        centroid_x=230.0,
        centroid_y=150.0,
        radius=16.0,
        area=800.0,
        circularity=0.94,
        solidity=0.98,
    )
    box3 = (214, 134, 32, 32)

    candidates = [(comp1, box1), (comp2, box2), (comp3, box3)]
    deduped = suppress_duplicates(candidates)

    # Candidate 2 must be suppressed as duplicate of candidate 1
    assert len(deduped) == 2
    centroids = [(c.centroid_x, c.centroid_y) for c in deduped]
    assert (150.0, 150.0) in centroids
    assert (230.0, 150.0) in centroids


# ============================================================================
# Test 6: Resource limits enforcement
# ============================================================================

def test_image_dimension_limits():
    """Reject images smaller than 16x16px."""
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", tiny_img)

    with pytest.raises(ValueError, match="smaller than minimum allowed"):
        estimate_visible_stones(buf.tobytes())


def test_payload_size_limit():
    """Reject payloads exceeding maximum 20 MB limit."""
    huge_payload = b"0" * (21 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds limit"):
        estimate_visible_stones(huge_payload)


# ============================================================================
# Test 7: Deterministic outputs for identical input images
# ============================================================================

def test_deterministic_outputs():
    """Repeated runs on identical images must produce identical component coordinates and counts."""
    stones = [(75, 100, 12), (150, 100, 14), (225, 100, 12)]
    image_bytes = create_synthetic_jewelry_image(stones)

    baseline = estimate_visible_stones(image_bytes)

    for _ in range(15):
        repeat = estimate_visible_stones(image_bytes)
        assert repeat.estimated_visible_stones == baseline.estimated_visible_stones
        assert repeat.confidence == baseline.confidence
        assert repeat.requires_human_review == baseline.requires_human_review
        assert len(repeat.detected_components) == len(baseline.detected_components)

        for b_comp, r_comp in zip(baseline.detected_components, repeat.detected_components):
            assert b_comp.centroid_x == r_comp.centroid_x
            assert b_comp.centroid_y == r_comp.centroid_y
            assert b_comp.area == r_comp.area
            assert b_comp.circularity == r_comp.circularity


# ============================================================================
# Test 8: Review flags for ambiguous or low-quality cases
# ============================================================================

def test_extreme_aspect_ratio_triggers_review():
    """Extreme panoramic or strip aspect ratios trigger human-review recommendations."""
    # 800x80 image (10:1 aspect ratio)
    strip = np.full((80, 800, 3), 40, dtype=np.uint8)
    cv2.circle(strip, (400, 40), 15, (240, 240, 240), -1)
    _, buf = cv2.imencode(".png", strip)

    result = estimate_visible_stones(buf.tobytes())
    assert result.requires_human_review is True
    assert any("aspect ratio" in r.lower() for r in result.review_reasons)


def test_dense_clustering_triggers_uncertainty_and_review():
    """Dense clustering (e.g. >35 tightly packed stones) triggers LOW confidence and review."""
    # 40 small tightly packed circles
    circles = []
    for r_idx in range(5):
        for c_idx in range(8):
            circles.append((30 + c_idx * 30, 30 + r_idx * 30, 8))

    dense_bytes = create_synthetic_jewelry_image(circles, width=320, height=220)
    result = estimate_visible_stones(dense_bytes)

    assert result.estimated_visible_stones >= 35
    assert result.confidence == VisionConfidenceLevel.LOW
    assert result.requires_human_review is True
    assert any("clustering" in r.lower() or "density" in r.lower() for r in result.review_reasons)


# ============================================================================
# Test 9: Regression test for real five-stone photograph false-positive fix
# ============================================================================

def test_real_five_stone_photo_regression():
    """
    Regression test for the reproduction case:
    A real photograph of a gold ring with 5 clearly visible round diamonds
    previously yielded 111 false-positive stone detections due to global Otsu
    background inversion and unmerged facet/prong/reflection micro-contours.

    Under the improved pipeline:
    1. Count must be drastically suppressed from 111 to a realistic candidate range (<= 10).
    2. Real photograph on light studio background must NOT falsely claim HIGH confidence.
    3. Human review MUST be flagged (requires_human_review=True) because arbitrary photographs
       contain specular glints and facet noise that classical vision cannot guarantee 100%.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "gold_ring_five_diamonds.jpeg"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        image_bytes = f.read()

    result = estimate_visible_stones(image_bytes)

    # 1. False positive count is eliminated: count is close to actual 5 stones (not 111!)
    assert 5 <= result.estimated_visible_stones <= 10, (
        f"Expected between 5 and 10 detected stones for 5-stone ring, got {result.estimated_visible_stones}"
    )

    # 2. Honest confidence reporting on real-world photo
    assert result.confidence in (VisionConfidenceLevel.MEDIUM, VisionConfidenceLevel.LOW)

    # 3. Requires human review due to complex surface reflections
    assert result.requires_human_review is True
    assert any("reflection" in r.lower() or "photo" in r.lower() for r in result.review_reasons)

    # 4. Diagnostics verify noise specks and facets were filtered
    assert result.diagnostics["noise_specks_filtered"] > 0
    assert result.diagnostics["raw_contours_found"] < 50  # Was previously over a hundred


# ============================================================================
# Test 10: Regression tests for all standard fixtures (0, 1, 3, 5, 8, 12 stones)
# ============================================================================

@pytest.mark.parametrize(
    "fixture_name, min_expected, max_expected, expected_confidence, expected_review",
    [
        ("ring_plain_0.png", 0, 0, VisionConfidenceLevel.LOW, True),
        ("ring_solitaire_1.png", 1, 1, VisionConfidenceLevel.HIGH, False),
        ("necklace_station_3.png", 3, 3, VisionConfidenceLevel.HIGH, False),
        ("necklace_five_stones_5.png", 5, 5, VisionConfidenceLevel.HIGH, False),
        ("ring_eight_stones_8.png", 8, 8, VisionConfidenceLevel.HIGH, False),
        ("pendant_twelve_clustered_12.png", 10, 12, VisionConfidenceLevel.HIGH, False),
    ],
)
def test_all_standard_stone_fixtures(
    fixture_name: str,
    min_expected: int,
    max_expected: int,
    expected_confidence: VisionConfidenceLevel,
    expected_review: bool,
):
    """
    Verify consistent vision estimation across all standard 0, 1, 3, 5, 8, and 12-stone fixtures.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / fixture_name
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        image_bytes = f.read()

    result = estimate_visible_stones(image_bytes)

    assert min_expected <= result.estimated_visible_stones <= max_expected, (
        f"Fixture {fixture_name}: expected [{min_expected}, {max_expected}], got {result.estimated_visible_stones}"
    )
    assert result.confidence == expected_confidence, (
        f"Fixture {fixture_name}: expected confidence {expected_confidence}, got {result.confidence}"
    )
    assert result.requires_human_review == expected_review, (
        f"Fixture {fixture_name}: expected requires_human_review={expected_review}, got {result.requires_human_review}"
    )


# ============================================================================
# Test 11: False-positive-heavy synthetic inputs (glints, prongs, metal reflections)
# ============================================================================

def test_false_positive_heavy_reflections_suppressed():
    """
    Test an input heavily contaminated with specular glare, metallic edge reflections,
    and prong highlights. Verify that spurious micro-highlights are suppressed and do not
    explode into dozens of false stone detections.
    """
    # 600x600 synthetic image with 3 distinct stones (r=18)
    img = np.full((600, 600, 3), 40, dtype=np.uint8)

    # Draw metallic ring arc (diffuse gray)
    cv2.ellipse(img, (300, 300), (200, 200), 0, 0, 360, (120, 120, 120), 16)

    # 3 legitimate stones
    stone_centers = [(200, 300), (300, 300), (400, 300)]
    for cx, cy in stone_centers:
        cv2.circle(img, (cx, cy), 18, (245, 245, 245), -1)

    # Add 40 tiny specular glints along the ring arc (simulating metallic glare / prongs)
    rng = np.random.RandomState(123)
    for angle_deg in rng.uniform(0, 360, 40):
        rad = np.deg2rad(angle_deg)
        gx = int(300 + 200 * np.cos(rad) + rng.uniform(-4, 4))
        gy = int(300 + 200 * np.sin(rad) + rng.uniform(-4, 4))
        # Tiny glint 1-3 px
        cv2.circle(img, (gx, gy), rng.randint(1, 3), (255, 255, 255), -1)

    _, buf = cv2.imencode(".png", img)
    result = estimate_visible_stones(buf.tobytes())

    # The 40 micro-glints must NOT become 40 stones
    assert result.estimated_visible_stones <= 5, (
        f"Micro-glints caused false positives: got {result.estimated_visible_stones} stones (expected <= 5)"
    )


# ============================================================================
# Test 12: Intricate dense halo pendant regression test
# ============================================================================

def test_intricate_dense_pendant_not_reduced_to_few_clusters():
    """
    Regression test for the intricate pear halo diamond pendant:
    Previously, the detector returned only 5 regions, treating multiple surrounding
    halo gemstones as large clumped clusters because Hough circles only searched
    a rigid 55-85px range and missed small pavé stones (r ~ 15-22px) and the pear stone.

    Under the improved pipeline:
    1. Visible stone count must NOT collapse to ~5 cluster regions; it must resolve
       the individual visible stones across the halo (> 15 stones).
    2. Real photograph with complex surface reflections and dense pavé MUST set
       requires_human_review = True.
    3. Review reasons must document the halo setting and surface reflections.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "pendant_pear_halo_diamond.jpeg"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        image_bytes = f.read()

    result = estimate_visible_stones(image_bytes)

    # Must resolve individual visible stones (> 15), not 5 clumped clusters
    assert result.estimated_visible_stones >= 15, (
        f"Intricate pendant was undercounted/collapsed to clusters: got {result.estimated_visible_stones}"
    )

    # Must honestly flag human review for complex jewelry photograph
    assert result.requires_human_review is True
    assert any("halo" in r.lower() or "reflection" in r.lower() for r in result.review_reasons)
    assert result.confidence == VisionConfidenceLevel.MEDIUM


# ============================================================================
# Test 13: Specular glare on plain band regression test (false positive resistance)
# ============================================================================

def test_plain_ring_with_specular_glare_rejected():
    """
    Test that a plain gold band photographed on a light studio background with
    intense metallic specular highlights does NOT falsely detect glints as gemstones.
    Must return 0 stones, LOW confidence, and requires_human_review = True.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "ring_plain_glare_light_0.png"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        image_bytes = f.read()

    result = estimate_visible_stones(image_bytes)

    assert result.estimated_visible_stones == 0, (
        f"Specular glare on plain band was falsely counted as stones: got {result.estimated_visible_stones}"
    )
    assert result.confidence == VisionConfidenceLevel.LOW
    assert result.requires_human_review is True


# ============================================================================
# Test 14: Light studio background studs and trilogy ring regression tests
# ============================================================================

def test_light_background_studs_and_trilogy():
    """
    Verify exact individual stone counting on light studio backgrounds for
    stud earrings (2 stones) and trilogy ring (3 stones).
    """
    studs_path = PROJECT_ROOT / "tests" / "fixtures" / "stud_earrings_light_2.png"
    trilogy_path = PROJECT_ROOT / "tests" / "fixtures" / "trilogy_ring_light_3.png"

    with open(studs_path, "rb") as f:
        studs_res = estimate_visible_stones(f.read())
    assert studs_res.estimated_visible_stones == 2
    assert studs_res.requires_human_review is True

    with open(trilogy_path, "rb") as f:
        trilogy_res = estimate_visible_stones(f.read())
    assert trilogy_res.estimated_visible_stones == 3
    assert trilogy_res.requires_human_review is True


# ============================================================================
# Test 15: Ambiguous pavé bloom triggers uncertainty and human review
# ============================================================================

def test_ambiguous_pave_bloom_triggers_review_and_uncertainty():
    """
    When optical blooming, glare, or severe blur makes individual stone boundaries
    indistinguishable, the vision system must:
    - Lower confidence (LOW or MEDIUM)
    - Flag requires_human_review = True
    - Document the uncertainty in review_reasons
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "halo_pave_ambiguous_19.png"
    assert fixture_path.exists(), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "rb") as f:
        image_bytes = f.read()

    result = estimate_visible_stones(image_bytes)

    assert result.requires_human_review is True
    assert result.confidence in (VisionConfidenceLevel.LOW, VisionConfidenceLevel.MEDIUM)
    assert len(result.review_reasons) > 0


# ============================================================================
# Test 16: API response schema and serialization compatibility
# ============================================================================

def test_vision_estimation_schema_compatibility():
    """
    Verify that VisionEstimationResult output conforms to the schema required by
    the API response and frontend serialization.
    """
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "ring_solitaire_1.png"
    with open(fixture_path, "rb") as f:
        result = estimate_visible_stones(f.read())

    dumped = result.to_dict()
    assert "estimated_visible_stones" in dumped
    assert "method" in dumped
    assert "confidence" in dumped
    assert "confidence_reason" in dumped
    assert "requires_human_review" in dumped
    assert "review_reasons" in dumped
    assert "detected_components" in dumped
    assert "diagnostics" in dumped
    assert "limitations_disclaimer" in dumped

    # Verify component field schemas
    assert len(result.detected_components) == 1
    comp = result.detected_components[0]
    assert isinstance(comp.centroid_x, float)
    assert isinstance(comp.centroid_y, float)
    assert isinstance(comp.radius, float)
    assert isinstance(comp.area, float)
    assert isinstance(comp.circularity, float)
    assert isinstance(comp.solidity, float)


