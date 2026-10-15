"""
Fixture Generator for Brief to Priced Estimate Evaluation.

Generates labeled synthetic reference images for the evaluation harness:
- ring_plain_0.png (0 visible stones)
- ring_solitaire_1.png (1 visible stone)
- necklace_station_3.png (3 visible stones)
- necklace_five_stones_5.png (5 visible stones)
- ring_eight_stones_8.png (8 visible stones)
- pendant_twelve_clustered_12.png (12 visible stones, cluster setting)

DISCLOSURE & LIMITATION:
These images are synthetic benchmark fixtures composed of geometric circles on a
dark background. They serve as deterministic benchmarks for contour detection,
geometric isolation, and noise filtering. They DO NOT establish empirical accuracy
on real-world jewelry photographs with metallic reflections, complex facets,
prongs, or perspective distortion.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def generate_image(
    circles: List[Tuple[int, int, int]],
    width: int = 300,
    height: int = 300,
    bg_intensity: int = 35,
    circle_intensity: int = 245,
) -> np.ndarray:
    """Generate in-memory grayscale image array with rendered circle components."""
    img = np.full((height, width), bg_intensity, dtype=np.uint8)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, circle_intensity, -1)
    return img


def create_all_fixtures(output_dir: Path) -> None:
    """Create and save all 6 synthetic evaluation image fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plain band (0 stones)
    img_0 = generate_image(circles=[])
    cv2.imwrite(str(output_dir / "ring_plain_0.png"), img_0)

    # 2. Solitaire ring (1 stone)
    img_1 = generate_image(circles=[(150, 150, 15)])
    cv2.imwrite(str(output_dir / "ring_solitaire_1.png"), img_1)

    # 3. Trilogy / station necklace (3 stones)
    img_3 = generate_image(circles=[(80, 150, 12), (150, 150, 12), (220, 150, 12)])
    cv2.imwrite(str(output_dir / "necklace_station_3.png"), img_3)

    # 4. Five-stone necklace (5 stones)
    img_5 = generate_image(
        circles=[(50, 150, 10), (100, 150, 10), (150, 150, 10), (200, 150, 10), (250, 150, 10)]
    )
    cv2.imwrite(str(output_dir / "necklace_five_stones_5.png"), img_5)

    # 5. Eight-stone eternity pattern (8 stones)
    angles_8 = [i * 2 * np.pi / 8 for i in range(8)]
    circles_8 = [(int(150 + 70 * np.cos(a)), int(150 + 70 * np.sin(a)), 8) for a in angles_8]
    img_8 = generate_image(circles=circles_8)
    cv2.imwrite(str(output_dir / "ring_eight_stones_8.png"), img_8)

    # 6. Twelve-stone clustered pave pattern (12 stones, dense cluster)
    angles_12 = [i * 2 * np.pi / 12 for i in range(12)]
    circles_12 = [(int(150 + 42 * np.cos(a)), int(150 + 42 * np.sin(a)), 10) for a in angles_12]
    img_12 = generate_image(circles=circles_12)
    cv2.imwrite(str(output_dir / "pendant_twelve_clustered_12.png"), img_12)

    print(f"Generated 6 synthetic fixtures in {output_dir.resolve()}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "images"
    create_all_fixtures(out)
