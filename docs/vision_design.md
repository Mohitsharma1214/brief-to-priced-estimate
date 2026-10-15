# Classical Computer Vision Baseline: Visible Stone Estimation

## 1. Overview

The **Computer Vision Baseline Module** (`app/vision.py`) uses classical image processing techniques implemented with **OpenCV** and **NumPy** to detect and count visible repeated small components (gemstones) in uploaded reference jewelry imagery.

The module operates under strict physical and algorithmic constraints:
- **Zero neural network weights**: No heavy deep learning dependencies or GPU requirements.
- **Deterministic output**: Identical input images yield identical coordinates, counts, and diagnostics.
- **Explicit uncertainty modeling**: Clearly separates visible 2D surface features from the true 3D design gemstone count.

---

## 2. Classical Vision Pipeline

```text
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Image Bytes    │ ──> │ Safe Decode & Bounds │ ──> │ Bilateral Smoothing    │
│  (PNG/JPG/WebP) │     │ (Min 16px, Max 4096) │     │ (Preserve crisp edges) │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Heuristic Review│ <── │ Spatial NMS Dedup    │ <── │ Binarization & Contour │
│ & Confidence    │     │ (IoU & Proximity)    │     │ Geometric Filtering    │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

### Pipeline Steps:

1. **Safe Decoding & Dimension Limits**:
   - Decodes bytes using `cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)`.
   - Enforces minimum dimensions ($\ge 16\times 16\text{ px}$) to prevent degenerate inputs.
   - Enforces maximum dimensions ($\le 4096\times 4096\text{ px}$) and maximum payload size ($20\text{ MB}$) to prevent DoS/memory exhaustion.
2. **Grayscale & Variance Check**:
   - Converts image to grayscale.
   - Computes standard deviation $\sigma_{\text{bg}}$. If $\sigma_{\text{bg}} < 1.0$, the image is flat/solid, returning 0 stones immediately.
3. **Edge-Preserving Bilateral Smoothing**:
   - Applies bilateral filter ($d=7, \sigma_{\text{color}}=50, \sigma_{\text{space}}=50$) to suppress high-frequency metal grain and surface scratches while preserving sharp gemstone facet edges.
4. **Adaptive Polarity Otsu Binarization**:
   - Computes Otsu's optimal global threshold.
   - If foreground pixel ratio exceeds $80\%$ (indicating dark stones on light paper/background), the binary mask is automatically inverted.
5. **Noise Pre-Filtering & Morphological Opening**:
   - Tallies noise contours with $\text{area} < 12\text{ px}$.
   - Applies morphological opening with a $3\times 3$ elliptical structuring element to cleanly disconnect lightly touching items and eliminate isolated pixel artifacts.
6. **Geometric Contour Filtering**:
   Contours are evaluated against strict physical criteria:
   - **Area ($A$)**: $\max(12\text{ px}, 0.00003 \cdot H \cdot W) \le A \le 0.15 \cdot H \cdot W$ (rejects sub-pixel noise and massive background regions).
   - **Circularity (Isoperimetric Quotient)**: $C = \frac{4\pi A}{P^2} \in [0.35, 1.35]$ (round brilliant, oval, and cushion cuts; rejects linear scratches or chains).
   - **Aspect Ratio**: $\min(w, h) / \max(w, h) \ge 0.35$ (rejects elongated slivers).
   - **Solidity**: $A / A_{\text{convex hull}} \ge 0.50$ (rejects concave, hollow, or branched shapes).
7. **Spatial Non-Maximum Suppression (Deduplication)**:
   - Gemstones often produce multiple facet reflection peaks or concentric highlights.
   - Bounding boxes and centroids are checked for overlap ($\text{IoU} > 0.30$ or centroid distance $< 0.80 \times r_{\text{component}}$).
   - Keeps the detection with the higher circularity and suppresses duplicates.
   - Sorts final components deterministically (top-to-bottom, left-to-right).

---

## 3. Designing for Uncertainty

### 3.1 Visible Count vs. Total Design Count
A critical design requirement is acknowledging the physical limitations of 2D photography:
- **Visible Stone Count**: The number of gemstone facets facing the camera on the visible hemisphere of the item.
- **Total Design Count**: The actual number of stones required to manufacture the piece.

> [!WARNING]
> **Occlusion Warning**: A 2D photograph cannot capture stones on the underside of a ring band, the reverse of a double-sided pendant, or stones wrapped around the back of a necklace or eternity bracelet. The visible count must be treated as a lower bound or baseline, never a definitive manufacturing count.

### 3.2 Heuristic Confidence Rules
Rather than inventing an uncalibrated probability, the system assigns explicit, rule-based confidence levels:

| Confidence | Conditions | Reasoning |
| :--- | :--- | :--- |
| **`HIGH`** | $1 \le N \le 12$ and mean circularity $\ge 0.70$. | Modest count, clear object separation, high circularity. |
| **`MEDIUM`**| $13 \le N \le 35$ or mean circularity $\in [0.50, 0.70)$. | Cluster/halo setting where partial facet overlap or mild occlusion is expected. |
| **`LOW`** | $N = 0$, $N > 35$, mean circularity $< 0.50$, or extreme aspect ratio ($> 4:1$). | Ambiguous, dense clustering, flat image, or extreme aspect ratio; classical vision is uncertain. |

### 3.3 Human Review Recommendations
`requires_human_review: bool` is set to `True` when:
- 0 components are detected (distinguishes plain metal bands from failed contrast/lighting).
- Heavy component clustering ($N > 35$).
- Duplicate detection suppression was unusually high ($> 3$ duplicates).
- Extreme image aspect ratios ($> 4:1$).

---

## 4. Limitations & Real-World Caveats

> [!IMPORTANT]
> **Synthetic Accuracy vs. Real-World Jewelry Performance**:
> The unit test suite validates synthetic circular objects with known coordinates and controlled contrast. **Synthetic test accuracy does NOT represent real-world jewelry photography accuracy.**

Real-world jewelry photographs introduce severe challenges that classical computer vision cannot fully resolve without deep learning or CAD data:
1. **Specular Flares & Reflections**: High-polish metal (platinum, yellow gold) creates specular hot-spots that mimic small diamonds.
2. **Transparent / Colorless Gemstones**: High-clarity diamonds transmit background color and reflect ambient studio lighting, reducing edge contrast against light settings.
3. **Pave and Micro-Prong Clustering**: Micro-pave stones set without gaps merge into continuous bands of brightness, making discrete contour segmentation unreliable.
4. **Lighting Angle Variations**: Backlit or side-lit stones cast asymmetric shadows that skew contour circularity and centroid positions.

**Role in the Pipeline**:
This classical CV baseline provides a rapid, zero-cost initial stone count estimate to compare against LLM-extracted brief numbers, flagging discrepancies for human appraisal.
