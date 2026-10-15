# Deterministic Jewelry Inventory Matcher Design

## 1. Overview

The deterministic inventory matcher is a core component of the **Brief to Priced Estimate** system. While an LLM is utilized in earlier stages to extract product specifications from unstructured sales briefs and OpenCV is used to estimate quantities from reference imagery, catalog retrieval and pricing must remain strictly deterministic.

This module guarantees:
- **Zero hallucination**: Prices and SKUs are strictly loaded from verified inventory data.
- **Repeatability**: Identical queries and catalogs yield identical rankings every time.
- **Explainability**: Every candidate score is decomposed into explicit attribute-level sub-scores.

---

## 2. Weighted Scoring Formula

Candidate matching uses a weighted linear combination of normalized attribute similarities:

$$\text{Overall Score} = \sum_{a \in \mathcal{A}_{\text{active}}} w'_a \cdot S_a(v_a^{\text{cand}}, v_a^{\text{spec}})$$

### Base Weights
The default weights represent business priorities for jewelry purchasing:
- **Product Type ($w_{\text{type}}$)**: **35%** (0.35)
- **Material ($w_{\text{mat}}$)**: **30%** (0.30)
- **Dimensions ($w_{\text{dim}}$)**: **20%** (0.20)
- **Stone Count ($w_{\text{stones}}$)**: **15%** (0.15)

$$\sum_{a \in \mathcal{A}} w_a = 0.35 + 0.30 + 0.20 + 0.15 = 1.00$$

All final overall scores are normalized to the closed interval $[0.0, 1.0]$.

### Heuristic Nature of the Match Score
The score calculated by this module is an **engineering similarity heuristic, NOT a probability or Bayesian likelihood**.
- It does **not** signify "probability of sale" or "probability of customer acceptance".
- It represents a bounded empirical closeness score based on calibrated distance kernels.
- Scores should not be multiplied, calibrated as probabilities, or interpreted as likelihood functions.

---

## 3. Attribute Similarity Functions

### 3.1 Categorical Attributes (`product_type`, `material`)
Categorical attributes use an exact-match indicator function (case-insensitive and whitespace-stripped):

$$S_{\text{cat}}(v_{\text{cand}}, v_{\text{spec}}) = \begin{cases} 1.0 & \text{if } \text{clean}(v_{\text{cand}}) = \text{clean}(v_{\text{spec}}) \\ 0.0 & \text{otherwise} \end{cases}$$

### 3.2 Dimensions (`dimensions_mm`)
Physical dimensions use an exponential decay similarity function that adapts to the scale of the target dimension:

$$S_{\text{dim}}(d_{\text{cand}}, d_{\text{spec}}) = \exp\left(-\frac{|d_{\text{cand}} - d_{\text{spec}}|}{\sigma_{\text{dim}}}\right)$$

where:
$$\sigma_{\text{dim}} = \max(0.15 \times d_{\text{spec}}, 1.5\text{ mm})$$

**Properties:**
- Exact match ($d_{\text{cand}} = d_{\text{spec}}$) yields strictly $1.0$.
- Smooth, monotonic decrease as difference $|d_{\text{cand}} - d_{\text{spec}}|$ grows.
- Scale-adaptive: A $2\text{ mm}$ discrepancy on a $17\text{ mm}$ ring is penalized appropriately, whereas a $2\text{ mm}$ discrepancy on a $450\text{ mm}$ necklace is negligible.

### 3.3 Stone Count (`stone_count`)
Stone count uses an exponential decay similarity function:

$$S_{\text{stones}}(s_{\text{cand}}, s_{\text{spec}}) = \exp\left(-\frac{|s_{\text{cand}} - s_{\text{spec}}|}{\sigma_{\text{stones}}}\right)$$

where:
$$\sigma_{\text{stones}} = \max(0.25 \times s_{\text{spec}}, 3.0)$$

**Properties:**
- Exact count match yields strictly $1.0$.
- For solitary pieces ($s_{\text{spec}} \in \{0, 1\}$), a baseline tolerance $\sigma = 3.0$ prevents abrupt cliffs while penalizing multi-stone mismatches.
- For high-count pave or cluster settings ($s_{\text{spec}} \ge 20$), tolerance scales proportionally with count.

---

## 4. Mandatory Compatibility Rules

Weighted scoring alone can hide critical mismatches (for example, high dimension similarity compensating for an entirely wrong category). To prevent this, two mandatory compatibility rules are enforced as **hard filters** before scoring:

1. **Product Type Gate**:
   - If `product_type` is specified in the sales brief, any candidate with a different product type is **strictly excluded**.
   - *Rationale*: Comparing dimensions across disparate product types (e.g. comparing a 17mm ring inner diameter to a 17mm pendant drop or a 450mm necklace) is physically meaningless.
2. **Material Gate**:
   - If `material` is specified in the sales brief, any candidate with a different material is **strictly excluded**.
   - *Rationale*: A customer requesting platinum cannot be offered silver simply because the dimensions and stone counts match.

Candidates failing any active compatibility rule are eliminated prior to ranking.

---

## 5. Missing-Value Handling Policy

Handling missing data requires distinguishing **missing query specifications** (unknown customer intent) from **missing inventory attributes** (incomplete catalog records).

### 5.1 Missing Query Specifications (Customer Brief)
Briefs frequently have incomplete specifications (e.g., stone count or dimension not yet determined). Rather than silently treating missing attributes as a 100% match, the system handles missing attributes explicitly:

1. **Active Weight Renormalization**:
   If an attribute $a \notin \mathcal{A}_{\text{active}}$ is unspecified (or `None`), it is excluded from the scoring calculation, and the weights of remaining active attributes are renormalized to sum to $1.0$:
   $$w'_a = \frac{w_a}{\sum_{j \in \mathcal{A}_{\text{active}}} w_j}$$
2. **Explicit Attribute Audit**:
   In candidate output, unspecified attributes are assigned `None` in `attribute_scores`, ensuring callers know the attribute was unverified rather than perfectly matched.
3. **Empty Specifications Guard**:
   If all supported attributes are omitted or null, the matcher immediately returns a safe no-match response (`is_reliable_match = False`, `matches = []`).

### 5.2 Missing Inventory Attributes (Catalog Records)
When an inventory record lacks data for a customer-requested attribute:
1. **Mandatory Categorical Attributes (`product_type`, `material`)**:
   - If an inventory record has `None`, empty string, or missing value for a requested categorical attribute, it **fails mandatory compatibility** and is excluded. A record of unknown category or material cannot be assumed compatible.
2. **Numeric Attributes (`dimensions_mm`, `stone_count`)**:
   - An unknown inventory dimension or stone count is treated as an **unfulfilled requirement** and receives an attribute score of `0.0`.
   - **Distinction from Known Mismatch**: A missing dimension is explicitly reported in candidate explanations as `"(target 18.0mm, inventory dimension unknown)"` rather than displaying a misleading `record 0.0mm`. This distinguishes missing metadata from a measured dimension of zero while ensuring incomplete records cannot outrank verified near-matches.

---

## 6. Acceptance Threshold & Limitations

A match is designated reliable (`is_reliable_match = True`) only if:
1. At least one candidate passes all mandatory compatibility gates.
2. The top candidate's overall score satisfies:
   $$\text{Overall Score}_{\text{top}} \ge \tau_{\text{acceptance}}$$
   *(Default: $\tau = 0.70$)*

If the top candidate scores below $\tau$, the system flags `is_reliable_match = False` and provides the rejection reason, while still returning candidate details for human inspection.

### Documented Limitations
- **Renormalization Sensitivity**: If only `product_type` and `material` are requested, an exact match on those two attributes yields an overall score of $1.0$, surpassing the $0.70$ threshold even though dimensions and stone counts were unverified. Downstream systems should inspect `attribute_scores` to confirm completeness.
- **Hard Scalar Boundary**: A rigid threshold (e.g., $0.70$) treats $0.699$ as a rejection and $0.701$ as an acceptance. It does not provide confidence intervals or measure the gap between 1st and 2nd rank.
- **Cross-category Dimension Scales**: While $\sigma_{\text{dim}}$ scales with target size, relative tolerances on intricate designs may require domain-specific adjustments in production.

---

## 7. Determinism & Tie-Breaking

To guarantee reproducibility:
- Candidates are sorted primarily by `overall_score` descending (rounded to 6 decimal places to prevent floating-point platform noise).
- Tied scores are broken deterministically by `sku` in ascending lexicographical order.
- The inventory price (`price_inr`) is copied directly from the catalog file; it is never computed, interpolated, or altered.

---

## 8. Why Deterministic Matching Over LLM Catalog Selection

| Criterion | Deterministic Matcher | LLM Inventory Selection |
| :--- | :--- | :--- |
| **Hallucination Risk** | **Zero** — Prices and SKUs come directly from verified records. | **High** — LLMs can invent SKUs, swap digits, or alter prices. |
| **Reproducibility** | **100% deterministic** — Identical inputs yield identical outputs. | **Non-deterministic** — Sampling temperature and seed drift can alter rankings. |
| **Auditability** | Complete mathematical breakdown per attribute. | Opaque latent reasoning; explanations can be rationalized post-hoc. |
| **Latency & Cost** | **< 1 ms execution**, zero API token costs, runs locally in memory. | 500 ms – 3000 ms per call, consumes token budget, requires network connectivity. |
| **Constraint Enforcement**| Hard constraints (material, category) are guaranteed 100% inviolable. | Soft prompt compliance; LLMs can still recommend invalid cross-category items. |

**Architectural Role**:
The LLM should perform what it excels at (translating unstructured human sales prose into structured parameters), while the deterministic matcher performs what code excels at (catalog filtering, constraint verification, mathematical distance scoring, and pricing).

---

## 9. Input Validation & Defensive Boundaries

The matcher strictly validates all arguments and input specifications before execution:

- **`top_k`**: Must be a positive integer ($\ge 1$). Booleans, non-integers, and values $\le 0$ raise `ValueError`.
- **`acceptance_threshold`**: Must be a finite float/int in $[0.0, 1.0]$. Values outside this range, non-finite values (`nan`, `inf`, `-inf`), and booleans raise `ValueError`.
- **`custom_weights`**: Must be a dictionary of non-negative finite numeric weights whose sum is strictly positive ($> 0$).
- **`dimensions_mm`**: Must be a positive finite number ($> 0$). Zero, negative numbers, non-numeric strings, and non-finite values raise `ValueError`.
- **`stone_count`**: Must be a non-negative integer ($\ge 0$). Negative integers, floats, booleans, and non-finite values raise `ValueError`.

