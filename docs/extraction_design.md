# LLM-Based Structured Extraction Design

## 1. Overview

The **Structured Extraction Module** bridges unstructured, natural-language customer sales briefs and the downstream deterministic inventory matcher.

Instead of directly asking an LLM to recommend inventory products or invent prices, the LLM is restricted to its core competency: **extracting explicit, structured physical attributes from human language**.

```text
┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Customer Sales Brief │ ───> │ LLM Structured Extractor│ ───> │ Deterministic Matcher   │
│  "18k gold ring..."  │      │  (Pydantic Validation)  │      │  (Inventory Ground Truth)│
└──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 2. Environment Variables & Configuration

All provider credentials and runtime parameters are read from environment variables. **No API keys are hardcoded or committed.**

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | `string` | `""` (Required for live calls) | Authentication token for OpenRouter API. |
| `OPENROUTER_MODEL` | `string` | `meta-llama/llama-3.3-70b-instruct` | Model identifier (supports any OpenRouter model). |
| `OPENROUTER_BASE_URL`| `string` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `LLM_TIMEOUT_SECONDS`| `float` | `15.0` | Network request timeout in seconds. |
| `LLM_MAX_RETRIES` | `integer`| `2` | Maximum retry attempts on malformed output. |

### Setup Instructions

1. **Set Environment Variables**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key"
   export OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct"
   ```

2. **Run Unit Tests (Offline / No Key Required)**:
   ```bash
   python3 -m pytest -v tests/test_extraction.py
   ```

3. **Programmatic Usage**:
   ```python
   from app.extraction import extract_specifications

   brief = "Looking for an 18k yellow gold band ring, size 17.5mm diameter, plain without stones."
   result = extract_specifications(brief)

   if result.status == "success":
       print(result.specifications.to_matcher_specifications())
   else:
       print("Extraction needs review:", result.review_reasons)
   ```

---

## 3. Provider Architecture & Decoupling

The module separates model communication from extraction and validation logic using Python's `Protocol`:

```python
class LLMProvider(Protocol):
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        ...
```

- **`OpenRouterProvider`**: Default implementation sending requests to OpenRouter.
- **Pluggability**: Alternate providers (e.g., Anthropic Claude, Google Gemini, OpenAI, or local vLLM/Ollama) can be introduced by simply implementing `LLMProvider` without modifying parsing, validation, or confidence scoring.
- **Hermetic Testing**: Unit tests supply a `MockLLMProvider`, ensuring fast, repeatable, and completely cost-free test runs.

---

## 4. Pydantic Schemas & Data Integrity

Defined in `app/schemas.py`:

- **`ProductSpecification`**:
  - `product_type`: `Optional[Literal["ring", "pendant", "necklace", "bracelet", "earrings"]]`
  - `material`: `Optional[Literal["gold", "silver", "platinum"]]`
  - `dimensions_mm`: `Optional[float]` (validated: positive, finite, $\le 2000.0\text{ mm}$)
  - `stone_count`: `Optional[int]` (validated: non-negative whole integer, $\le 1000$)
  - `field_confidences`: Mapping of attribute name to `FieldConfidence`.
- **Representing Unknowns**:
  - Omitted or unmentioned attributes are explicitly stored as `None`.
  - String tokens like `"none"`, `"null"`, `"unknown"`, and `"unspecified"` are converted to `None` by field validators.
  - The model **never fabricates or guesses default values** for missing fields.

---

## 5. Bounded Retry Policy

Large Language Models occasionally produce syntactically invalid JSON or unmapped enum values. The system implements an automated, bounded retry loop:

1. **Attempt 0**: Initial extraction request with schema instructions and zero temperature.
2. **Failure Detection**: Catches `json.JSONDecodeError`, `ValueError`, and Pydantic `ValidationError`.
3. **Targeted Correction Prompt**:
   ```text
   Your previous response had a validation/formatting error: {error_details}
   Please output strictly valid JSON conforming to the schema for this brief:
   """{brief}"""
   ```
4. **Retry Bound**: Controlled by `max_retries` (default: 2 retries, total 3 calls maximum).
5. **Safe Failure Fallback**: If retries are exhausted or a provider network error occurs:
   - Returns `ExtractionResult` with `status = ExtractionStatus.FAILED`.
   - Populates `requires_human_review = True`.
   - Leaves specifications safe and empty (`None` for all attributes).
   - Records the exact diagnostic error message in `error_message`.
   - **Never raises uncaught exceptions or crashes the service.**

---

## 6. Deterministic Confidence Calibration

LLMs are notoriously prone to uncalibrated overconfidence (e.g., asserting `"confidence": "high"` even when speculating). To protect catalog matching, **confidence scoring is performed deterministically using explicit evidence rules**:

| Attribute | High Confidence Criteria | Medium Confidence Criteria | Low Confidence Criteria |
| :--- | :--- | :--- | :--- |
| **`product_type`** | Verbatim category keyword in brief (`ring`, `pendant`, `necklace`, `bracelet`, `earrings`, `studs`, `hoops`, etc.). | Category inferred from descriptive phrasing without exact keyword. | Attribute omitted or unrecognized in brief. |
| **`material`** | Explicit material or hallmark keyword (`gold`, `18k`, `white gold`, `silver`, `925`, `platinum`, `pt950`). | Material inferred from color or general description. | Material omitted or unspecified. |
| **`dimensions_mm`** | Explicit numerical measurement accompanied by unit (`mm`, `cm`, `inch`, `diameter`, `length`). | Numerical value present, but unit context is implicit. | Value converted, derived, or not present. |
| **`stone_count`** | Explicit count digit or recognized term (`solitaire` $\to 1$, `plain band` $\to 0$). | Count inferred from descriptive context. | Stone count omitted or unspecified. |

---

## 7. Human Review Flags

Every `ExtractionResult` computes `requires_human_review: bool` and a list of `review_reasons`:
- Flagged if `status != ExtractionStatus.SUCCESS`.
- Flagged if either mandatory field (`product_type` or `material`) is `None` or has `LOW` confidence.
- Flagged if one or more retry attempts were necessary.
- Flagged if the provider returned an error or timeout.

---

## 8. Limitations & Edge Cases

1. **Multi-Product Briefs**: If a brief requests multiple distinct jewelry items (e.g. "a matching necklace and earrings set"), the current extractor extracts a single dominant product specification. Multi-item segmentation should precede extraction in future iterations.
2. **Unit Conversions**: Briefs specifying non-metric units (e.g. inches or regional ring gauge sizes) rely on LLM mathematical conversion into millimeters. Low or medium confidence is assigned when metric units are not explicit.
3. **Implicit Stone Counts**: Phrasings like "pavé encrusted" imply many small stones without specifying an exact integer; in such cases, `stone_count` is set to `None` with `LOW` confidence to prompt computer vision verification (Step 4).
