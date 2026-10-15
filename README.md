# Brief to Priced Estimate

Production-minded multi-modal pipeline and FastAPI service that extracts jewelry product specifications from customer sales briefs, estimates visible stone counts using classical computer vision (OpenCV), and deterministically matches against a verified catalog inventory for ground-truth pricing.

---

## Architecture Overview

```text
                     ┌────────────────────────────────────────┐
                     │          Customer Sales Brief          │
                     │       & Optional Reference Image       │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │           POST /analyze                │
                     │          (FastAPI Service)             │
                     └───────┬────────────────────────┬───────┘
                             │                        │
                             ▼                        ▼
        ┌─────────────────────────────┐    ┌─────────────────────────────┐
        │  LLM Structured Extraction  │    │  Classical Computer Vision  │
        │      (app/extraction.py)    │    │       (app/vision.py)       │
        │ - Product Type & Material   │    │ - 2D Surface Contours       │
        │ - Dimensions & Stone Count  │    │ - Noise & Duplicate Filter  │
        │ - Field-level Confidences   │    │ - Circularity / Solidity    │
        └──────────────┬──────────────┘    └──────────────┬──────────────┘
                       │                                  │
                       │           Cross-Check            │
                       ├──────────────────────────────────┤
                       │  Stone count discrepancy check   │
                       │   triggers human-review flags    │
                       │                                  │
                       ▼                                  │
        ┌─────────────────────────────┐                   │
        │ Deterministic Catalog Match │                   │
        │      (app/matcher.py)       │                   │
        │ - Compatibility Hard Gates  │                   │
        │ - Normalized Decay Kernels  │                   │
        │ - Top-k Deterministic Ties  │                   │
        │ - Ground-Truth Pricing      │                   │
        └──────────────┬──────────────┘                   │
                       │                                  │
                       └─────────────────┬────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │         AnalyzeResponse JSON           │
                     │ - Extracted Spec & Confidences         │
                     │ - Vision Estimate & Diagnostics        │
                     │ - Top-K Matches & Explanations         │
                     │ - Best Match & Catalog Price (INR)     │
                     │ - Review Flags & Human Audit Warning   │
                     └────────────────────────────────────────┘
```

---

## Project Structure

```text
brief-to-priced-estimate/
├── app/
│   ├── __init__.py              # FastAPI app and schema exports
│   ├── main.py                  # FastAPI server with /health and /analyze endpoints
│   ├── extraction.py            # Structured LLM extraction module with retry logic
│   ├── matcher.py               # Deterministic inventory matcher & scoring kernels
│   ├── schemas.py               # Pydantic schemas, enums, and response models
│   └── vision.py                # Classical OpenCV visible stone estimator
├── docs/
│   ├── extraction_design.md     # LLM extraction architecture & schema design
│   ├── matcher_design.md        # Deterministic matcher scoring documentation
│   └── vision_design.md         # Classical CV baseline, filters, and uncertainty limits
├── evaluation/
│   ├── generate_fixtures.py     # Synthetic image fixtures generator
│   ├── images/                  # 12 benchmark image fixtures (synthetic + real photos)
│   ├── manifest.json            # 12-image vision ground-truth dataset manifest
│   ├── results.csv              # Machine-readable evaluation results table
│   ├── results.json             # Machine-readable evaluation results JSON
│   ├── run_evaluation.py        # Executable evaluation harness CLI (supports --vision-only)
│   ├── test_cases.json          # 20 labeled benchmark evaluation cases
│   ├── vision_results.csv       # Dedicated CV benchmark results table
│   └── vision_results.json      # Dedicated CV benchmark machine-readable metrics
├── inventory/
│   ├── products.json            # 40 validated fictional jewelry inventory records
│   └── validate_inventory.py   # Dataset validation script
├── scripts/
│   └── run_evaluation.py        # Runner entry point delegating to evaluation/
├── EVALUATION.md                # Comprehensive evaluation report and failure analysis
├── requirements.txt             # Python dependencies
└── tests/
    ├── __init__.py
    ├── test_api.py              # API endpoint integration tests (12 tests)
    ├── test_evaluation.py       # Evaluation harness unit tests (4 tests)
    ├── test_extraction.py       # Extraction unit tests (12 tests, 100% offline)
    ├── test_matcher.py          # Matcher unit tests (57 tests)
    └── test_vision.py           # Computer vision unit & regression tests (27 tests)
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+ (tested up to Python 3.14)
- Virtual environment (recommended)

### Installation (Local)
```bash
# Clone and enter the repository directory
cd brief-to-priced-estimate

# Create and activate a virtual environment (or conda environment)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Installation & Execution via Docker
```bash
# Build and run using Docker Compose
docker-compose up --build -d

# Or run standalone Docker container
docker build -t auragem-service .
docker run -p 8000:8000 --env-file .env auragem-service
```

---

## Environment Variables Configuration

The application is configured through environment variables:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | string | _None_ | OpenRouter API authentication key (required only for live LLM extraction; unit tests and offline evaluation run 100% offline). |
| `OPENROUTER_MODEL` | string | `meta-llama/llama-3.3-70b-instruct` | Primary model identifier requested via OpenRouter API. |
| `OPENROUTER_FALLBACK_MODEL` | string | `nvidia/nemotron-3.5-lightning:free` | Fallback model identifier triggered if primary fails, times out, or hits payment/rate limits. |
| `OPENROUTER_BASE_URL` | string | `https://openrouter.ai/api/v1` | Base URL for OpenRouter API requests. |
| `LLM_TIMEOUT_SECONDS` | float | `15.0` | Request timeout in seconds. |
| `LLM_MAX_RETRIES` | integer | `2` | Maximum retry attempts on malformed output before safe degradation. |
| `INVENTORY_PATH` | string | _Auto_ | Custom filesystem path to `products.json`. Defaults to `inventory/products.json`. |

> **Security Note**: Never commit API keys or secret credentials to version control. Set them in your shell or use a `.env` file excluded by `.gitignore`.

---

## Starting the Server

Start the FastAPI application with Uvicorn:

```bash
# Run server with hot reload on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, interactive OpenAPI documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Usage & Examples

### Health Check

```bash
curl -s http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "inventory_records": 40,
  "inventory_path": "/home/mohitsharma/Desktop/Task-1/brief-to-priced-estimate/inventory/products.json"
}
```

---

### 1. Analyze Sales Brief (Text Only)

```bash
curl -X POST http://localhost:8000/analyze \
  -F "brief=Looking for a simple 18k gold band ring, 17.3mm inner diameter, plain without stones."
```

**Response:**
```json
{
  "success": true,
  "extracted_spec": {
    "product_type": "ring",
    "material": "gold",
    "dimensions_mm": 17.3,
    "stone_count": 0,
    "field_confidences": {
      "product_type": {
        "confidence": "high",
        "reason": "Explicit product type keyword 'ring' found in brief text.",
        "evidence": "ring"
      },
      "material": {
        "confidence": "high",
        "reason": "Explicit material keyword 'gold' found in brief text.",
        "evidence": "gold"
      },
      "dimensions_mm": {
        "confidence": "high",
        "reason": "Explicit numerical dimension with unit verified in brief text.",
        "evidence": "17.3mm"
      },
      "stone_count": {
        "confidence": "high",
        "reason": "Explicit phrasing indicating plain/stone-free design verified.",
        "evidence": "plain/no stone"
      }
    }
  },
  "vision_estimate": null,
  "inventory_matches": [
    {
      "sku": "JWL-RNG-001",
      "product_type": "ring",
      "material": "gold",
      "dimensions_mm": 17.3,
      "dimensions": 17.3,
      "stone_count": 0,
      "price_inr": 24500.0,
      "price": 24500.0,
      "overall_score": 1.0,
      "attribute_scores": {
        "product_type": 1.0,
        "material": 1.0,
        "dimensions_mm": 1.0,
        "stone_count": 1.0
      },
      "explanation": "Score: 1.0000 [Weights: product_type=0.35, material=0.30, dimensions_mm=0.20, stone_count=0.15] | product_type: 1.00 (requested 'ring'); material: 1.00 (requested 'gold'); dimensions: 1.00 (target 17.3mm, record 17.3mm); stone_count: 1.00 (target 0, record 0)"
    },
    {
      "sku": "JWL-RNG-002",
      "product_type": "ring",
      "material": "gold",
      "dimensions_mm": 18.1,
      "dimensions": 18.1,
      "stone_count": 1,
      "price_inr": 48000.0,
      "price": 48000.0,
      "overall_score": 0.9044,
      "attribute_scores": {
        "product_type": 1.0,
        "material": 1.0,
        "dimensions_mm": 0.7347,
        "stone_count": 0.7165
      },
      "explanation": "Score: 0.9044 [Weights: product_type=0.35, material=0.30, dimensions_mm=0.20, stone_count=0.15] | product_type: 1.00 (requested 'ring'); material: 1.00 (requested 'gold'); dimensions: 0.73 (target 17.3mm, record 18.1mm); stone_count: 0.72 (target 0, record 1)"
    }
  ],
  "best_match": {
    "sku": "JWL-RNG-001",
    "product_type": "ring",
    "material": "gold",
    "dimensions_mm": 17.3,
    "dimensions": 17.3,
    "stone_count": 0,
    "price_inr": 24500.0,
    "price": 24500.0,
    "overall_score": 1.0,
    "attribute_scores": {
      "product_type": 1.0,
      "material": 1.0,
      "dimensions_mm": 1.0,
      "stone_count": 1.0
    },
    "explanation": "Score: 1.0000 [Weights: product_type=0.35, material=0.30, dimensions_mm=0.20, stone_count=0.15] | product_type: 1.00 (requested 'ring'); material: 1.00 (requested 'gold'); dimensions: 1.00 (target 17.3mm, record 17.3mm); stone_count: 1.00 (target 0, record 0)"
  },
  "estimated_price_inr": 24500.0,
  "confidence": "high",
  "requires_human_review": false,
  "review_flags": [],
  "errors": [],
  "pricing_disclaimer": "Estimated price reflects matching catalog items from current inventory. This is an illustrative benchmark, not a custom fabrication quote."
}
```

---

### 2. Analyze Sales Brief with Reference Image

```bash
curl -X POST http://localhost:8000/analyze \
  -F "brief=Customer desires an 18k yellow gold band ring, 18.0mm diameter, with 3 round accent stones." \
  -F "image=@samples/three_stone_ring.png"
```

**Response with Vision and Stone-Count Cross-Checking:**
```json
{
  "success": true,
  "extracted_spec": {
    "product_type": "ring",
    "material": "gold",
    "dimensions_mm": 18.0,
    "stone_count": 3,
    "field_confidences": {
      "product_type": { "confidence": "high", "reason": "Explicit product type keyword 'ring' found in brief text." },
      "material": { "confidence": "high", "reason": "Explicit material keyword 'gold' found in brief text." },
      "dimensions_mm": { "confidence": "high", "reason": "Explicit numerical dimension with unit verified in brief text." },
      "stone_count": { "confidence": "high", "reason": "Explicit count '3' verified in brief text." }
    }
  },
  "vision_estimate": {
    "estimated_visible_stones": 3,
    "method": "classical_morphological_contour_analysis",
    "confidence": "high",
    "confidence_reason": "Detected 3 well-isolated components with high average circularity (0.86).",
    "requires_human_review": false,
    "review_reasons": [],
    "detected_components": [
      { "centroid_x": 60.0, "centroid_y": 125.0, "radius": 11.4, "area": 408.0, "circularity": 0.8591, "solidity": 0.9623 },
      { "centroid_x": 125.0, "centroid_y": 125.0, "radius": 11.4, "area": 408.0, "circularity": 0.8591, "solidity": 0.9623 },
      { "centroid_x": 190.0, "centroid_y": 125.0, "radius": 11.4, "area": 408.0, "circularity": 0.8591, "solidity": 0.9623 }
    ],
    "diagnostics": {
      "image_width": 300,
      "image_height": 300,
      "background_std_dev": 25.27,
      "raw_contours_found": 3,
      "noise_specks_filtered": 0,
      "duplicates_suppressed": 0,
      "mean_circularity": 0.8591,
      "estimated_visible_count": 3
    },
    "limitations_disclaimer": "Visible stone estimation reflects 2D surface components detected by classical filters. Hidden, occluded, pavilion, or underside stones in the 3D design cannot be detected and require human or CAD verification."
  },
  "inventory_matches": [
    {
      "sku": "JWL-RNG-002",
      "product_type": "ring",
      "material": "gold",
      "dimensions_mm": 18.1,
      "dimensions": 18.1,
      "stone_count": 1,
      "price_inr": 48000.0,
      "price": 48000.0,
      "overall_score": 0.9197,
      "explanation": "Score: 0.9197 | product_type: 1.00; material: 1.00; dimensions: 0.96; stone_count: 0.51"
    }
  ],
  "best_match": {
    "sku": "JWL-RNG-002",
    "product_type": "ring",
    "material": "gold",
    "dimensions_mm": 18.1,
    "dimensions": 18.1,
    "stone_count": 1,
    "price_inr": 48000.0,
    "price": 48000.0,
    "overall_score": 0.9197
  },
  "estimated_price_inr": 48000.0,
  "confidence": "high",
  "requires_human_review": false,
  "review_flags": [],
  "errors": [],
  "pricing_disclaimer": "Estimated price reflects matching catalog items from current inventory. This is an illustrative benchmark, not a custom fabrication quote."
}
```

---

## Testing & Validation

### 1. Validate Inventory Dataset
```bash
python3 inventory/validate_inventory.py
```
*Expected: All 8 dataset constraints pass (40 unique SKUs, 5 categories × 8, 3 metals, valid price/dimension/stone ranges).*

### 2. Run Comprehensive Test Suite
```bash
python3 -m pytest -v
```
*Expected: 112 passing tests across all modules (12 API + 4 evaluation + 12 extraction + 57 matcher + 27 vision).*

### 3. Run Pipeline Evaluation & Vision Benchmark
```bash
# Run full deterministic 100% offline pipeline evaluation (20 cases)
python3 evaluation/run_evaluation.py --verbose

# Run dedicated 12-image computer vision benchmark (MAE, accuracy, side-by-side comparison)
python3 evaluation/run_evaluation.py --vision-only

# Alternatively run via scripts entrypoint
python3 scripts/run_evaluation.py
```
*Produces machine-readable outputs in `evaluation/results.json`, `evaluation/results.csv`, `evaluation/vision_results.json`, and `evaluation/vision_results.csv`, plus formatted console benchmark tables. See [EVALUATION.md](EVALUATION.md) for full methodology, metric definitions, and failure analyses.*

### Test Breakdown
- **`tests/test_api.py`** (12 tests):
  - Health check endpoint verification
  - Valid brief without an image
  - Valid brief with a matching image
  - Empty or whitespace brief (returns HTTP 400)
  - Missing required brief parameter (returns HTTP 422)
  - Invalid / corrupt image upload (graceful fallback without crashing text analysis)
  - LLM extraction failure (safe fallback response with review flags)
  - Vision processing failure (graceful fallback preserving catalog pricing)
  - No acceptable inventory match (sets `best_match` to highest candidate, `estimated_price_inr = None`, flags review)
  - Stone-count discrepancy between text and vision produces explicit review flag
  - Response schema validation & strict catalog price provenance
  - Upload size limit enforcement (rejects >20 MB uploads with HTTP 413)
- **`tests/test_evaluation.py`** (4 tests): Dataset completeness (20 cases, 5 categories, 3 metals, negative cases), image fixture decoding, end-to-end execution, and deterministic reproducibility.
- **`tests/test_extraction.py`** (12 tests): Offline mock tests for OpenRouter structured extraction, JSON repair, missing attributes, and range validations.
- **`tests/test_matcher.py`** (57 tests): Strict category/metal gating, monotonic similarity kernels, top-k ordering, deterministic SKU tie-breaking, and boundary validation.
- **`tests/test_vision.py`** (27 tests):
  - In-memory synthetic geometric circles (1, 3, 5, 8 stones)
  - Standard stone fixtures (0, 1, 3, 5, 8, 12 stones)
  - Real 5-diamond gold ring photograph regression (5 stones exact, suppressing 111 false positives)
  - Dense intricate halo pendant segmentation (>15 components detected, eliminating 5-cluster undercount)
  - Specular glare rejection on plain gold bands (0 stones detected)
  - Light studio stud pairs and trilogy rings (exact counts)
  - Touching stone Euclidean distance transform peak separation
  - Optical bloom pavé edge cases triggering review and low confidence
  - Aspect ratio sanity, payload bounds (20MB), dimension limits (16-4096px), and API schema compatibility.

---

## Known Limitations & Design Boundaries

1. **Synthetic Image Verification vs. Real-World Jewelry Photographs**:
   - The computer vision baseline is validated against controlled synthetic images (known geometric circles on contrastive backgrounds).
   - **Real jewelry photographs** exhibit complex faceted reflections, metallic glare, prong/bezel shadows, colored gemstones, and varying camera angles. The classical OpenCV contour baseline does **not** establish high accuracy on arbitrary real-world customer photos.
   - For production on real customer photography, a deep learning object detector (e.g., fine-tuned YOLOv8 or Faster R-CNN on jewelry datasets) is necessary.

2. **2D Surface Limitation**:
   - Classical 2D computer vision can only estimate visible foreground stones on the camera plane.
   - Hidden, occluded, pavilion, side-gallery, or underside stones in 3D settings cannot be detected from a single 2D image and require human or CAD review.

3. **Deterministic Catalog Pricing vs. Custom Fabrication Quotes**:
   - `estimated_price_inr` strictly reflects an **existing catalog item** in inventory that closely matches the customer's requested specifications.
   - It is **not** a custom manufacturing quote. Actual custom jewelry quotes require calculating raw gold/platinum weight, stone carat weight, labor charges, CAD modeling costs, and metal market spot rates.
   - When no catalog item meets the acceptance threshold (`>= 0.70`), `estimated_price_inr` is set to `null` and a human review flag is raised.

4. **Hermetic Extraction**:
   - The extraction module never hallucinates missing fields into facts. If a customer brief leaves dimensions or stone count unspecified, they remain `null` and the matcher renormalizes scoring over the remaining known attributes.

---

## Why Deterministic Matching Should NOT Be Left to the LLM

A core engineering design decision in this service is delegating inventory matching and pricing strictly to a deterministic, mathematical scoring engine (`app/matcher.py`), rather than prompting the LLM with the catalog to pick a match. The critical reasons include:

1. **Strict Financial Provenance & Anti-Hallucination**:
   - LLMs are probabilistic text generators. When asked to look up inventory items, LLMs frequently hallucinate prices, conflate SKUs (e.g. inventing `JWL-RNG-999`), or subtly round dimension and stone count specifications.
   - In commercial e-commerce and retail jewelry valuation, quoting an incorrect or fabricated price creates legal liability, margin loss, or broken customer trust. Ground-truth catalog pricing must directly reference immutable inventory records.
2. **Determinism & Reproducibility**:
   - The exact same sales brief and specifications must always return the exact same ranked inventory candidates and scores across multiple evaluations, regardless of model version updates, temperature jitter, or API provider outages.
3. **Auditable Score Decompositions**:
   - Our scoring engine returns an exact mathematical breakdown for every candidate (e.g. `product_type: 1.0 (weight 0.35); material: 1.0 (weight 0.30); dimensions: 0.94 (weight 0.20); stone_count: 1.0 (weight 0.15)`). Sales reps and auditors can inspect why an item ranked #1 or #3 with 100% explainability.
4. **Latency & Cost Efficiency at Scale**:
   - Injecting an entire catalog of 50,000 SKUs into an LLM context window is computationally infeasible, slow (seconds of latency), and exorbitantly expensive ($0.10+ per lookup). The deterministic mathematical kernel evaluates 40 inventory records in `<1.5ms` locally with zero token cost.

---

## Per-Request Cost, Latency & Audit Logging

Every incoming request to `/analyze` and `/health` is automatically monitored by FastAPI audit middleware and logged with structured metrics:

- **Response Latency**: Measured via high-resolution monotonic clock (`time.perf_counter()`), appended to response headers as `X-Response-Time-Ms`, and logged in stdout (`[AUDIT_LOG] path=/analyze status=200 duration_ms=18.42 client=127.0.0.1`).
- **Token & Cost Estimation**: Structured specification extraction tracks prompt/completion token consumption based on standard character-to-token ratios and logs estimated API cost in USD (`[COST_AUDIT] endpoint=/analyze est_input_tokens=148 est_output_tokens=60 est_cost_usd=$0.000151 match_count=3`).
- **Discrepancy Auditing**: When CV-detected stone counts disagree with customer brief statements, explicit audit entries are generated and appended to `review_flags`.

---

## AWS Production Deployment Architecture (Services & Rationale)

To deploy this multi-modal valuation service in a production AWS enterprise environment:

```text
               ┌───────────────────────┐
               │  Route 53 & CloudFront│
               └───────────┬───────────┘
                           │ HTTPS / WAF
                           ▼
               ┌───────────────────────┐
               │ Application Load      │
               │ Balancer (ALB)        │
               └───────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │ ECS Fargate     │         │ ECS Fargate     │
    │ Container Task  │         │ Container Task  │
    │ (FastAPI + CV)  │         │ (FastAPI + CV)  │
    └────────┬────────┘         └────────┬────────┘
             │                           │
    ┌────────┴────────┬──────────────────┴────────┐
    ▼                 ▼                           ▼
┌──────────────┐ ┌──────────────┐      ┌─────────────────────┐
│ AWS Secrets  │ │ Amazon S3    │      │ Amazon RDS          │
│ Manager      │ │ (Uploaded    │      │ PostgreSQL          │
│ (API Keys)   │ │  Jewelry     │      │ (Catalog Inventory  │
│              │ │  Images)     │      │  & Audit Quotes)    │
└──────────────┘ └──────────────┘      └─────────────────────┘
```

1. **Compute Layer: AWS ECS on AWS Fargate**:
   - *Why*: The application is containerized via our multi-stage `Dockerfile`. ECS Fargate provides serverless container execution with auto-scaling based on CPU/memory utilization or ALB request count. Unlike AWS Lambda, Fargate has no 15-minute execution limit, supports heavy OpenCV runtime dependencies without custom Lambda layers, and eliminates cold-start latency.
2. **Traffic Routing & Security: ALB + AWS WAF**:
   - *Why*: Application Load Balancer distributes traffic across multiple availability zones (AZs), performs SSL termination, and routes health checks to `/health`. AWS WAF protects against DDoS, rate-limits abusive clients, and blocks malformed payloads.
3. **Blob Storage: Amazon S3 (Server-Side Encryption)**:
   - *Why*: Customer jewelry photos and diagnostic detection overlays are archived in S3 buckets with lifecycle rules (e.g. transition to S3 Glacier after 90 days). The API receives pre-signed upload URLs, offloading large multipart image streams from the container memory.
4. **Data Layer: Amazon RDS PostgreSQL (Multi-AZ)**:
   - *Why*: Scales catalog inventory lookups beyond our synthetic 40 records to enterprise catalogs of 500,000+ SKUs using indexed columns and `pgvector` for hybrid semantic and deterministic search.
5. **Secret Management: AWS Secrets Manager / Parameter Store**:
   - *Why*: Automatically rotates and injects `OPENROUTER_API_KEY` into ECS container task definitions as environment variables without persisting credentials on disk or in source control.
6. **Observability: Amazon CloudWatch & AWS X-Ray**:
   - *Why*: Collects structured JSON audit logs (`[AUDIT_LOG]`, `[COST_AUDIT]`), tracks P95/P99 latency dashboards, alarms on elevated 4xx/5xx error rates, and traces cross-modal execution paths.

---

## AI-Tool Usage Disclosure: What was Changed and Rejected

This project was built with assistance from an advanced AI coding assistant (Antigravity AI / DeepMind pair-programming agent) alongside OpenRouter multi-modal APIs. AI tools accelerated initial boilerplate drafting, test fixture scaffolding, and mathematical documentation.

**What was explicitly reviewed, changed, and rejected from AI proposals:**
1. **Rejected Generative LLM Pricing**: Early AI suggestions proposed prompting the LLM with the inventory to directly select an item and quote a price. This was strictly rejected because LLMs hallucinate non-existent SKUs, fabricate retail figures, and cannot guarantee deterministic repeatability across customer inquiries. We insisted on building an immutable, weighted mathematical distance kernel in `app/matcher.py`.
2. **Rejected Naive Global Otsu Inversion in Vision**: An initial AI-generated baseline used global Otsu thresholding on inverted grayscale studio images, which caused metallic gold shanks and shadows to be misclassified as 111 gemstone candidates. We replaced this with an illumination profiler, Canny edge facet density gating ($\ge 25$), HSV yellow-metal saturation rejection ($S < 45$), and Euclidean distance transform peak detection.
3. **Changed Cluster Grouping to Distance Transform Peak Extraction**: When an intricate halo pendant was initially collapsed into 5 merged blobs, AI proposed simple morphological dilation. We rejected dilation (which further merged abutting pavé stones) and implemented Euclidean Distance Transform local peak detection to separate touching stones while suppressing specular metal reflections.
4. **Enforced Rigorous Mathematical Boundaries**: AI-generated scoring functions initially lacked bounds checking. We manually refactored the matcher with strict Pydantic enum validation, monotonic bounded exponential kernels, finite-number guards (`nan`/`inf` protection), and deterministic SKU tie-breaking validated by 112 automated tests.

