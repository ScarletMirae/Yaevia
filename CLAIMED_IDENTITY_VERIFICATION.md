# Claimed-Identity Verification Mode — Technical Specification & Documentation
**Yaevia — Offline Handwriting Biometric System**
**Protocol Version:** Frozen Baseline H0 (Experiment K1 Alignment)

---

## 1. Overview & Architecture

Yaevia implements two distinct biometric operational modes:

```
                                    +------------------------------+
                                    |    Input Query Image (Q)     |
                                    +--------------+---------------+
                                                   |
                                                   v
                                    +------------------------------+
                                    |   H0 Preprocessing (256x256) |
                                    +--------------+---------------+
                                                   |
                                                   v
                                    +------------------------------+
                                    |   HOG Feature Extraction     |
                                    |     (34,596 Dimensions)      |
                                    +--------------+---------------+
                                                   |
                    +------------------------------+------------------------------+
                    |                                                             |
                    v                                                             v
+---------------------------------------+                     +---------------------------------------+
|    MODE 1: 1-to-1 Verification        |                     |   MODE 2: 1-to-N Identification       |
|    (Claimed-Identity Verification)    |                     |   (Competitive KNN Multi-Class)       |
+---------------------------------------+                     +---------------------------------------+
| Input: Claimed Writer C               |                     | Input: 20-Class Reference Set         |
| Reference: Filter samples of C only   |                     | Reference: All 320 samples (16/writer)|
| Metric: Mean Top-5 Euclidean Distance |                     | Metric: Distance-Weighted KNN (K=5)   |
| Decision: Distance <= 25.1291         |                     | Output: Top-1 Predicted Name & Top-5  |
|           -> VALID / ACCEPT           |                     |         Vote Shares                   |
|           Distance > 25.1291          |                     |                                       |
|           -> TIDAK VALID / REJECT     |                     | Role: Supporting Diagnostic Context   |
+---------------------------------------+                     +---------------------------------------+
```

### Core Architectural Separation
1. **1-to-1 Verification (Primary Decision Mode):**
   - Answers: *"Does this query sample belong to claimed writer $C$?"*
   - Independent of competing classes.
   - Evaluates absolute feature proximity to the claimed writer's reference subspace.
2. **1-to-N Identification (Supporting Diagnostic Mode):**
   - Answers: *"Which of the 20 registered writers has the most similar handwriting?"*
   - Uses distance-weighted voting across all classes.
   - **Crucial Rule:** Identification output does **NOT** override or dictate the verification decision.

---

## 2. Mathematical Definition

### 2.1 Feature Extraction ($H_0$)
- Letterbox Preprocessing: Grayscale, Gaussian Blur, Otsu Thresholding, Morphological Noise Removal, Contour ROI extraction with 256x256 letterbox padding.
- HOG Parameters: `orientations = 9`, `pixels_per_cell = (8, 8)`, `cells_per_block = (2, 2)`, `block_norm = 'L2-Hys'`.
- Dimension: $31 \times 31 \text{ blocks} \times 36 \text{ features/block} = 34,596\text{ features}$.

### 2.2 Verification Score
For query feature $\mathbf{q} \in \mathbb{R}^{34596}$ and claimed writer $C$ with $N_C \ge 5$ reference features $\{\mathbf{r}_1^C, \dots, \mathbf{r}_{N_C}^C\}$:
1. Compute pairwise Euclidean distances:
   $$d_i = \|\mathbf{q} - \mathbf{r}_i^C\|_2 = \sqrt{\sum_{j=1}^{34596} (q_j - r_{i,j}^C)^2}, \quad \forall i \in \{1, \dots, N_C\}$$
2. Sort distances in ascending order: $d_{(1)} \le d_{(2)} \le \dots \le d_{(N_C)}$.
3. Verification Score is defined as the mean of the top $K=5$ closest reference samples:
   $$\text{Score}(\mathbf{q}, C) = \frac{1}{5} \sum_{k=1}^{5} d_{(k)}$$

### 2.3 Decision Rule & Frozen Threshold
$$\text{Decision}(\mathbf{q}, C) = \begin{cases} \text{ACCEPT (VALID)}, & \text{if } \text{Score}(\mathbf{q}, C) \le \theta_{\text{global}} \\ \text{REJECT (TIDAK VALID)}, & \text{if } \text{Score}(\mathbf{q}, C) > \theta_{\text{global}} \end{cases}$$

- **Operating Threshold:** $\theta_{\text{global}} = 25.1291$
- **Threshold Source:** Experiment K1 Estimated Equal Error Rate (EER) operating point.

---

## 3. Database Schema Migration

The `verifications` table in `database.db` was extended with backward-compatible columns:

| Column Name | Type | Description |
|---|---|---|
| `claimed_writer` | `TEXT` | Claimed writer name selected during verification |
| `verification_score` | `REAL` | Mean Top-5 Claimed Euclidean Distance |
| `verification_threshold` | `REAL` | Frozen decision threshold (`25.1291`) |
| `verification_decision` | `TEXT` | Decision outcome: `ACCEPT` or `REJECT` |
| `verification_status_verif` | `TEXT` | Indonesian label: `VALID` or `TIDAK VALID` |
| `verification_method` | `TEXT` | `mean_top5_claimed_writer_euclidean` |
| `top_claimed_distances_json` | `TEXT` | JSON array of the 5 closest Euclidean distances |

---

## 4. API Endpoints

### 4.1 `GET /api/verify/claimed-writers`
Returns list of registered writers for the claimed identity dropdown.

**Response:**
```json
{
  "success": true,
  "total": 20,
  "writers": [
    { "name": "Aditya", "nim": "20051204001" },
    { "name": "Ahmad", "nim": "20051204002" },
    ...
  ]
}
```

### 4.2 `POST /api/verify`
Performs verification against claimed identity and generates 1-to-N supporting diagnostics.

**Request Form Data:**
- `file`: Image file (multipart/form-data)
- `claimed_writer`: Selected writer name (e.g. `"Fahim"`)
- `ground_truth_name` (optional): Ground truth name

**Response Payload Structure:**
```json
{
  "success": true,
  "verification": {
    "claimed_writer": "Fahim",
    "score": 24.1205,
    "raw_score": 24.1205123,
    "threshold": 25.1291,
    "threshold_source": "Experiment K1 estimated EER operating point",
    "decision": "ACCEPT",
    "status": "VALID",
    "method": "mean_top5_claimed_writer_euclidean",
    "k_neighbors": 5,
    "available_reference_count": 16,
    "neighbor_distances": [23.1201, 23.8540, 24.1102, 24.5019, 25.0163],
    "model_version": "20260926_190940"
  },
  "predicted_name": "Fathurrahman",
  "similarity_percent": 68.42,
  "euclidean_distance": 22.9510,
  "top_matches": [ ... ],
  "analysis_time_seconds": 0.084
}
```

---

## 5. UI/UX Hierarchy in `verify.html`

1. **Input Section:**
   - Dropdown selection for Claimed Identity (loaded dynamically from database).
   - Display of student NIM upon selection.
   - Verification button: *"Verifikasi Klaim Identitas"*.
2. **Primary Result Card (Hero Banner):**
   - Prominent badge: **VALID (ACCEPT)** in green / **TIDAK VALID (REJECT)** in red.
   - Claimed Identity and registered NIM.
   - Verification Score vs Threshold ($d \le 25.1291$).
   - Breakdown of the 5 closest Euclidean distances to the claimed writer.
   - Explicitly notes: *Distance Metric (Lower = Closer)*.
3. **Secondary Result Card:**
   - KNN 1-to-N identification winner, vote share %, and top candidate list as comparative context.

---

## 6. Experimental Basis, Constraints & Limitations

1. **Frozen Baseline H0:**
   - Model `knn_model_20260926_190940.joblib` (SHA-256: `c68e2f9ea54bbba04ea7d7560252df77e08ba96e69c94d89c26711516cc437ef`).
   - Zero model retraining or pipeline modification.
2. **Threshold Characterization:**
   - The threshold $\theta = 25.1291$ is strictly characterized as the *Experiment K1 estimated EER operating point* across 400 reference images. It is **not** claimed as an optimal universal threshold.
3. **Known Limitations:**
   - In cross-device or high intra-writer variation scenarios (e.g. Holdout Test B), genuine trials may produce scores $> 25.1291$, resulting in False Rejections.
   - Writer clustering / magnet classes (e.g. Fathurrahman) may win 1-to-N KNN voting, but claimed-identity verification correctly operates independently of competitive voting.
