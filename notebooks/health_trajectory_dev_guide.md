# Development Guide — Health Trajectory Pipeline
### Hướng dẫn triển khai cho Claude Code · giữ cấu trúc 4 file

> Mục tiêu tài liệu: đảm bảo Claude Code implement đúng thiết kế đã chốt,
> không tự ý đổi kiến trúc. **Đọc hết Mục 0 trước khi code bất kỳ file nào.**

---

## 0. Bối cảnh & ràng buộc BẮT BUỘC (đọc trước)

### 0.1 Sự thật về dữ liệu (đã kiểm chứng)
- **Single-subject** — chỉ 1 người. **Không** được đưa ra bất kỳ claim mang tính cohort/quần thể từ chính dữ liệu này.
- ~356k bản ghi HR, ~4.8 năm (Nov 2017 – Sep 2022), 289 workout. Múi giờ `Australia/Adelaide` (+09:30/+10:30, có DST).
- Coverage: mật độ theo tháng đều; **giờ đêm (0–5h) CÓ dữ liệu khi gộp toàn bộ** (~3–4k điểm/giờ) → đáy circadian quan sát được. Nhưng **theo từng ngày thì thưa** (chỉ ~11% ngày phủ ≥18h, ~27% ngày có đêm, chuỗi liền tốt nhất 9 ngày).
- **Hệ quả:** baseline cosinor phải là **pooled theo giờ-trong-ngày** trên toàn bộ dữ liệu. **KHÔNG** làm cosinor theo từng ngày.

### 0.2 Kiến trúc 4 file (ánh xạ với sơ đồ p3/9)
| File | Vai trò | Stage sơ đồ |
|------|---------|-------------|
| **File 1** `01_ingestion_signal_abstraction.ipynb` | Ingestion + Signal Abstraction | Behavioral Data Ingestion |
| **File 2** `02_context_semantic.ipynb` | Context Enrichment **+** Semantic Construction (dựng BKG) | Context Enrichment + Semantic Construction |
| **File 3** `03_baseline_gnn_anomaly.ipynb` | Personal Baseline (Standard Definitions) **+** context-aware GNN | Anomaly Detection / GNN |
| **File 4** `04_audit_metrics.ipynb` | Đối chiếu & đánh giá có ground truth | (đánh giá) |

### 0.3 Guardrails toàn cục — vi phạm là đi lệch thiết kế
1. **LLM KHÔNG phán anomaly.** LLM chỉ làm *ngữ nghĩa* (chuẩn hóa nhãn context, map ontology) ở File 2. Tuyệt đối không đưa nhãn/`confidence_score` của LLM vào GNN làm feature hay target.
2. **Không circularity.** GNN học độc lập trên cấu trúc đồ thị; File 4 mới so LLM-semantic vs GNN-structural.
3. **Không có ROC/F1 nếu không có ground truth.** Ground truth đến từ **synthetic anomaly injection** (File 4). Không bao giờ lấy output model này làm "nhãn thật" cho model kia.
4. **tz-aware xuyên suốt** (`Australia/Adelaide`). Không để naive datetime.
5. **merge_asof luôn có `tolerance`.** **DBSCAN dùng `haversine`**, không Euclid trên lat/lon.
6. **Cluster ID ≠ nhãn ngữ nghĩa.** Việc gán 'home'/'gym'/'work' cần rule trong *Context Definitions* (Mục 0.4).
7. **Baseline = pooled hour-of-day, chỉ trên REST windows, weighted by `n_samples`.** Không per-day cosinor.
8. **Reproducibility:** mọi LLM call `temperature=0` + Pydantic `Literal` (enum), không dùng `str` tự do cho nhãn.

### 0.4 Đầu vào còn thiếu — Claude Code PHẢI hỏi user trước khi code phần liên quan
- **COPE ontology**: danh sách node type / edge type / thuộc tính (cần cho File 2 semantic + File 3 graph schema).
- **Context Definitions**: bộ rule map `cluster → location_label` và taxonomy activity/location (cần cho File 2).
- **Weather API**: mặc định đề xuất Open-Meteo *archive* (miễn phí, không cần key). Xác nhận với user.
- **Optional File 3b (MMASH cohort baseline)**: user có muốn thêm để trả lời phản biện "chỉ đại diện 1 người" của giáo sư không.

Nếu chưa có các input trên, code phần khác trước và để `TODO(user-input)` rõ ràng ở chỗ cần.

---

## File 1 — Ingestion & Signal Abstraction  ✅ ĐÃ XONG
> **KHÔNG viết lại file này.** Đã build & test. Nhiệm vụ của Claude Code là *tiêu thụ đúng output contract* của nó.

**Output contract (các file sau phải dùng đúng tên & schema này):**
- `df_hr_raw`: `[datetime (tz-aware), value (float)]` — HR thô, đã sort.
- `df_hr_features`: `[datetime (tz-aware, đầu cửa sổ 15'), avg_hr, max_hr, min_hr, n_samples (int), hrv_sdnn (float, NaN nếu thiếu)]`.
- `df_workouts`: `[type, start_time (tz), end_time (tz), duration (phút), distance]`.

**Lưu ý dùng lại:** `n_samples` là trọng số chất lượng — bắt buộc mang xuống File 3 để weight baseline.

---

## File 2 — Context Enrichment + Semantic Construction
`02_context_semantic.ipynb`

**Mục tiêu:** biến `df_hr_features` + workouts + location + weather thành **behavioral episodes** có nhãn ngữ nghĩa, rồi dựng **Behavioral Knowledge Graph (BKG)** typed theo COPE.

### Bước 1 — Gộp workout & môi trường vào từng cửa sổ
- HR cho workout: **interval-join** (với mỗi workout, cắt `df_hr_raw` trong `[start_time, end_time]` rồi agg). Nếu dùng `merge_asof` cho ghép thô thì bắt buộc `tolerance=pd.Timedelta('15min')`, `direction='nearest'`.
- Gắn cờ `is_workout` cho mọi cửa sổ 15' trùng khoảng workout (dùng ở File 3 để lọc REST windows).
- Weather: Open-Meteo archive theo giờ; merge theo `datetime.floor('1h')` + toạ độ cluster. Cache lại (đừng gọi API mỗi lần chạy).

### Bước 2 — Location clustering (nếu có GPS)
```python
from sklearn.cluster import DBSCAN
import numpy as np
coords_rad = np.radians(df_gps[['lat', 'lon']].values)
labels = DBSCAN(eps=100/6_371_000, min_samples=5, metric='haversine').fit_predict(coords_rad)
# eps ~100m. -1 = noise. labels là CLUSTER ID, CHƯA phải nhãn ngữ nghĩa.
```
- **GPS chỉ có khi workout** → đa số cửa sổ non-workout sẽ `location_type='unknown'`. Đây là bình thường, không được crash.

### Bước 3 — Context Definitions: map cluster → nhãn (cần rule từ user)
- Áp `CONTEXT_RULES` (heuristic, để config dict). Ví dụ mặc định nếu user chưa cung cấp — `TODO(user-input)`:
  - `home` = cluster nhiều mẫu 00:00–05:00 nhất; `work` = cluster nhiều mẫu 09:00–17:00 ngày thường; `gym` = cluster trùng địa điểm workout; else `unknown`.

### Bước 4 — LLM chuẩn hóa ngữ cảnh (KHÔNG phán anomaly)
```python
from pydantic import BaseModel
from typing import Literal
class EpisodeContext(BaseModel):
    activity: Literal['rest','walk','run','cycle','row','strength','sleep','unknown']
    location_type: Literal['home','work','gym','outdoor','transit','unknown']
# LLM: temperature=0, .with_structured_output(EpisodeContext)
# Vai trò: ánh xạ mô tả context mập mờ về controlled vocab. KHÔNG có trường is_anomaly.
```
Output: `behavioral_episodes: list[dict]` — mỗi episode giữ `timestamp_iso`, `avg_hr`, `max_hr`, `hrv_sdnn`, `n_samples`, `is_workout`, `activity`, `location_type`, `weather_temp`, `weather_humidity`.

### Bước 5 — Semantic Construction: dựng BKG (cần COPE)
- **Nodes** = episodes (gắn `node_type` theo COPE — `TODO(user-input)`).
- **Edges**: (a) *temporal* — episode kề nhau; (b) *similarity* — kNN trong không gian feature chuẩn hóa; (c) *context* — cùng location / cùng activity. Gắn `edge_type`.
- Output: PyG object (`Data`/`HeteroData` tuỳ COPE) + `node_table`, `edge_table` (giữ để File 4 audit).

**Guardrails File 2:** merge_asof có tolerance; DBSCAN haversine; cluster ID ≠ nhãn; LLM chỉ ngữ nghĩa; tz-aware; GPS thiếu → `unknown` chứ không lỗi.

---

## File 3 — Personal Baseline + Context-aware GNN Anomaly Detection
`03_baseline_gnn_anomaly.ipynb`

### Phần A — Standard Definitions: pooled cosinor baseline (ĐÃ CHỐT)
**Spec bắt buộc:**
- **Multi-harmonic**, k = 1,2,3 chu kỳ/ngày.
- **Pooled theo giờ-trong-ngày** trên TOÀN BỘ dữ liệu (KHÔNG per-day).
- **Chỉ fit trên REST windows** — loại mọi cửa sổ `is_workout==True` (nếu không baseline sẽ lẫn circadian với vận động).
- **Weighted least squares**, weight = `n_samples`.
- **Block-bootstrap CI**, block = **theo ngày** (resample nguyên ngày để tôn trọng autocorrelation), refit ~500–1000 lần.

```python
import numpy as np, statsmodels.api as sm
def cosinor_design(t_hours, K=3):
    X = {}
    for k in range(1, K+1):
        X[f'cos{k}'] = np.cos(2*np.pi*k*t_hours/24)
        X[f'sin{k}'] = np.sin(2*np.pi*k*t_hours/24)
    return sm.add_constant(pd.DataFrame(X))

rest = df_hr_features[~df_hr_features['is_workout']].copy()
rest['t_hour'] = rest['datetime'].dt.hour + rest['datetime'].dt.minute/60
X = cosinor_design(rest['t_hour'])
model = sm.WLS(rest['avg_hr'], X, weights=rest['n_samples']).fit()
# expected_hr(giờ) = model.predict(cosinor_design(hour_grid))
# deviation_z = (avg_hr - expected_hr) / robust_scale(residuals)   # robust_scale ~ 1.4826*MAD
# block-bootstrap: lấy mẫu nguyên ngày -> refit -> CI cho đường cong + band
```
**Output Phần A** (cho mỗi node/episode): `expected_hr`, `band_lo`, `band_hi`, `deviation_z`. Đây chính là **Standard Definitions** đưa vào GNN làm node feature.

> *(Tuỳ chọn File 3b — MMASH cohort baseline):* chỉ làm nếu user đồng ý (Mục 0.4). Fit `statsmodels.MixedLM` random effects theo `subject_id` trên MMASH → baseline cohort thật để trả lời giáo sư. Ghi rõ caveat MMASH (24h/1 chu kỳ/người, nam trẻ, Polar chest-strap).

### Phần B — Context-aware GNN (unsupervised, structural)
- **Input:** BKG từ File 2 + node feature gồm: `deviation_z` (Phần A), one-hot `activity`/`location_type`, `avg_hr`, `hrv_sdnn`, `n_samples`, mã hoá giờ (sin/cos). **KHÔNG có nhãn anomaly của LLM.**
- **Model:** reconstruction-based structural anomaly (kiểu **DOMINANT**: tái tạo attribute + cấu trúc; anomaly score = tổng có trọng số của lỗi tái tạo). Có thể dùng `pygod` (DOMINANT/AnomalyDAE) trên PyG cho nhanh.
- **Output cả hai** (đúng sơ đồ "nodes and edges"): `node_anomaly_score` và `edge_anomaly_score` (lỗi tái tạo cạnh / chuyển tiếp bất ngờ).

**Guardrails File 3:** không per-day cosinor; baseline chỉ REST windows + weight n_samples; GNN không nhận nhãn LLM; xuất cả node & edge score.

---

## File 4 — Audit & Metrics
`04_audit_metrics.ipynb`

**Mục tiêu:** đánh giá có ground truth thật, so hai view độc lập.

### Bước 1 — Synthetic anomaly injection (tạo ground truth)
Tiêm anomaly có nhãn biết trước vào một bản sao episodes/graph:
- **HR spike:** đẩy `avg_hr` một số cửa sổ REST lên cao bất thường.
- **Circadian phase-shift:** dịch pha một block ngày (mô phỏng lệch nhịp).
- **Impossible transition:** chèn cạnh/chuỗi vi phạm adjacency điển hình (vd HR cường độ cao ngay sau 'sleep' không có ramp).
Giữ vector nhãn `y_true` cho từng loại.

### Bước 2 — Hai detector ĐỘC LẬP, chấm trên cùng ground truth
- **LLM-semantic detector:** chấm mỗi episode so với context + ontology + Standard Definitions (Pydantic `Literal`, `temperature=0`) → `is_anomaly`.
- **GNN-structural detector:** từ File 3.
- Tính **ROC/PR/F1** cho *cả hai* so với `y_true` synthetic.

### Bước 3 — Đo đồng thuận & bất đồng
- `cohen_kappa_score` + **MCC** + báo cáo **prevalence** (anomaly hiếm làm kappa lệch).
- Xuất `model_disagreements.csv` (dòng LLM≠GNN) — nền để tinh chỉnh định nghĩa.

### Bước 4 — Robustness của LLM (tách theo field)
- Chạy LLM 3–5× trên 100 episode; báo cáo RIÊNG: tỷ lệ lật `is_anomaly`, phương sai `confidence`, entropy `anomaly_type`. Không gộp thành một con số.

**Output File 4:** `model_disagreements.csv`, ROC/PR curves, bảng F1 theo loại anomaly, bảng robustness theo field.

**Guardrails File 4:** ROC/F1 chỉ tính trên synthetic ground truth; LLM & GNN là hai view độc lập; không coi agreement là accuracy.

---

## Phụ lục — Thứ tự triển khai đề xuất
1. File 2 Bước 1–4 (cần Context Definitions từ user) → episodes.
2. File 2 Bước 5 (cần COPE từ user) → BKG.
3. File 3 Phần A (baseline — spec đã đủ, code được ngay) → z-score.
4. File 3 Phần B (GNN) → anomaly scores.
5. File 4 (audit).

**Trước khi bắt đầu File 2/3-graph:** hỏi user 2 input ở Mục 0.4 (COPE, Context Definitions). File 3 Phần A không cần chờ input nào — có thể code trước.
