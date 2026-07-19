# Tổng kết Pipeline & Kết quả — Health-Trajectory (Apple Health, n=1)

> Tài liệu hệ thống hoá: **pipeline làm gì → cho ra kết quả gì → rút ra luận điểm gì.**
> Cập nhật 2026-07-16. Subject: 1 người, Apple Watch, 11/2017–09/2022 (2.71M record).

---

## 0. Bối cảnh & định hướng

Dự án xây một pipeline phát hiện & diễn giải bất thường (anomaly) trên dữ liệu nhịp
tim Apple Health. Định hướng của giáo sư **Asara** (3 directive):

1. **Context là meta-information ngoài** — được định nghĩa & nạp vào; pipeline giữ generic.
2. **LLM dựng "global baseline of normality"** cho cohort mà context xác định.
3. **Dò deviation + diễn giải** ý nghĩa lifestyle/health.

Luồng dữ liệu:

```
export.xml → [File 1] hr_raw / hr_features / workouts
          → [File 2] behavioral_episodes + edge_table (đồ thị BKG) + weather_hourly
          → [File 3] standard_definitions (baseline) + node/edge anomaly scores
          → [File 4] audit_metrics (đánh giá vs ground truth)
   [File 3b] verify cohort (BIDSleep) — chưa chạy
```

---

## 1. Pipeline từng bước

### File 1 — `01_ingestion_signal_abstraction.ipynb`
**Việc:** đọc `export.xml` thô → time-series HR sạch, window 15 phút.
1. Làm sạch XML (bỏ DOCTYPE, khử attribute trùng).
2. Stream 1 lượt qua 2.71M record → HR (356.862), HRV SDNN (4.885), Resting HR (1.759), Workouts (289).
3. Timezone UTC → `Australia/Adelaide` (chuẩn DST).
4. Resample window 15 phút: `avg/max/min_hr`, `n_samples` (trọng số chất lượng), merge HRV.
5. Chẩn đoán coverage (đủ dày? có đêm 0–5h?).
- **Output:** `hr_raw.parquet`, `hr_features.parquet` (104.565 window), `workouts.parquet`.

### File 2 — `02_context_semantic.ipynb`
**Việc:** gắn ngữ cảnh → "behavioral episodes" + dựng đồ thị BKG.
1. Merge workout vào window (`is_workout`, interval-join HR, `activity`).
2. GPS từ GPX → DBSCAN haversine gom cụm vị trí → `location_type`.
3. Weather: fetch Open-Meteo (temp+humidity) → `weather_hourly.parquet`.
4. LLM semantic normalization: map context tự do → controlled vocab (LLM **không** sinh anomaly).
5. Dựng BKG — 3 loại cạnh: **temporal** (liên tiếp ≤30′), **similarity** (kNN k=8), **context** (cùng location+activity ±2h).
- **Output:** `behavioral_episodes.parquet`, `edge_table.parquet`, `node_table`, `node_features.npy`, `weather_hourly.parquet`.

### File 3 — `03_baseline_gnn_anomaly.ipynb`
**Việc:** định nghĩa "normal" + dò anomaly cấu trúc. Hai phần:

**Part A — Context baseline (approach a của Asara):**
1. Load episodes + `attach_weather` (vá join) + dựng `SubjectContext` (tuổi 40→`40_49` nhập tay; fitness/climate predict từ data).
2. `establish_baseline` → **LLM (Gemini) sinh khoảng normal theo cohort**.
3. `detect_against_baseline`: chọn sub-range theo activity+giờ → **điều biến band theo nhiệt** → tính `deviation` + gán **lifestyle** tag.
4. Lưu `standard_definitions.parquet`.

**Part B — GCN-DOMINANT (detector quan hệ):**
5. `build_node_features` (structural + deviation_z + weather; **không** nhãn LLM).
6. Train GCN-DOMINANT trên đồ thị → điểm anomaly node + edge.
- **Output:** `standard_definitions.parquet`, `node_anomaly_scores.parquet`, `edge_anomaly_scores.parquet`.

### File 3b — `03b_cohort_baseline.ipynb` *(PENDING)*
**Việc:** verify band `sleep_hr` của LLM khớp HR đêm cohort thật (BIDSleep 47 người) — rebut "n=1". Chưa chạy (chưa có data).

### File 4 — `04_audit_metrics.ipynb`
**Việc:** đánh giá với ground truth, so 2 view độc lập.
1. Load episodes + std + edge_table.
2. **Inject synthetic anomaly** (hr_spike, phase_shift, impossible-transition) → `y_true`.
3. **2 detector độc lập**: LLM-semantic (deviation vs band) + GCN-DOMINANT (reconstruction).
4. Metrics ROC/PR/F1 vs synthetic truth.
5. Agreement (kappa/MCC) — *agreement ≠ accuracy*.
6. LLM robustness theo field.
- **Output:** `audit_metrics.csv`, `model_disagreements.csv`, `llm_robustness.csv`.

### Module dùng chung
- `context_baseline/context_library.py` — schema context (SubjectContext/EpisodeContext, vocab, registry).
- `context_baseline/context_providers.py` — predict context từ data + `attach_weather` + `build_subject_context`.
- `context_baseline/global_baseline.py` — `establish_baseline`, `detect_against_baseline` (+weather+lifestyle), `translate`.
- `graph_model.py` — `build_node_features` + `DominantLite` (GCN) + `structural_scores` (1 model dùng chung File 3 & 4).

---

## 2. Kết quả

### 2.1 Chân dung subject (từ data)
- **Tải tim mạch thấp:** 62% workout là đi bộ; đỉnh HR workout ~152; chỉ 77 lần >150 bpm trong 5 năm.
- HR: median 86, p90 104, p99.9 137. Resting (p10) ~74.
- **Không suy được tuổi từ HR** — vì HR không bao giờ tiệm cận HRmax → tuổi phải cấp từ ngoài (40).

### 2.2 Baseline LLM sinh ra (cohort 40–49, sedentary)
| Sub-range | Khoảng (bpm) |
|---|---|
| sleep | 45–70 |
| resting | 60–85 |
| light | 90–120 |
| vigorous | 130–165 |
| HR-max | 177 |

**Kiểm chứng cơ chế:** test multi-cohort cho thấy context điều khiển baseline **đúng hướng y học** (athlete trẻ resting 40–60/HRmax 195; older sedentary+HTN HRmax 165; pregnant nâng sàn resting). ⇒ baseline không tuỳ tiện.

### 2.3 Audit — 2 view độc lập (trước → sau enrichment tuổi+weather)
| Detector | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| LLM-semantic (node) | 0.744 → **0.701** | 0.266 → **0.334** ↑ | 0.390 → **0.427** ↑ |
| GCN-DOMINANT (node) | 0.870 → 0.850 | 0.117 → 0.106 | 0.333 → 0.303 |
| GCN-DOMINANT (edge) | 0.671 → **0.732** ↑ | 0.046 → **0.088** ↑ (~2×) | 0.165 → **0.243** ↑ |

*Prevalence anomaly = 0.9% (node) → base rate PR ≈ 0.009. Các detector đang cao 30–40× chance.*

### 2.4 Biện luận GNN vs GCN
- Chỉ có **1 model** = DOMINANT nền GCN (đã gộp `graph_model.py`).
- GCN "kiếm cơm" ở anomaly **impossible-transition** (cạnh sleep→gắng sức, 2 đầu mút đều bình thường độ lớn → baseline mù): synthetic PR .20 vs .02; block real HR cao .57 vs .02.

---

## 3. Luận điểm rút ra

### LĐ1 — Hai detector nhìn hai thứ khác nhau: đó là điểm mạnh
LLM-semantic bắt anomaly **độ lớn** (spike); GCN bắt anomaly **quan hệ** (transition lạ).
Agreement thấp (kappa 0.35) là **đúng và lành mạnh** — hai view bổ sung, không trùng. Nếu
chúng đồng ý hoàn toàn thì một cái là thừa. ⇒ giữ cả hai view độc lập là hợp lý.

### LĐ2 — Context thực sự thay đổi "normal", không phải trang trí
Chỉ nhập tuổi → band siết → toàn cảnh đổi (flag mô tả 0.85% → 15%; deviation_z median
0.36 → 1.02). Weather điều biến band + thêm feature → metrics dịch chuyển. Đây là **bằng
chứng trực tiếp** cho luận điểm trung tâm của Asara: *context định nghĩa normality*.

### LĐ3 — Enrichment không phải "bữa trưa miễn phí"
Tuổi + weather **giúp** LLM-semantic (F1 0.39→0.43) và relational edge (PR ~2×), nhưng
**hơi hại** GCN node (2 feature thêm gây chút nhiễu reconstruction). ⇒ thêm feature ≠ luôn
tốt hơn; cần chọn lọc theo từng view.

### LĐ4 — Subject này là "ca khó" riêng cho GNN
Sức mạnh của GCN (anomaly quan hệ) cần người *có gắng sức thật* để tạo impossible-transition.
Người này ít vận động nên GCN đúng về nguyên lý (synthetic .57 vs .02) nhưng chưa phát huy
hết. ⇒ GCN nên được validate trên subject/dataset vận động mạnh hơn.

### LĐ5 (QUAN TRỌNG NHẤT) — Tension population-vs-individual → cần baseline HYBRID

**Quan sát:** baseline population 40–49 sedentary cho resting 60–85, nhưng resting thật của
subject median ~86 → **15% window bị flag "above"**. Con số 15% này **không phải "15% bất
thường"** mà là *"người này chạy cao một cách hệ thống so với chuẩn 40–49 sedentary"* — tức
đó là tín hiệu **mạn tính** (chronic), không phải sự kiện cấp tính.

**Vì sao không thể bỏ baseline population:**
- Là cái Asara muốn (LLM dựng normality theo cohort).
- Cho **mốc tuyệt đối**: "người này đứng đâu so với chuẩn cohort?" → chính là insight sức khỏe/lifestyle.
- Robust khi thiếu data cá nhân (người mới, ít lịch sử).

**Vì sao không thể chỉ dùng personal baseline (cosinor cũ):**
- Fit cá nhân nhưng **không so được với chuẩn** — mất khả năng đánh giá "họ có khỏe không".
- **Vòng lặp (circular):** lấy "normal của chính họ" làm chuẩn thì một người đã deconditioned
  sẽ *không bao giờ* thấy vấn đề — chính họ là chuẩn.

**⇒ Luận điểm: cần baseline HYBRID — hai mốc, trả lời hai câu hỏi khác nhau:**

| Lớp | Câu hỏi | Dùng để |
|---|---|---|
| **Population** (approach a, đã có) | "HR này có bình thường cho *người 40–49 sedentary* không?" | Đánh giá sức khỏe/lifestyle, so cohort, mạn tính |
| **Personal** (thêm) | "HR này có bình thường cho *chính người này* không?" | Bắt deviation cấp tính (bệnh, stress, mất ngủ) |

Kết hợp cho phép **phân loại anomaly**:
- `population-high AND personal-normal` → **mạn tính / lifestyle** (vd deconditioning — trường hợp subject này).
- `personal-high` (lệch so với chính họ) → **cấp tính / sự kiện** (event).

**Điểm mấu chốt:** 15% flag của population **không phải noise cần loại bỏ** — nó là tín hiệu
mạn tính có thật; lớp personal sẽ lọc ra event cấp tính *trong số đó*. Chính **dữ liệu đang
đòi hỏi** hybrid, chứ không phải ý kiến chủ quan.

### LĐ6 — Đọc metrics trung thực
F1 ~0.3–0.43 là thấp **tuyệt đối** nhưng cao **30–40× so với chance** (anomaly hiếm 0.9%).
Đây là bản chất rare-event detection, không phải model yếu. PR-AUC là thước đúng để đọc, KHÔNG
phải F1 tuyệt đối.

---

## 4. Insight sức khỏe cho subject (non-diagnostic)
Kết hợp "ít gắng sức + nghỉ vẫn cao (86 > chuẩn 60–85)" là **dấu hiệu deconditioning nhẹ** —
không phải chẩn đoán, nhưng đúng kiểu pattern gợi ý *"nên tăng vận động aerobic"*. Đáng chú ý:
pipeline **tự surface** được insight lifestyle này, đúng tinh thần directive 3 của Asara (diễn
giải chứ không chỉ flag).

---

## 5. Hướng tiếp theo
1. **Baseline hybrid** (LĐ5) — thêm lớp personal cạnh population; phân loại chronic vs acute.
2. **Validate GNN** trên subject/dataset vận động mạnh (LĐ4) — nơi impossible-transition có thật.
3. **File 3b (BIDSleep)** — verify band đêm LLM với cohort thật (rebut n=1).
4. **Housekeeping:** patch bug DST ở weather-fetch File 2 (join đã vá qua `attach_weather`).

## 6. Caveat trung thực (không giấu)
- **n=1** — không suy ra kết luận dân số.
- Weather chỉ có **nhiệt độ + độ ẩm**, không có "trời nắng" → "favourable" là proxy thermal-comfort.
- HRV Gemini thiên **clinical 5-phút** ≠ wearable.
- Vai trò LLM ở đây = **knowledge base cung cấp normative prior**, KHÁC vai trò ở nhánh
  enrichment (nơi ML nhẹ thắng LLM ở context-labeling) — hai việc khác nhau, không mâu thuẫn.
