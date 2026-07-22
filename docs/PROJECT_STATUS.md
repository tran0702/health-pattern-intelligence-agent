# PROJECT STATUS — Single Source of Truth

> **Đây là tài liệu trạng thái DUY NHẤT của dự án.** Cập nhật **2026-07-22**.
> Repo `tran0702/health-pattern-intelligence-agent`, HEAD `5a725f9`.
> Đọc file này là đủ để tiếp tục làm, không cần lịch sử hội thoại.
>
> Ký hiệu nguồn: ✅ **đã tự chạy đo trên dữ liệu thật** · 📄 từ tài liệu dự án (có thể cũ) ·
> 🔬 từ nguồn ngoài (gián tiếp, chưa tự kiểm chứng) · 💡 quyết định thiết kế.

---

## 1. Bối cảnh

Pipeline phát hiện & diễn giải bất thường nhịp tim từ Apple Watch, **n=1**, 11/2017–09/2022.
Luận điểm trung tâm: **context định nghĩa thế nào là "normal"** — meta-information bên ngoài (tuổi,
thể trạng, bệnh nền, nghề nghiệp, thời tiết, địa điểm) xác định một cohort; LLM dựng khoảng normal
cho cohort đó; rồi dò lệch chuẩn và diễn giải.

**Nguyên tắc kiến trúc bất di bất dịch — 2 tầng:**
```
Data thô ──[CODE: đo, thống kê]──► fingerprint có cấu trúc ──[LLM: suy luận]──► nhãn trong vocab
```
LLM **không** nuốt data thô, **không** phán anomaly. **Code đo → LLM đặt tên.**

---

## 2. Cấu trúc logic

### 2.1 Luồng chính
```
export.xml
 → [File 1] 01_ingestion            → hr_raw · hr_features (window 15') · workouts
 → [File 2] 02_context_semantic     → behavioral_episodes + BKG (node/edge_table) + weather
 → [File 3] 03_baseline_gnn_anomaly
        Part A: context baseline (LLM / physiology fallback) → standard_definitions
        Part B: GCN-DOMINANT (autoencoder)                   → node/edge anomaly scores
 → [File 4] 04_audit_metrics        → tiêm synthetic anomaly → 2 detector độc lập → ROC/PR/F1
   [File 3b] cohort baseline (BIDSleep) — PENDING, chưa có data
```

### 2.2 Bốn tầng code

| Tầng | Thành phần |
|---|---|
| **1. Pipeline** | File 1–4 (+3b) |
| **2. Module chung** | `graph_model.py` (node features + GCN-DOMINANT) · `context_baseline/`: `context_library.py` (vocab, SubjectContext/EpisodeContext) · `context_providers.py` (Individual context từ data) · `global_baseline.py` (LLM baseline + detect + translate) · `global_context.py` (**Global context**: roles + fingerprint + classify) |
| **3. Track cô lập** | `context_vocab/` (Task 1) · `enrichment_experiment/` (LLM-vs-ML trên ExtraSensory) · `location_context/` (DBSCAN + geocode) |
| **4. Agent legacy** | `src/agents/` — đã unify Gemini, tự load `.env` |

**Hạ tầng:** 1 file `.env` ở repo root · **1 provider LLM: Gemini (`google-genai`)** · mọi call
`temperature=0` + cache đĩa + fallback deterministic.

### 2.3 Đối chiếu sơ đồ p3/9 (kiến trúc của giáo sư)

| Tầng sơ đồ | Trạng thái |
|---|---|
| Behavioral Data Ingestion | ✅ |
| Context Enrichment ← *Context Definitions* | ✅ |
| Semantic Construction ← *Ontology* → BKG | ⚠️ BKG có, **ontology (COPE) còn generic/TODO** |
| Context-aware Anomaly Detection (GNN) ← *Standard Definitions* | ✅ (node + edge) |
| Post-hoc Reasoning ← LLM | ⚠️ chỉ có `translate()` deterministic |
| Personal Lifestyle Semantic Construction → Lifestyle KG | ❌ chưa xây |
| Lifestyle Resolution → lifestyle map | ❌ chưa xây |

**Khái niệm chưa mô hình hóa:** Observer Point · stateful/multi-cycle theo epoch (hiện **stateless,
1 lượt**) · anomalous community/ego-net. ⇒ **Code phủ nửa trên sơ đồ.**

---

## 3. Task Asara & trạng thái

**Context profile Asara gợi ý (spine 10 chiều):** weather · geographic location · workout type ·
workout duration · workout location · sleep · heart health · medical conditions (current & past) ·
occupation · demographic attributes

| # | Task | Trạng thái |
|---|---|---|
| **1** | LLM tự sinh context vocabulary (không gõ tay) | ✅ **XONG** — 10 chiều/70 term, **10/10 từ LLM thật**, freeze `vocabulary.json` + code-gen `generated_vocab.py` |
| **2** | Tự phân loại 2 cấp trên dataset **bất kỳ**: **Global** (dataset nói về gì) + **Individual** (người này là ai) | ✅ **XONG** — Individual ✅ (đã gỡ circularity §4①) · Global ✅ (`global_context.py`, §4②) · **bước 6 nối prior ✅** (`global_prior_from_context` → `build_subject_context`, confidence-gated) · (enhancement `occupation`/`sleep` predictor từ data = §4④, không chặn Task 2) |
| **3** | Thử nghiệm mô hình Transformer cho trích xuất ngữ cảnh | ⏸ sau Task 2 |

**Đã có sẵn & chạy được cho Task 2:** `build_subject_context` **tự động** gán `age_band` /
`fitness_level` / `home_climate`; chuỗi `SubjectContext → establish_baseline →
detect_against_baseline` (đúng cái Asara mô tả) **đã hoạt động**. Từ 2026-07-22 nó còn nhận
**global prior** (confidence-gated) từ tầng Global — hoàn tất vòng 2 cấp Asara yêu cầu.

**Task 2 core ✅ đủ.** Enhancement còn để ngỏ (không chặn Task 2): predictor `occupation`/`sleep`
suy từ data (§4④) — hiện là trường user/prior, chưa có bộ dự đoán riêng.

---

## 4. Thiết kế việc tiếp theo

### ① Gỡ circularity cho `fitness_level` — ƯU TIÊN CAO NHẤT

✅ **Vấn đề đã kiểm chứng:** `predict_fitness_level` suy nhãn **từ resting HR** (`_resting_hr` = p10
của non-workout `avg_hr`), nhãn đó **chọn band resting HR**, band đó **phán xét chính resting HR ấy**.

Hệ quả thật (rest median = **85,0** bpm):

| nhãn | band | kết luận |
|---|---|---|
| `sedentary` | 62–92 | bình thường |
| `recreational` ← hệ thống đang gán | 54–76 | **bất thường** |

Cùng dữ liệu → kết luận ngược nhau. Đây là câu hỏi về tính hợp lệ của **đúng cơ chế Asara mô tả**:
*"Once your system automatically tags a user as 'sedentary' versus an 'athlete', your algorithms can
assess whether a spike is a genuine anomaly."* Chữ **automatically** là mấu chốt — tự động nghĩa là
suy từ dữ liệu, mà dữ liệu duy nhất ta có là nhịp tim.

💡 **Thiết kế: TÁCH TÍN HIỆU** — vẫn tự động (giữ đúng ý Asara), nhưng tín hiệu-gán-nhãn ≠
tín hiệu-bị-phán-xét. Thêm vào `context_providers.py`:

```python
def predict_fitness_from_exertion(frames) -> FieldEstimate:
    """Suy fitness từ TÍN HIỆU GẮNG SỨC, không đụng resting HR."""
```
Tín hiệu (đều có sẵn):
1. **Khối lượng workout** — `workouts.parquet`: số buổi/tuần, tổng phút/tuần.
2. **Peak HR gắng sức** — tái dùng `_peak_hr(frames)` (đã có).
3. **HR recovery** — tụt HR 1–3 phút sau `end_time` mỗi workout, tính từ `hr_raw`. *(chỉ số fitness kinh điển, độc lập hoàn toàn với resting HR)*

⚠️ **Ràng buộc:** hàm này **không được** gọi `_resting_hr()`. Giữ hàm cũ nhưng ghi rõ trong docstring
"resting-based — chỉ để BÁO CÁO, KHÔNG feed vào band"; `report()` in **cả hai** để thấy chênh lệch.

**Dự đoán trước:** 289 workout / 4,8 năm ≈ **1,2 buổi/tuần**, peak HR chỉ **152** → bản exertion nhiều
khả năng ra *sedentary/low*. Điều đó **ổn** — cái được là nhãn đến từ gắng sức, không từ đại lượng
đang bị phán xét.

#### ✅ Kiểm chứng data + đánh giá 3 hướng gỡ circularity (review 2026-07-22)

Đã đối chiếu code/data thật trước khi chọn hướng:
- `predict_fitness_from_exertion` **chưa tồn tại**; `predict_fitness_level` vẫn gọi `_resting_hr` (p10)
  → **vòng lặp còn nguyên**, chưa sửa.
- HRV SDNN **có thật: 4.885 record** (ingestion trích cùng HR / Resting HR / Workouts trong 1 lượt).
- 🔒 **KHÔNG có sleep-stage** trong pipeline — ingestion chỉ trích HR, HRV, Resting HR, Workouts,
  **không có `SleepAnalysis`**. Đây là ràng buộc cứng cho mọi hướng cần giấc ngủ.
- Cơ chế **user override đã có sẵn** trong `build_subject_context` (user-value luôn thắng, conf=1) —
  **không cần xây mới**, chỉ cần truyền `user={"fitness_level": ...}`.

| Hướng | Phán quyết | Lý do |
|---|---|---|
| **① HRR** — tụt HR 1–3′ sau `end_time` mỗi workout | ✅ **TRỤ CHÍNH — làm** | Chỉ số fitness kinh điển, độc lập với resting HR tuyệt đối; đúng plan §4①. Ràng buộc: **lọc chỉ workout đủ gắng sức** (peak~152, 62% đi bộ) + **kiểm mật độ `hr_raw` sau `end_time`** (cooldown Apple Watch hay thưa mẫu) trước khi tin HRR |
| **② HRV deep-sleep** | ⚠️ **chỉ tín hiệu PHỤ, không primary** | HRV có (4.885) nhưng **không có sleep-stage** để cô lập "deep sleep"; đêm 00–05h chỉ **16% ngày** có data (trùng đáy circadian); SDNN Apple là sample lẻ ~1′ nền ≠ continuous. Chỉ dùng được nếu **bỏ yêu cầu "deep sleep"**, gate "đêm + HR thấp + đứng yên" → bổ trợ cho HRR |
| **③ latent / UMAP / PCA-50** | ❌ **KHÔNG dùng gỡ circularity → đẩy Task 3** | (a) phá **kiến trúc 2 tầng** + triệt tiêu nhãn ngữ nghĩa Asara cần; (b) chính là nhánh SensorFM **đã GÁC §8**; (c) nếu resting HR là PC trội thì latent vẫn mã hóa tín hiệu bị phán xét → **giấu** circularity chứ không cắt; (d) n=1 + đêm khuyết → PCA-50 daily-features **overfit/bất ổn** |

⚠️ Cảnh báo phạm trù cho alias mapping (xem §4③): **occupation ≠ fitness** — không map `"nhân viên văn
phòng" → sedentary`; office worker vẫn có thể là athlete. Chỉ chuẩn hóa alias **trong cùng một chiều**.

**✅ ĐÃ HIỆN THỰC hướng ① (2026-07-22):** thêm `predict_fitness_from_exertion` + helper `_hrr60` /
`_workout_load` / `_nearest_after` vào `context_providers.py`; `_PREDICTORS["fitness_level"]` trỏ sang nó;
`predict_fitness_level` (resting) hạ xuống **REPORT-ONLY**; `build_subject_context` in **cả hai** nhãn
(`fitness_level` exertion + `fitness_resting_report`). Demo persona nâng cấp có **tail hồi phục** → khôi
phục đúng athlete / recreational / sedentary từ HRR+volume (không dùng resting).

> **Kết quả đo ĐẢO dự đoán "sedentary/low" ở trên:** gate peak≥130 bpm → **34 workout đủ gắng sức**,
> **HRR60 median = 19 bpm (hồi phục TỐT)**; kết hợp volume **1,2 buổi/tuần** → subject thật ra
> **`recreational` (conf 0.80)** — năng lực tim mạch tốt, chỉ **khối lượng tập thấp**, KHÔNG phải
> deconditioned. HRR thô (chưa gate) chỉ 6–7 bpm vì **62% buổi là đi bộ** (peak ~114, không có gì để hồi
> phục). Nhãn này giờ **độc lập hoàn toàn với resting HR** → đúng mục tiêu; resting-report cũng ra
> `recreational` (p10=74 → "average"), còn narrative deconditioning là từ median~85 (§7).

### ② Global Context — ✅ XONG (bước 1–6, 2026-07-22)

**✅ Đã có `notebooks/context_baseline/global_context.py`** (3 stage) + `demo_global_context.py`:
- `infer_column_roles` LLM structured + fallback regex/dtype; **adversarial rename (tên cột tiếng Anh
  khó) → LLM khôi phục đủ 4 vai trò** heart_rate/timestamp/subject_id/workout_type từ dtype+stats.
- `dataset_fingerprint` (thuần code): **inter-subject** (median & IQR resting theo từng người) **+
  intra-subject** (std baseline theo ngày) **+ missingness** (`night_missing_heavy` khi night/day < 0.5).
- `classify_global_context` LLM + rule fallback → `{dataset_domain (free-form + normalize), population_
  descriptor, dominant_activities (chuẩn hóa workout_type vocab), evidence, confidence}`.
- Validate offline PASS: cohort synthetic **athletic/clinical/office khôi phục đúng domain**; subject thật
  → `consumer_wearable` (LLM conf 0.85), cờ night gap bật → **không bịa context giấc ngủ**.
- 🔒 Không gửi dòng data thô cho LLM (chỉ tên cột + dtype + stats + tập nhãn low-cardinality).

**✅ Bước 6 ĐÃ HIỆN THỰC (2026-07-22) — nối `global_prior` xuống individual:**
- **`global_prior_from_context(gc)`** trong `global_context.py`: map `dataset_domain` → prior per-field,
  confidence = confidence của chính bản phân loại dataset. Bảng `_DOMAIN_PRIOR` cố ý **nhỏ + bảo vệ được
  y học**: `athletic_performance` → `fitness_level=trained`+`heart_health=elite_athletic`; `clinical_cardiac`/
  `clinical_cohort` → `heart_health=at_risk`; `general_population` → `heart_health=average_sedentary`.
  `consumer_wearable`/`sleep_study`/domain lạ → **prior rỗng** (không đoán). **KHÔNG** map `health_conditions`
  (bịa chẩn đoán per-subject từ cờ dataset = đúng cái guardrail cấm; cohort lâm sàng vẫn có nhóm chứng).
- **`build_subject_context(..., global_prior=None, min_global_conf=0.5)`**: thứ tự ưu tiên
  **user > tín hiệu cá nhân (conf≥min_conf) > global prior (chỉ `_PRIOR_FIELDS`, chỉ khi dataset-conf≥
  min_global_conf) > unknown**. Prior bị chiết khấu (`_PRIOR_DISCOUNT=0.7`) để không bao giờ giả làm bằng
  chứng cá nhân. Không import chéo track — prior là **dict thuần** (one-way, artifact-style). Đồng thời
  **populate các trường vocab mới** (`heart_health`/`occupation`/`sleep`/`age_years`) vào `SubjectContext`
  (constructor trước đây bỏ sót chúng).
- **Verify (`demo_global_context.py` §5, PASS cả offline lẫn live):** athletic → `heart_health=elite_athletic`
  [prior applied]; cardiac → `at_risk`; office (rule conf **0.45 < 0.50**) → **gated → unknown** (không đoán bừa);
  athletic có workout → `fitness=athlete` [individual, prior không đè]; athletic bỏ workout → `fitness=trained`
  [prior điền]; user override → `heart_health=average_sedentary` [user thắng prior]. **Subject thật
  (`consumer_wearable`) → prior `{}`** → tầng individual không đổi (đúng: 1 wearable đơn không suy được tier).

Ghi lại thiết kế gốc bên dưới.

<details><summary>Thiết kế gốc (đã hiện thực)</summary>

File mới `notebooks/context_baseline/global_context.py`:

1. **`infer_column_roles(df) -> dict`** *(schema-agnostic → đáp yêu cầu "dataset bất kỳ")*
   Đưa LLM **tên cột + dtype + thống kê tóm tắt**; map sang vai trò chuẩn (`heart_rate`, `timestamp`,
   `subject_id`, `workout_type`, `diagnosis`, `age`, `sex`…). Pydantic + `temperature=0` + cache.
   Fallback: regex tên cột (`hr|heart|bpm` → `heart_rate`…).
   🔒 **KHÔNG gửi dòng dữ liệu bệnh nhân thật** — tên cột + dtype + stats là đủ.

2. **`dataset_fingerprint(frames, roles) -> dict`** *(thuần code)*
   `n_subjects`, `n_records`, khoảng thời gian, mật độ mẫu · phân bố HR (median, p10, p90, p99) ·
   tỷ lệ & histogram workout · cờ dataset lâm sàng (có cột `diagnosis`/thuốc).
   ⚠️ **Nhiều người → phải là phân bố GIỮA các subject** (median & IQR resting-HR *theo từng người*),
   **không** percentile gộp — gộp thì không phân biệt được global với individual.

3. **`classify_global_context(fingerprint) -> GlobalContext`**
   `{dataset_domain, population_descriptor, dominant_activities, evidence, confidence}` —
   LLM (cache, temp=0, structured) + **fallback rule-based**. Cấp **prior** cho tầng individual.

**Validate:** cohort synthetic có ground-truth 2 cấp (VĐV / bệnh tim / văn phòng) theo mẫu
`make_persona` trong `demo_context_providers.py` → kiểm tra tự khôi phục đúng; rồi chạy subject thật.
</details>

### ③ Tách `demographic` + regenerate vocab — ✅ PHẦN LỚN ĐÃ XONG (2026-07-22)

**✅ Đã làm:**
- **Tách** `demographic` → **`age_band`** + **`sex`** trong `context_vocab/context_profile.py`; description
  **ghim đúng bộ token canonical** (`under_18…60_plus`, `male/female/unknown`) để khớp `context_library`.
- Regenerate: `python demo_vocab.py` → **LLM thật cho 2 chiều mới** (key .env), cache cho 9 chiều cũ →
  `vocabulary.json` **11 chiều / 71 term, source=llm**, `AgeBandVocab`/`SexVocab` khớp chính xác library.
- **Nối vào `context_library.py`:** import artifact `context_vocab.generated_vocab` (one-way, có fallback
  local nếu artifact thiếu); `SubjectContext` **thêm** `age_years: float|None` + `occupation` + `heart_health`
  + `sleep` (dùng Literal từ vocab); `availability()` giờ 9 trường; `global_baseline` sửa hardcode `/6`→động.
- Demo verify: `demo_vocab.py` PASS (cả llm lẫn `--offline`); `demo_context_providers`/`demo_context_baseline`
  chạy sạch, không regression.

**⏳ Còn lại (đụng pipeline / thuộc task khác — chưa làm):**
- **`age_continuous` → Tanaka:** trường `age_years` đã có trong schema nhưng `global_baseline` **vẫn dùng
  `_AGE_MID`** band→midpoint (line 99); nối `age_years` vào Tanaka là bước sau.
- **Tách `health_conditions` → current/past:** **hoãn** — đổi schema sẽ phá các constructor `SubjectContext`
  hiện có (compare_cohorts, justify_gnn, demo_*). Cần refactor đồng loạt.
- **Chiều `dataset_domain`:** thuộc **Global Context (Task 2)** — làm khi dựng `global_context.py`.
- **Predictor cho `occupation`/`sleep`:** hiện là trường nhận từ user; predictor từ data là Task 2/§4④.

> 💡 **Quyết định có chủ ý:** `context_library` được phép import `context_vocab.generated_vocab` —
> phụ thuộc **một chiều vào artifact đã đóng băng & commit**, không phải coupling runtime giữa hai
> track. Đây chính là mục đích tồn tại của Task 1. Ghi rõ trong docstring.

### ④ `occupation` — chính ví dụ của Asara

Asara nêu *"A sedentary worker on an 8-to-5 schedule"* vs *"A 25-year-old cyclist"* — hiện **không có
predictor nào**. Suy từ **nhịp thời gian**: tương phản ngày thường vs cuối tuần · phân bố giờ có
`location_type == 'work'` (File 2 đã gán 283 episode) · giờ xảy ra workout.

⚠️ Tín hiệu yếu (chỉ HR + GPS thưa) → thiết kế để **confidence thấp → rơi về `unknown`** qua gate
`min_conf`, chấp nhận user override. Thà `unknown` còn hơn đoán bừa.

---

## 5. Guardrail

1. LLM **không phán anomaly**; chỉ cấp reference range + nhãn ngữ nghĩa.
2. **Không circularity** — xem §4①. Giờ là guardrail hạng nhất.
3. ROC/F1 **chỉ** trên ground truth thật (synthetic injection File 4 / persona truth).
4. Mọi LLM call `temperature=0` + Pydantic structured + **cache đĩa** + **fallback deterministic**
   (chạy được không cần API key).
5. Thiếu dữ liệu → `unknown` + nới band, **không bao giờ crash**.
6. **Báo cáo trung thực:** fallback không bao giờ được gắn nhãn kết quả thật.
7. Track cô lập: chỉ ghi `data/<track>/` và `results/<track>/`; không sửa File 1–4.
8. 🔒 Không gửi dòng dữ liệu bệnh nhân thô cho LLM.
9. tz-aware xuyên suốt · DBSCAN dùng **haversine** · `merge_asof` luôn có tolerance.

---

## 6. Số liệu đã kiểm chứng — ĐỪNG tính lại

**Dữ liệu:** 1 người · 27/11/2017 → 23/09/2022 (**1.762 ngày**) · 2,71M record · 356.862 HR ·
**104.565** window 15' · **289 workout** · tz `Australia/Adelaide`.

**Nhịp tim (103.644 rest window):** p10 = **74,3** · median = **85,0** · p75 = 92,2 · p90 = 99,3 ·
**49,1%** vượt 85. Band cohort 40–49 sedentary = **60–85**.
> ⚠️ **74,3 nằm TRONG band 60–85** (KHÔNG cao hơn — đừng lặp lại nhầm lẫn này).
> Luận điểm chính xác: resting **thật sự** (p10 = 74, lúc yên nhất) **bình thường**; cái chạy cao là
> **trương lực nghỉ ban ngày** (median 85, đúng trần band). Con số 📄 "15% bị flag" là ngưỡng
> `deviation ≥ 1` nửa-độ-rộng (HR > 97,5), khác với 49,1% "vượt trần".

**Context pipeline tự suy (fitness đã gỡ circularity, chạy lại 2026-07-22):**
```
age_band               = unknown       (conf 0.15 — HRmax_obs=152 → age~73 không đáng tin)
fitness_level          = recreational  (conf 0.80 — EXERTION: HRR60~19bpm/34 workout + 1,2 buổi/tuần)
fitness_resting_report = recreational  (conf 0.63 — REPORT-ONLY: rest p10~74 → "average"; KHÔNG feed band)
home_climate           = temperate     (conf 0.90 — home = Gamma Crescent, Panorama)
```

**Coverage:** 1.761 ngày có dữ liệu · ≥12h: **98,6%** · ≥16h: 50,9% · ≥18h: 11,4% · ≥4h đêm: **13,3%**
· median **16h/ngày**. Giờ đêm 00–05h chỉ **16%** ngày có dữ liệu vs ngày 08–22h **91,7%** (lệch
**5,7×** — chỗ khuyết **có cấu trúc**, trùng đúng đáy circadian).

**Tín hiệu đã trích (ingestion):** HR (356.862) · **HRV SDNN (4.885)** · Resting HR (1.759) · Workouts (289).
**KHÔNG có** dữ liệu bước chân (steps) **và KHÔNG có sleep-stage** (`SleepAnalysis`) ở bất kỳ đâu trong
pipeline → giới hạn cứng cho mọi hướng cần giấc ngủ (xem §4① hướng ②).

**Kết quả audit** 📄 (prevalence anomaly ≈ 0,9% → PR-AUC chance ≈ 0,009), trước → sau enrichment:

| Detector | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| LLM-semantic (node) | 0.744 → 0.701 | 0.266 → **0.334** | 0.390 → **0.427** |
| GCN-DOMINANT (node) | 0.870 → 0.850 | 0.117 → 0.106 | 0.333 → 0.303 |
| GCN-DOMINANT (edge) | 0.671 → **0.732** | 0.046 → **0.088** | 0.165 → **0.243** |

---

## 7. Vướng mắc & cần xác minh

- ✅ **Circularity ĐÃ SỬA (2026-07-22)** — `fitness_level` giờ suy từ EXERTION (HRR60 + volume), độc lập
  resting HR; resting-based hạ REPORT-ONLY (xem §4①). **Còn lại:** nối vào File 3/4 và chạy lại metrics.
- 🔴 **Nhãn fitness nào sinh ra bảng metrics ở §6?** 📄 `tong_ket` ghi baseline **"40–49 sedentary"**,
  nhưng pipeline hôm nay predict **`recreational`**. ✅ **Nguồn thật của lệch nhãn (2026-07-22):** KHÔNG
  phải missingness đẩy resting lên ~86 — mà do **chọn thống kê**: `_resting_hr` dùng **p10=74** (→ chart
  "average" → `recreational`), còn narrative cũ dùng **median~85** (trương lực nghỉ ban ngày, sát trần →
  sedentary/deconditioning). Hai tín hiệu **KHÁC nhau** — đúng luận điểm hybrid §7/LĐ5, không phải bug
  "chọn một nhãn". ✅ **Đã chốt nhãn feed-band = `recreational`** qua exertion (§4①, độc lập resting).
  **Vẫn cần chạy lại File 3/4** để đo bảng §6 stale tới đâu với nhãn/band này.
- ✅ Vocab Task 1 **đã được nối vào `context_library`** (2026-07-22, §4③): import one-way artifact
  `generated_vocab`, cấp Literal cho `age_band`/`sex`/`occupation`/`heart_health`/`sleep`.
- 🟡 Chất lượng normalize phụ thuộc alias LLM sinh — ✅ quan sát thật: `heart_health "very fit" → unknown`.
- 🟠 File 3/4 **chưa chạy lại** sau khi `location_context` thêm park/water vào File 2 → results có thể **stale**.
- 🟠 File 3b (BIDSleep) shelved · nửa dưới sơ đồ p3/9 chưa xây (§2.3).

**Luận điểm mở quan trọng nhất (📄 LĐ5):** baseline population cho resting 60–85 nhưng subject chạy
sát/trên trần → ~15% window bị flag. Đó là tín hiệu **mạn tính**, không phải chuỗi sự kiện cấp tính.
Cần **baseline HYBRID**: `population-high AND personal-normal` → mạn tính; `personal-high` → cấp tính.
Chưa hiện thực. 🔬 Nguồn ngoài (foundation model quy mô 5 triệu người) **thừa nhận đây là giới hạn và
không có giải pháp** ⇒ ý tưởng hybrid này là **đóng góp gốc**, đáng trình bày với Asara.

---

## 8. Đã GÁC — đừng đào lại

Nhánh sau **không phục vụ task nào của Asara**, phát sinh từ đề xuất ngoài:
- Phân cụm **trạng thái theo ngày** (ngày stress / nghỉ / vận động).
- **Cosinor per-day, Intradaily Variability**, bộ 20 daily features.
- Trích **StepCount** từ `export.xml`.
- Đối chiếu **SensorFM**. *Kết luận đã rút và giữ lại:* không có public weights → Task 3 tự dựng
  Transformer là đúng · `age` nên để **liên tục** · họ né circularity nhờ demographics **khai báo từ
  ngoài** + z-score **tĩnh toàn cục**.
  ⚠️ Quy trình *interpolate rồi fit Cosinor* của họ **KHÔNG dùng được** cho data này — chỗ khuyết của
  ta **có cấu trúc** (§6), nội suy sẽ tự chế ra cái đáy circadian.

---

## 9. Kiểm thử & thứ tự

Mỗi phần một demo chạy **offline** (không cần API key); chạy lần 2 phải **tái lập** nhờ cache + seed.
- `python demo_context_providers.py` → persona có truth khôi phục đúng; **in cả fitness exertion-based
  lẫn resting-based**; thiếu dữ liệu → `unknown` không crash.
- `demo_global_context.py` → cohort synthetic có truth 2 cấp khôi phục đúng; **§5 bước 6**: prior gated đúng
  (athletic/cardiac applied, office conf<0.5 gated, individual/user thắng prior, subject thật → prior rỗng);
  chạy được subject thật. PASS cả `--offline` lẫn live.
- `python demo_vocab.py` → vocab regenerate, `age_band`/`sex`/`dataset_domain` tách đúng, Literal khớp JSON.

**Thứ tự:** ①  gỡ circularity ✅ → §7 xác minh nhãn fitness → ③  tách demographic + regenerate vocab ✅ →
②  Global Context ✅ (**bước 1–6 xong → Task 2 hoàn tất**) → ④  occupation (enhancement) → sau đó mới
**Task 3** (scope chưa chốt: text-encoder trên feature→text **hay** time-series Transformer — cần quyết
vì đổi hẳn thiết kế).

---

## 10. Bản đồ repo

```
notebooks/
  01_ingestion…ipynb · 02_context_semantic.ipynb · 03_baseline_gnn_anomaly.ipynb
  03b_cohort_baseline.ipynb (pending) · 04_audit_metrics.ipynb
  graph_model.py                       # GCN-DOMINANT dùng chung File 3 & 4
  context_baseline/   context_library.py · context_providers.py · global_baseline.py
  context_vocab/      context_profile.py · vocab_generator.py · vocabulary.json · generated_vocab.py
  enrichment_experiment/ · location_context/
src/agents/           legacy Gemini agents
results/              output theo track (git-tracked) · data/ (git-ignored)
docs/                 PROJECT_STATUS.md (file này) · workflow.md · methodology.md
```

Tài liệu tham chiếu khác: `notebooks/health_trajectory_dev_guide.md` (kiến trúc + guardrail gốc) ·
`notebooks/tong_ket_pipeline_va_ket_qua.md` (kết quả & luận điểm) · `*_design.md` trong mỗi track.
