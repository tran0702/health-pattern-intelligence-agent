# Context-Driven Global Baseline — Design (approach (a))
### Trả lời 3 task của Asara · track cô lập `notebooks/context_baseline/`

> Chốt với user 2026-07-14: theo **phương án (a)** — LLM sinh **trực tiếp** khoảng
> "normal" kỳ vọng từ **prior y văn**, index theo context; không cần dataset cohort có nhãn.

Ánh xạ 3 kết luận họp Asara:
1. **External Context Definition** — context (location, health conditions, goals) là
   *meta-information ngoài*, nạp vào; pipeline **generic**, không tự quyết context.
2. **LLMs for Global Baselines** — context xác định cohort → **LLM dựng global baseline
   of normality** cho nhóm đó (approach (a): emit khoảng kỳ vọng từ prior).
3. **Anomaly Detection** — có baseline → dò deviation **và diễn giải** ý nghĩa
   lifestyle/health.

---

## Task 1 — Context Library
File: [`context_library.py`](context_library.py)

Context = một **registry thuộc tính** được xác định TRƯỚC pipeline. Mỗi thuộc tính có
controlled vocabulary (Pydantic `Literal`), `source` (cách lấy), `fallback` (khi thiếu),
và `consumed_as` (pipeline dùng vào việc gì). Tách 2 mức:

- **Subject-level = cohort descriptor** (`SubjectContext`): `age_band`, `sex`,
  `fitness_level`, `health_conditions[]`, `goal`, `home_climate`. **Đây là thứ định
  nghĩa "cohort"** và được đưa cho LLM để dựng baseline.
- **Episode-level = dynamic** (`EpisodeContext`): `activity`, `location_type`,
  `hour_of_day`, `weather`. Quyết định **sub-range nào** của baseline áp cho cửa sổ đó.

| field | level | source | consumed_as |
|-------|-------|--------|-------------|
| age_band | subject | predict từ HR (Tanaka/HRV) **hoặc** user | baseline_index |
| sex | subject | user (optional) | baseline_index |
| fitness_level | subject | predict từ HR (resting percentile, HR-recovery) **hoặc** user | baseline_index |
| health_conditions | subject | user / clinical | baseline_index |
| goal | subject | user | interpretation |
| home_climate | subject | derive từ location (geocode) **hoặc** user | interpretation |
| activity | episode | derive từ signal (workout + motion) | chọn sub-range |
| location_type | episode | derive từ signal (GPS + reverse-geocode, tái dùng enrichment C2) | interpretation |

**Robust khi thiếu data (yêu cầu Asara):** mọi field mặc định sentinel `"unknown"`;
`availability()`/`coverage()` báo field nào thật sự biết. Thiếu field → baseline
**nới rộng band** (xem Task 2) chứ không crash. Đây là tổng quát hoá pattern đã có sẵn
trong repo (GPS thiếu → `unknown`, weather fail → NaN, `elig_*` của enrichment track).

**Về age (điểm 4 họp):** `age_band` dự đoán từ HR có cơ sở y văn — Tanaka
`HRmax = 208 − 0.7·age` (đảo ngược từ max-HR quan sát trong workout), resting-HR & HRV
giảm theo tuổi. Chỉ xuất **band thô** + caveat, không phải tuổi chính xác.

### Providers — establish context TỪ DATA ([`context_providers.py`](context_providers.py))
Đây là nửa "predict" của Task 1 (Goal: *identify exactly what you predict from the
initial dataset*). Mỗi provider trả `FieldEstimate(value, confidence, evidence)` và
tự lùi về `unknown` khi thiếu input:
- **`predict_age_band`** — đảo Tanaka từ **đỉnh HR trong workout** (p99.5), de-bias
  `HRmax_est = peak/0.97`. **Chỉ tin đỉnh từ gắng sức**: nếu không có HR mức workout →
  đỉnh nghỉ underestimate max → confidence thấp → `unknown` (trung thực: **không suy được
  tuổi cho người ít vận động**, đây là giới hạn sinh lý chứ không phải bug).
- **`predict_fitness_level`** — từ resting-HR (p10 non-workout): athletic bradycardia.
- **`predict_home_climate`** — từ phân bố nhiệt/ẩm của weather (Open-Meteo).
- `sex`/`health_conditions`/`goal`: không có predictor → chỉ nhận `user` override.
- **`build_subject_context(frames, user)`**: user override luôn thắng (conf 1.0); field
  predict chỉ được nhận nếu `confidence ≥ min_conf`, không thì `unknown`.

**Validate ([`demo_context_providers.py`](demo_context_providers.py)):** persona có tuổi
biết trước → suy đúng band khi có workout (athlete 24→`18_29`, recreational 35→`30_39`);
sedentary không gắng sức → `age_band=unknown` (đúng); bỏ workouts / bỏ hết data → toàn
`unknown`, coverage 0%, không crash. Chạy trên `data/processed/*.parquet` thật khi có
(`load_frames()`).

**Kết quả trên DATA THẬT (đã parse export.xml, 356k HR / 289 workout):**
`fitness_level=sedentary` (resting~74); **`age_band=unknown`** — subject vận động nhẹ
(90% HR ≤104, workout 62% đi bộ, đỉnh workout p99.5=140) nên KHÔNG suy được tuổi từ HRmax.
Đây làm lộ (và đã sửa) một lỗi over-confidence: provider ban đầu ra `60_plus` conf .60 →
thêm **peak-gate** (đỉnh < 160 bpm ⇒ confidence thấp ⇒ `unknown`, vì đỉnh thấp không phân
biệt được "người già ở max" với "người trẻ chưa gắng sức"). ⇒ với subject này `age_band`
phải được **cung cấp từ ngoài** — đúng minh hoạ cho nguyên tắc "context established externally
+ robust khi thiếu".

---

## Task 2 — Biểu diễn & cách dùng Context
File: [`global_baseline.py`](global_baseline.py)

**Trả lời thẳng câu hỏi Asara (training / rule / normal-abnormal?):** context được dùng
để **XÁC ĐỊNH NORMAL** (chọn/sinh baseline), **không** làm training target, **không** làm
rule cứng phán anomaly.

### Biểu diễn
`SubjectContext` (typed, controlled vocab) — cùng "ngôn ngữ" giữa cá nhân & cohort. Baseline
là `NormativeBaseline`: `resting_hr`, `sleep_hr`, `light_activity_hr`,
`vigorous_activity_hr`, `hrv_sdnn_ms` (mỗi cái là `HrRange{low,high}`), `max_hr_bpm`,
`rationale`, `caveats`, `source`.

### Approach (a): LLM dựng baseline
`establish_baseline(ctx, source=...)`:
- `source="llm"`: Gemini `gemini-3.1-flash-lite`, `temperature=0`, structured output =
  `NormativeBaseline`, `thinking_budget=0`, **cache theo hash prompt**. LLM emit khoảng
  kỳ vọng cho *cohort* (không phải data người này) từ prior sinh lý.
- `source="default"` (offline, cũng là fallback): công thức textbook —
  **Tanaka** HRmax, **Karvonen** HR-reserve (light 30–50%, vigorous 60–85% HRR),
  bảng resting-HR theo fitness, bảng SDNN theo tuổi. Thiếu context → band tự nới.
- `source="auto"`: có key → LLM, không → default.

> **Guardrail #1 vẫn giữ:** LLM chỉ đặt **mốc tham chiếu tĩnh** (baseline). Nó **không**
> flag từng sample. Việc dò deviation do detector **deterministic** làm → LLM là cây
> thước, detector là phép đo. Không circularity.

### Dùng context: dò deviation + diễn giải
- `detect_against_baseline(episodes, base)`: mỗi episode, `EpisodeContext.activity` +
  giờ chọn sub-range (sleep/resting/light/vigorous); `deviation` = khoảng cách **ngoài**
  band / nửa-độ-rộng band (0 nếu trong band); flag khi `deviation ≥ 1`.
- `translate(row, ctx)`: dịch deviation thành câu lifestyle/health, **non-diagnostic**,
  có điều biến theo `location_type`/`home_climate` (vd nóng → HR nghỉ cao).

### Ba vai trò của context — giữ TÁCH BẠCH
1. **Baseline indexing (chính):** subject context → `P_normal(HR | cohort)`. Anomaly =
   lệch band. Có backoff khi thiếu: `age+fitness` → nới → toàn dân.
2. **Node feature cho GNN (phụ, nếu dùng):** nhãn context one-hot làm attribute — chỉ
   NHÃN, không đưa phán quyết anomaly của LLM (tránh circularity).
3. **Không** rule cứng, **không** training target.

---

## Task 3 — Biện luận GNN vs GCN

**Sự thật code (đã kiểm):** trong toàn pipeline chỉ có **MỘT** model đồ thị —
`DominantLite`, một autoencoder kiểu **DOMINANT** với các lớp **`GCNConv`**. Xuất hiện ở
File 3 (train trên graph thật) và File 4 (train lại trên graph đã tiêm anomaly).
"GNN" trong repo **chỉ là tên gọi/paradigm** (tiêu đề, comment, nhãn "GNN-structural");
"GCN" (`GCNConv`) là toán tử thực. **Không phải hai model cạnh tranh** — GNN là *họ*,
GCN là *một thành viên*.

⇒ Dự án **đã** thoả yêu cầu "chọn một model xuyên suốt": model đó là **DOMINANT nền GCN**.
Việc cần làm:

1. **Chuẩn hoá cách gọi:** đừng viết lỏng "GNN"; mô tả chính xác *"DOMINANT autoencoder
   nền GCN (thuộc họ GNN convolution)"*.
2. **Biện luận chọn GCN** (thay vì GAT/GraphSAGE): graph đồng nhất, cỡ vừa, cạnh đã
   curate & gán loại sẵn (temporal/similarity/context), bài toán **unsupervised
   reconstruction** → chuẩn hoá đối xứng của GCN là baseline mạnh, ít tham số, ít overfit;
   DOMINANT gốc định nghĩa bằng GCN. GAT thêm attention dễ overfit trên đồ thị 1-người;
   SAGE (inductive sampling) thừa vì bài toán transductive. Nếu Asara muốn: **1 ablation**
   đổi `GCNConv→GATConv` để chứng minh độ bền, nhưng GCN là model chính thức.

**2 điểm rigor cần dọn (kèm theo, độc lập với approach a):**
- File 4 có **PCA fallback** khi thiếu `torch_geometric` → đảm bảo số "GNN-structural"
  báo cáo là từ GCN thật, không phải PCA.
- `DominantLite` **định nghĩa trùng** ở File 3 và File 4 (hơi khác nhau) → gộp **một
  module dùng chung** để đúng nghĩa "một model xuyên suốt".

---

## Ăn khớp với pipeline cũ (01–04 / 03b)
- Approach (a) **thay** vai trò *baseline* của **File 3 Phần A** (personal cosinor):
  giờ baseline đến từ **context → LLM/prior**, không từ đường cong cá nhân.
- **File 3b** (cohort BIDSleep) đổi vai: từ "baseline" thành **kiểm chứng** khoảng do
  approach (a) sinh ra có khớp data cohort thật không.
- **GNN (GCN-DOMINANT)** vẫn dùng được như **detector cấu trúc thứ hai** (transition lạ),
  song song với deviation-vs-baseline — đúng tinh thần "hai view độc lập" của File 4.

## Files & chạy
```
notebooks/context_baseline/
  context_library.py           # Task 1: schema + registry
  global_baseline.py           # Task 2 (approach a): LLM/offline baseline + detect + translate
  demo_context_baseline.py     # end-to-end demo (offline mặc định; --llm để gọi Gemini)
  context_baseline_design.md   # (file này)
data/context_baseline/baseline_cache/   # cache baseline LLM theo hash
results/context_baseline/               # demo_scored_episodes.csv, ...
```
```bash
cd notebooks/context_baseline
python demo_context_baseline.py          # baseline physiology offline (không cần key)
python demo_context_baseline.py --llm    # baseline Gemini (cần GEMINI_API_KEY trong .env)
```
**Cô lập:** chỉ đọc `data/`, ghi `data/context_baseline/` + `results/context_baseline/`.
Không import / không sửa 01–04 / 03b / enrichment_experiment.
