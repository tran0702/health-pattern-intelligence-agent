# Plan — Context Enrichment Experiment (LLM vs Rule)
### Track thí nghiệm ĐỘC LẬP · tách hoàn toàn khỏi `health_trajectory` pipeline

> Mục tiêu: kiểm chứng **cách nào enrich context tốt hơn — LLM hay rule-based** — bằng
> một dataset cohort **có nhãn context thật** (ExtraSensory). "Enrichment trước, anomaly sau."
>
> **Nguyên tắc cô lập:** track này KHÔNG import, KHÔNG ghi đè, KHÔNG chia sẻ parquet/biến
> với `01–04`/`03b`. Thư mục riêng, `data/` riêng, `results/` riêng. Hai hướng chỉ được
> so sánh **ở tầng kết quả**, không trộn code — để không hướng nào "nhiễm" hướng kia.

---

## 0. Vì sao track này tồn tại (đọc trước)

- Giáo sư muốn dùng **LLM để enrich context** và định nghĩa "lifestyle qua anomaly so cohort".
- Quyết định đã chốt với user: **cohort baseline lấy từ DATA THẬT** (LLM chỉ enrich, không tự
  đẻ "normal"); **deliverable = lifestyle axis** (tách khỏi health/GNN của pipeline cũ).
- **Enrichment là mắt xích gốc** của cả hướng đó → phải chứng minh enricher đáng tin **trước**.
  Nếu enricher sai, mọi "lifestyle anomaly" phía sau đều rác.
- ExtraSensory có **nhãn context tự báo** → cho phép **chấm accuracy** enrichment (không chỉ agreement).

### Sự thật dữ liệu (đã scout)
- **ExtraSensory** (cohort chính): 60 người, 326,687 mẫu ~1 phút, 51 nhãn context tự báo
  (posture / **location: gym, beach, home...** / activity / companion / phone-position).
  Feature cảm biến: accel, gyro, magnet, watch-accel, location, audio(MFCC), phone-state.
  **KHÔNG có HR.** Format: CSV/người. License: free-for-research (cite) — *xác minh điều khoản trên site.*
- **LifeSnaps** (cohort dự phòng, giàu sinh lý): 71 người, 4 tháng, Fitbit Sense (HR, sleep,
  stress) + EMA location/mood 3×/ngày. CC-BY 4.0, Zenodo.

---

## 1. Câu hỏi nghiên cứu & phạm vi

**RQ1 (trọng tâm lần này):** Cho cùng feature cảm biến của một mẫu, **LLM** hay **rule/ML baseline**
map về nhãn context **chính xác hơn** (so nhãn tự báo thật)?

**RQ2 (sau, ngoài scope enrichment-first):** Dùng enricher tốt nhất → dựng cohort behavioral
baseline `P_cohort(context)` → đo độ lệch lifestyle của cá nhân Apple-Health. *Chỉ làm khi RQ1 xong.*

**Ngoài scope:** health/HR anomaly (đã có ở pipeline cũ). Track này KHÔNG đụng tới.

---

## 2. Context Definitions — schema CỐ ĐỊNH (Bước 0, giáo sư nhấn mạnh)

LLM **chỉ được điền vào schema này**, không tự thêm chiều. Chốt trước khi code:

```python
from pydantic import BaseModel
from typing import Literal
class ContextLabel(BaseModel):
    location: Literal['home','work','gym','beach','restaurant','transit','outdoors','unknown']
    activity: Literal['sitting','standing','walking','running','cycling','exercise',
                      'sleeping','eating','unknown']
    companion: Literal['alone','with_friends','with_family','with_coworkers','unknown']
    # KHÔNG có is_anomaly / confidence-về-anomaly. Enrichment chỉ NGỮ NGHĨA.
```
- Đây cũng là schema sẽ tái dùng cho cá nhân ở RQ2 → đảm bảo cùng "ngôn ngữ" giữa cá nhân & cohort.

### 2.1 Bộ 51 nhãn gốc ExtraSensory (5 nhóm) — để đối chiếu
> `TODO(user-input)`: xác nhận với giáo sư; đây là bộ chuẩn của paper, cần khớp lại header CSV thật.

- **Posture/Movement (7):** Lying down, Sitting, Standing, Walking, Running, Bicycling, Stairs (up/down)
- **Location (13):** At home, At main workplace, At school, **At the gym**, At a restaurant, At a bar,
  At a party, **At the beach**, In a car, On a bus, Elevator, Indoors, Outside
- **Activity (24):** Sleeping, Eating, Cooking, Shopping, Strolling, Drinking(alcohol), Bathing-shower,
  Toilet, Grooming, Dressing, Cleaning, Doing laundry, Washing dishes, Watching TV, Surfing internet,
  Talking, Computer work, In a meeting, In class, Lab work, Exercise, Singing, Drive-driver, Drive-passenger
- **Companion (2):** With friends, With co-workers
- **Phone position (4):** Phone in pocket, Phone in hand, Phone in bag, Phone on table

### 2.2 `LABEL_MAP` (dùng TÊN CỘT THẬT — đã xác minh trên file, 278 cột / 225 feat / 51 label)
Nhãn có prefix `label:`; giá trị 1 / 0 / `nan`(=missing). `LOC_*` là nhãn location đã được
researcher làm sạch bằng absolute location → **ưu tiên dùng cho field location**.

```python
LOCATION_MAP = {   # ưu tiên nơi-cụ-thể > indoors/outdoors
  'gym':        ['label:AT_THE_GYM'],
  'beach':      ['label:LOC_beach'],
  'home':       ['label:LOC_home'],
  'work':       ['label:LOC_main_workplace','label:AT_SCHOOL','label:IN_CLASS',
                 'label:IN_A_MEETING','label:LAB_WORK'],
  'restaurant': ['label:FIX_restaurant','label:AT_A_BAR','label:AT_A_PARTY'],
  'transit':    ['label:IN_A_CAR','label:ON_A_BUS','label:ELEVATOR',
                 'label:DRIVE_-_I_M_THE_DRIVER','label:DRIVE_-_I_M_A_PASSENGER'],
  'outdoors':   ['label:OR_outside'],
  # 'label:OR_indoors' -> cờ phụ, không phải location cụ thể
}
ACTIVITY_MAP = {   # ưu tiên hành động-cụ-thể > posture
  'sleeping':['label:SLEEPING'], 'running':['label:FIX_running'],
  'cycling':['label:BICYCLING'], 'exercise':['label:OR_exercise',
      'label:STAIRS_-_GOING_UP','label:STAIRS_-_GOING_DOWN'],
  'walking':['label:FIX_walking','label:STROLLING'], 'eating':['label:EATING'],
  'sitting':['label:SITTING'], 'standing':['label:OR_standing'], 'lying':['label:LYING_DOWN'],
  # [CHỐT] mịn (COOKING/CLEANING/SHOPPING/WATCHING_TV/COMPUTER_WORK/TALKING/...) -> 'unknown' (gộp cho tiện).
}
COMPANION_MAP = {'with_friends':['label:WITH_FRIENDS'],
                 'with_coworkers':['label:WITH_CO-WORKERS']}   # else 'alone' (suy diễn)
# Phone position (PHONE_IN_POCKET/HAND/BAG/ON_TABLE): KHÔNG đưa vào target — chỉ là feature/nuisance.
```
- **Multi-label → 1 nhãn/field theo ưu tiên** trên (đầu danh sách = ưu tiên cao). Nếu không nhãn nào =1 → `unknown`.
- **Xử lý `nan`(missing) — [CHỐT]:** khi *derive nhãn thật* coi `nan`=không-liên-quan; khi *chấm điểm*
  một field, **loại các mẫu mà cả field đó toàn `nan`** (gold không xác định) để F1 không bị nhiễu.

---

## 3. Thiết kế thí nghiệm (RQ1) — A/B công bằng

Cùng input, cùng split, chỉ đổi enricher.

### 3.1 Data prep
- Đọc ExtraSensory (per-user `*.features_labels.csv.gz`). Ghép feature + nhãn. Chuẩn hoá nhãn thật → schema Mục 2.
- **Split theo NGƯỜI** dùng **partition 5-fold CHÍNH THỨC** của ExtraSensory (file cross-validation,
  cùng split với paper Vaizman2017a) → tái lập được, so được với paper. Train/dev enricher trên
  train-users của fold, chấm trên test-users. Rule/ML fit ở train-users; LLM few-shot ví dụ lấy từ train-users.
- Giữ **gold test set = test-users của fold** với nhãn thật, không cho enricher thấy nhãn.
- Primary features chỉ có location *tương đối*; **KHÔNG** dùng absolute lat/long ở giai đoạn đầu
  (giữ công bằng, đúng thiết lập paper) — để dành làm biến thể sau.

### 3.2 Hai enricher
- **(Ctrl) Rule/ML baseline** — cách "không LLM":
  - Rule ngưỡng đơn giản trên feature (vd accel-variance → walking/running) **hoặc**
  - ML nhẹ (logistic/RandomForest) train trên users A. Đây là **control mạnh, không strawman**.
- **(Exp) LLM enricher** — cách giáo sư:
  - **Model: `gemini-3.1-flash-lite`** (stable id, đã verify 2026-07-09 còn hiệu lực). Gọi qua
    **SDK mới `google-genai`** (`from google import genai`) — KHÔNG dùng `google-generativeai`
    (đã deprecated 30/11/2025). API user tự cấp; truyền key tường minh `genai.Client(api_key=...)`.
    Structured output: `types.GenerateContentConfig(temperature=0, response_mime_type='application/json',
    response_schema=ContextLabel)` → `resp.parsed` trả Pydantic instance. Set **thinking minimal**.
  - Prompt: mô tả feature-đã-tóm-tắt của mẫu (vd "accel năng lượng cao, GPS tốc độ 8km/h,
    **khung giờ 6–12h** lấy từ `discrete:time_of_day`") → trả `ContextLabel`. `temperature=0`,
    **cache theo hash prompt**, pin đúng model id, log parse-fail.
  - **Giờ-địa-phương [CHỐT phương án (c)]:** KHÔNG suy timezone từ epoch. Dùng thẳng 8 feature
    `discrete:time_of_day:between{0and6..21and3}` (researcher đã tính theo giờ địa phương) để mô tả thời điểm.
  - **Mù với nhãn thật.** Few-shot ví dụ (nếu dùng) chỉ lấy từ users A.

> **Guardrail chống rò rỉ:** không đưa nhãn thật (kể cả gián tiếp) vào prompt/feature. Không
> đưa bất kỳ tín hiệu "bất thường" nào — đây là gán context, không phán anomaly.

### 3.3 Biến kiểm soát
- Cùng test users, cùng feature-tóm-tắt, cùng schema, cùng bộ nhãn hợp lệ.
- Chạy LLM 3–5 lần trên 200 mẫu để đo ổn định (temperature=0 vẫn cần kiểm).

---

## 4. Metric đánh giá (RQ1)

Chấm **vs nhãn tự báo thật** trên test users:
- **Per-field, per-class F1** + **balanced accuracy** (nhãn lệch tần suất mạnh → đừng dùng accuracy trần).
- **Macro-F1** cho từng field (location / activity / companion) — báo RIÊNG, không gộp một số.
- **Coverage**: %mẫu enricher dám gán khác `unknown`, và độ chính xác trên phần đó.
- **Over-confidence**: tỷ lệ gán nhãn cụ thể ở mẫu mà feature không đủ căn cứ (đối chiếu nhãn thật=unknown).
- **Confusion matrix** per field (gym↔outdoors, beach↔outdoors... để thấy LLM lẫn ở đâu).
- **Ổn định LLM**: flip-rate nhãn qua các lần chạy, per field.
- **Chi phí/độ trễ**: $/1k mẫu, giây/mẫu — LLM có "đáng" hơn baseline không.
- **Agreement mô tả**: Cohen's kappa LLM-vs-rule (KHÔNG coi là accuracy).

**Điều kiện "LLM thắng":** macro-F1 cao hơn control **có ý nghĩa** (bootstrap CI theo user),
mà không phải trả giá bằng over-confidence/chi phí bất hợp lý.

---

## 5. Guardrails của track (đối chiếu guide gốc)

1. LLM **chỉ ngữ nghĩa context**, không phán health-anomaly. ✅ giữ #1 guide.
2. Không rò rỉ nhãn thật vào enricher; split theo user. (chống circularity/leakage)
3. Accuracy chỉ tính **vs nhãn thật ExtraSensory** (ground truth thật, không synthetic ở RQ1).
4. tz-aware; `temperature=0` + Pydantic `Literal` + cache + pin model id. ✅ #4,#8 guide.
5. Nhãn cụm/feature ≠ nhãn ngữ nghĩa → schema Mục 2 quyết. ✅ #6 guide.
6. **Cô lập tuyệt đối** với `01–04`/`03b`: thư mục & data & results riêng.

---

## 6. Deliverables & bố cục file (tách biệt)

```
notebooks/enrichment_experiment/
  context_enrichment_experiment_plan.md   <- (file này)
  E1_extrasensory_ingest.ipynb            <- tải + ghép feature/nhãn + split theo user
  E2_context_definitions.ipynb            <- schema Mục 2 + LABEL_MAP + kiểm nhãn
  E3_enrichers.ipynb                      <- (Ctrl) rule/ML  +  (Exp) LLM, cùng interface
  E4_evaluation.ipynb                     <- metric Mục 4, bảng so sánh + confusion + robustness
data/enrichment_experiment/extrasensory/
  features_and_labels/                    <- 60 file [UUID].features_labels.csv.gz  ✓ đã có
  cross_validation_partition/cv_5_folds/  <- fold_{0..4}_{train,test}_{android,iphone}_uuids.txt  ✓ đã có
data/enrichment_experiment/llm_cache/     <- cache output Gemini theo hash prompt
results/enrichment_experiment/            <- bảng F1, confusion, cost, robustness (RIÊNG)
```

**Tải ExtraSensory (user tự làm):** từ site UCSD, chỉ cần gói
`ExtraSensory.per_uuid_features_labels.zip` (~150 MB, 60 CSV/người) — **KHÔNG** cần data cảm biến
thô (hàng chục GB). Giải nén vào `data/enrichment_experiment/extrasensory/`.

Output cuối RQ1: **một bảng so sánh** LLM vs rule (macro-F1/field, coverage, over-confidence,
cost, ổn định) + confusion matrices → trả lời "cách nào tốt hơn".

---

## 7. Giai đoạn sau (RQ2 — chỉ khi RQ1 xong, ghi để không quên)

1. Dùng enricher thắng cuộc enrich **cả** cá nhân (Apple) **và** cohort → cùng schema.
2. `P_cohort(context)` từ ExtraSensory (data thật); `P_individual(context)` từ Apple.
3. Lifestyle deviation = JS/KL divergence theo từng chiều context → "signature".
4. Eval: **synthetic lifestyle-swap** (hoán nhãn một block) làm ground truth + face validity.
5. **Caveat cứng:** context cá nhân từ Apple **thưa** (GPS chỉ khi workout) → signature giới hạn
   ở **phenotype vận động**, không phải "24h ở đâu". Cảm biến cá nhân ≠ cảm biến ExtraSensory →
   phải map feature về không gian chung hoặc chỉ so ở tầng NHÃN (khuyến nghị: so ở tầng nhãn).

---

## 8. Quyết định đã chốt (2026-07-09) — PLAN FINAL, sẵn sàng code

- [x] **Control = ML nhẹ** (logistic/RandomForest trên feature ExtraSensory) — non-LLM. ✓ user xác nhận.
- [x] **LLM = `gemini-3.1-flash-lite`** qua SDK mới **`google-genai`** (verify 2026-07-09; `google-generativeai`
      cũ đã deprecated). API user tự cấp, truyền key tường minh vào `genai.Client(...)`.
- [x] **Giờ cho prompt LLM = phương án (c)**: dùng feature `discrete:time_of_day` có sẵn, không giả định timezone.
- [x] **ExtraSensory OK** cho thử nghiệm (license research).
- [x] **Folder riêng** `notebooks/enrichment_experiment/` (đã tạo) — tránh nhầm với pipeline cũ.
- [x] Data sẵn & **định dạng đã xác minh**: 60 `features_and_labels/*.csv.gz` (278 cột) +
      `cross_validation_partition/cv_5_folds/` (fold_{0..4}_{train,test}_{android,iphone}_uuids.txt).
- [x] **`LABEL_MAP` chốt** (Mục 2.2): activity mịn → `unknown` (gộp); **loại mẫu `nan`** khi chấm điểm.

> **Trạng thái (cập nhật 2026-07-09):** E1–E4 **ĐÃ CODE + CHẠY XONG**. RQ1 đã có kết luận (xem Mục 9).
> RQ1b **Bước 0 + C2 ĐÃ CHẠY XONG** (xem §9.8): semantic location (geocode) **cứu LLM location** —
> beach recall 0→.28, macro-F1 location LLM .248→.455 (vượt ML .344). C1 (raw coords) chưa chạy.

---

## 9. Nhánh RQ1b — Absolute-GPS variant (PLAN, chưa code — cho Claude Code session sau)

### 9.0 Bối cảnh (đọc trước — trạng thái hiện tại của track)
- **Đã có (source of truth = code, không chỉ plan):**
  - `ee_common.py` (paths, `ContextLabel`, `LABEL_MAP`, loaders, folds, `derive_gold_labels`, Gemini helpers),
    `ee_enrichers.py` (`summarize_features`, `build_fewshot`/`fewshot_by_fold`, `LLMEnricher` có retry+cache).
  - Notebook `E1_extrasensory_ingest`, `E2_context_definitions`, `E3_enrichers`, `E4_evaluation` — đã chạy.
  - Artifacts: `data/enrichment_experiment/e2_gold_labels.parquet` (377,346 dòng, gold+elig 3 field);
    `results/enrichment_experiment/`: `e3_pred_ml.parquet` (ML full test), `e3_pred_llm.parquet` (LLM few-shot,
    trên eval sample), `e3_eval_sample.parquet` (1,530 mẫu, stratify theo location, dùng CHUNG để so công bằng),
    `e3_llm_cost.json`, `e4_comparison.csv`, `e4_ml_full_scores.csv`.
  - LLM cache: `data/enrichment_experiment/llm_cache/*.json` (hash prompt → không gọi lại). Key ở
    `notebooks/enrichment_experiment/.env` (`GEMINI_API_KEY`).
- **Kết quả RQ1 (đã chốt):** ML thắng LLM cả 3 field, bootstrap-CI theo user < 0.
  macro-F1 ML / LLM-zeroshot / LLM-fewshot: location .344/.226/.248; activity .283/.155/.229; companion .403/.230/.287.
  LLM **over-confident** (location coverage .85, over-conf .91), **không bao giờ đoán `beach`**, companion → hầu như luôn `alone`.

### 9.1 Câu hỏi RQ1b
> LLM yếu ở **location** vì **thiếu tín hiệu phân biệt** (feature chỉ *tương đối*, không có toạ độ thật)
> hay vì **bản thân LLM dở**? → Thêm **vị trí tuyệt đối** rồi đo lại, tập trung vào field **location**.

### 9.2 SỰ THẬT DỮ LIỆU (quan trọng — đã xác minh 2026-07-09)
- Bộ `features_and_labels/*.csv.gz` trên đĩa **KHÔNG có lat/long tuyệt đối** — chỉ có vị trí *tương đối*:
  `location:log_latitude_range/log_longitude_range/min_altitude/max_altitude/min_speed/max_speed/diameter/…`
  và `location_quick_features:std_lat/std_long/lat_change/long_change/…`. Đây là **độ biến thiên vị trí trong
  cửa sổ mẫu**, KHÔNG cho biết *ở đâu* (beach vs home không phân biệt được khi ngồi yên).
- **Absolute location là gói TẢI RIÊNG** từ site UCSD (extrasensory.ucsd.edu):
  `ExtraSensory.per_uuid_absolute_location.zip` → mỗi người 1 CSV (đại khái cột `timestamp, latitude, longitude`;
  **verify tên cột trên file thật** khi tải về). Giải nén vào `data/enrichment_experiment/extrasensory/absolute_location/`.
- **CAVEAT KHOA HỌC (đừng bỏ qua):** toạ độ tuyệt đối gần như là **định danh riêng từng người** (nhà/gym của mỗi
  người là 1 toạ độ cố định) → cho ML thì gần như "học thuộc" chỗ của từng người (person-specific, gần rò rỉ,
  KHÔNG generalize). Chính vì thế paper gốc **cố tình loại** absolute location. ⇒ Không so macro-F1 của điều kiện
  có-GPS với điều kiện không-GPS như thể cùng độ khó; phải diễn giải cẩn thận (xem 9.5).

### 9.3 BƯỚC 0 — Pre-check RẺ, KHÔNG cần tải gì (làm trước, có thể trả lời luôn câu hỏi)
- Tính **F1 per-class cho field location** cho **cả ML (full 225 feat) lẫn LLM few-shot** trên `e3_eval_sample`,
  kèm **support** mỗi lớp. Dùng `e3_pred_ml.parquet` + `e3_pred_llm.parquet` + `e2_gold_labels.parquet`.
- **Logic suy luận:** nếu **ngay cả ML full-feature** cũng ~0 ở `beach/gym/restaurant` (các lớp "tĩnh, ít tín hiệu")
  nhưng cả hai đều tốt ở `transit`(tốc độ)/`home` → **trần bị chặn bởi TÍN HIỆU, không phải mô hình** → ủng hộ giả
  thuyết "LLM thiếu tín hiệu" mà **không cần GPS**. Nếu ML khá hơn hẳn LLM ở các lớp đó → nghiêng về "LLM dở hơn".
- **Deliverable:** `results/enrichment_experiment/e5_location_per_class.csv` (class, support, ml_f1, llm_f1) + 1 đoạn nhận định.
- ⏸ **Sau bước 0, quyết định** có cần tải GPS thật hay không.

### 9.4 BƯỚC 1–3 — Nếu vẫn muốn thử GPS thật
1. **Lấy data:** user tải `ExtraSensory.per_uuid_absolute_location.zip` → giải nén → merge lat/long vào mẫu theo
   `(uuid, timestamp)`. Thêm loader vào `ee_common.py` (`load_absolute_location(uuid)` / merge helper). Nhiều mẫu
   sẽ **thiếu GPS** (giữ `NaN`, đừng bịa).
2. **3 điều kiện, giữ CÙNG `e3_eval_sample` để so được:**
   - **C0** = baseline hiện tại (chỉ feature tương đối) — đã có.
   - **C1 (toạ độ thô):** ML thêm 2 cột lat/long vào ma trận; LLM thêm dòng "latitude X, longitude Y" vào prompt.
     *Kỳ vọng:* giúp ML nhiều (học thuộc chỗ), giúp LLM **ít** (toạ độ số không có nghĩa ngữ nghĩa với LLM).
   - **C2 (geocode ngữ nghĩa — PHÉP THỬ THẬT cho LLM):** reverse-geocode toạ độ → **loại địa điểm/POI gần đó**
     ("gần một bãi biển / phòng gym / nhà hàng") rồi đưa MÔ TẢ đó vào prompt LLM. Đây mới là thứ cho LLM tín hiệu
     location dùng được. ⚠️ C2 biến hệ thành **LLM + geocoder** (hybrid) — bản thân điều đó là 1 finding về "LLM cần
     gì". Dùng offline (reverse_geocoder/OSM Nominatim local) hoặc API maps; **cache** kết quả geocode.
3. **Code:** thêm cờ `CONDITION ∈ {C0,C1,C2}` — mở rộng `ee_enrichers.summarize_features`/prompt (thêm coords/geocode),
   và biến thể ML có cột toạ độ. Notebook mới `E5_absolute_gps.ipynb` (hoặc thêm cell vào E3). **Tái dùng eval sample cũ**;
   LLM cache tự khác hash nên chỉ gọi mẫu mới.

### 9.5 Metric & CÁCH ĐỌC (tập trung field location)
- **F1 per-class** (nhất là **beach/gym/restaurant**): **recall `beach` của LLM có nhảy từ 0 lên >0 không?**
- Over-confidence lại (C0 vs C1 vs C2), cho cả ML & LLM.
- **Diễn giải:**
  - C2 nâng LLM location mạnh → xác nhận **"LLM bị bỏ đói tín hiệu, không phải kém"**.
  - C1 nâng ML nhưng không nâng LLM, phải tới C2 LLM mới lên → LLM cần **location ngữ nghĩa**, không phải toạ độ thô.
  - Không gì nâng được `beach` → nhãn/tín hiệu quá thưa (giới hạn của dataset).
- **Nhắc:** C1/C2 với ML là **bài toán DỄ HƠN/person-specific** → báo cáo rõ, đừng so trực tiếp với C0 như cùng độ khó;
  câu trả lời nằm ở **thay đổi thứ hạng ML-vs-LLM** và **recall lớp hiếm của LLM**, không phải con số macro-F1 tuyệt đối.

### 9.6 Phương án thay thế (nếu tải abs-location phiền)
- Cùng giả thuyết có thể test trên dataset **sẵn có GPS + ngữ cảnh**: LifeSnaps (EMA location) hoặc **chính Apple Health
  workout GPS của user** — nhưng đây là **port lớn hơn**, cân nhắc sau.

### 9.7 Isolation & deliverable
- Vẫn **cô lập tuyệt đối**: chỉ đọc `data/enrichment_experiment/`, ghi `results/enrichment_experiment/e5_*`.
  KHÔNG đụng File 01–04/03b, không sửa E1–E4 đã chốt (tạo E5 mới).
- Thứ tự đề xuất: **Bước 0 (30') → quyết định → Bước 1–3 nếu cần**.

### 9.8 KẾT QUẢ (đã chạy 2026-07-09) — Bước 0 + C2
**Đã có thêm (source of truth = code + results):**
- `ee_common.load_absolute_location/load_all_absolute_location` (60 file `absolute_location/*.absolute_locations.csv.gz`,
  cột `timestamp,latitude,longitude`, join theo `(uuid,timestamp)`, phủ 85% eval, **beach/gym = 100%**).
- `ee_geocode.py` (Nominatim reverse → phrase ngữ nghĩa, cache `data/enrichment_experiment/geocode_cache/`, ~1 req/s).
- `ee_enrichers.py` geo-aware **backward-compatible** (C0 byte-identical — đã verify prompt trúng cache E3).
- `E5_absolute_gps.py`; artifacts `results/`: `e5_location_per_class.csv` + `e5_location_per_class_note.md` (Bước 0),
  `e5_c2_location_per_class.csv` + `e5_c2_note.md` + `e5_pred_llm_c2.parquet` + `e5_llm_c2_cost.json` (C2).

**Bước 0 (§9.3):** trần location phụ-thuộc-lớp. `gym` là trần TÍN HIỆU (ML full-feat cũng ~0). `beach/restaurant`
ML rút được tín hiệu từ feature tương đối mà LLM không → điểm yếu rút-trích của LLM. ⇒ quyết định chạy C2.

**C2 (§9.4–§9.5) — KẾT LUẬN:** *LLM yếu location vì THIẾU TÍN HIỆU, không phải kém.* Cho gợi ý địa điểm ngữ nghĩa
(reverse-geocode) → **beach recall 0 → .28** (F1 0→.43, precision .92), gym F1 .077→.589, restaurant .059→.470,
work .313→.593. **macro-F1 location LLM: .248 → .455**, VƯỢT ML (.344). Over-confidence gần như không đổi (.911→.922);
các lớp được cứu đều **high-precision** (không phải đoán bừa). transit/outdoors ~phẳng (đã đủ tín hiệu tốc độ / lớp khuếch tán).

**C1 (raw coords cho LLM) — ĐÃ CHẠY (E5b):** macro-F1 location **.248→.307** (chỉ +.06, vẫn DƯỚI ML .344).
Toạ độ thô **không cứu lớp khó**: beach recall .00→.008, gym/restaurant <.15 F1. Bump của C1 dồn vào lớp
phổ-biến-cụm-không-gian (work .31→.55, home .43→.49) — giống LLM *match toạ-độ gần* với few-shot (chỗ-cụ-thể),
không phải hiểu ngữ nghĩa. ⇒ **Thứ tự C0 .248 < C1 .307 < ML .344 < C2 .455.** Kết luận sạch (§9.5):
**LLM cần LOCATION NGỮ NGHĨA, không phải toạ độ.** Artifacts: `e5_pred_llm_c1.parquet`,
`e5_all_conditions_location_per_class.csv`, `e5_all_conditions_note.md`, `e5_llm_c1_cost.json`.

**Caveat cứng:** C2 = **LLM + geocoder (hybrid)** — phần lớn lift là geocoder *gọi tên* chỗ (toạ độ gần "…State Beach…" ⇒
beach dễ). Đọc là "**semantic location mở khoá LLM**", KHÔNG phải "LLM > ML nói chung" (chưa cho ML toạ độ; C1/C2-ML là
bài toán person-specific/gần-rò-rỉ, paper gốc cố tình loại). **Chỉ còn (tuỳ chọn):** bootstrap-CI theo user cho khoảng
tin cậy C0→C2.

---

## 10. Nhánh RQ1c — Transformer arms (Task 3 của Asara)

### 10.1 Câu hỏi
> Asara giao: **tự tay dựng Transformer** cho context extraction (để có kinh nghiệm), thay vì chỉ dùng
> GCN + Gemini-as-API. RQ1c: **các biến thể Transformer khác nhau** map feature cảm biến → nhãn context
> chính xác tới đâu, **so với nhau** và so với ML control + LLM đã có (RQ1)?

### 10.2 Quyết định thiết kế đã chốt
- **Transformer đọc TRỰC TIẾP feature số (225 chiều), KHÔNG phải text.** Transformer text huấn luyện từ đầu
  trên ~40k câu ngắn sẽ học embedding kém → không đo đúng năng lực Transformer; **feature-token self-attention**
  là chuẩn cho tabular và so **apples-to-apples** với ML control (cùng 225 feature).
- **CPU-only**, torch 2.10+cpu, **không cần API key**. `temperature=0`-analogue = seed cố định (torch+numpy)
  → tái lập; `selftest` assert 2 lần seeded trùng khớp.
- Mọi arm xuất **full-test, đúng schema `e3_pred_ml.parquet`** → E4 chấm y hệt ML (macro-F1/field + bootstrap CI theo user).

### 10.3 Ba biến thể (khác nhau ở *token đại diện cho cái gì*)
| # | Biến thể | Token | Trục | Nhà |
|---|---|---|---|---|
| **A** | `feature` — FT-Transformer | 1 token/feature (+[CLS]) → 226 | attention **giữa các feature** | `ee_transformer.ContextTransformer` (`FeatureTokenizer`) |
| **B** | `group` | 1 token/nhóm sensor (~12) | attention **giữa nhóm cảm biến** | `ee_transformer.ContextTransformer` (`GroupTokenizer`) |
| **C** | `temporal` | 1 token/mẫu-phút, cửa sổ K mẫu gần nhất của **cùng người** (gap-aware) → predict nhãn mẫu cuối | attention **theo thời gian** | `ee_transformer.TemporalTransformer` |

Kiến trúc chung: encoder d_model=64 / 2 lớp / 4 head / GELU, pool `[CLS]` (A,B) hoặc slot cuối (C), **3 head
multi-task** location/activity/companion, tiền xử lý y hệt ML control (median impute + StandardScaler, fit per-fold
train). C: `_make_windows` xây cửa sổ phải-canh-lề, ngắt khi qua ranh giới người hoặc gap > `max_gap_s` (mặc định
600 s); diagnostic `mean_real_window` báo số timestep thật/cửa sổ (~1.0 = data quá thưa cho temporal).

### 10.4 Deliverables & trạng thái
- ✅ **`ee_transformer.py`** — A/B (`tf_fit_predict`, `token_mode`) + C (`temporal_fit_predict`) + dispatcher
  `run_fold(variant)` + `selftest(variant)`. **Đã viết + compile OK; CHƯA chạy.**
- ✅ **`E6_transformer.py`** — runner 5 fold/biến thể → `e6_pred_{feature,group,temporal}.parquet` +
  `e6_{variant}_meta.json` (config, giây/fold, macro-F1 full-test + eval-sample, `mean_real_window` cho temporal).
  Flags `--variant {feature|group|temporal|all}`, `--folds` (probe 1 fold không lưu preds), `--selftest`,
  `--train-cap/--epochs/--window/--max-gap`. **Đã viết + compile OK; CHƯA chạy.**
- ✅ **E4 mở rộng** — thêm §7: bảng đa-arm (ML · LLM · 3 Transformer) `e4_multi_arm.csv` + **bootstrap CI theo
  user** cho mọi cặp (Transformer−ML, Transformer−LLM, Transformer−Transformer) `e4_multi_arm_ci.csv`.
  Guard theo `e6_pred_*.parquet` tồn tại → E4 vẫn chạy nếu chưa có (no-op). **Đã thêm cell; CHƯA chạy.**

### 10.5 Verify (khi chạy — session sau)
1. `python E6_transformer.py --variant feature --folds 0` → probe timing fold-0 (226 token có thể chậm CPU).
   Nếu chậm → `--variant group` (~12 token) hoặc giảm `--train-cap`.
2. `python E6_transformer.py --variant all --selftest` → PASS cả 3 (2 lần seeded trùng).
3. `python E6_transformer.py --variant all` → 3 parquet + 3 meta. Kiểm `mean_real_window` của temporal
   (nếu ~1.0 → temporal thoái hoá về single-sample, báo trung thực).
4. Re-run E4 (nbconvert) → §7 bảng đa-arm + CI; assert index khớp gold, không NaN ở cột eval.
