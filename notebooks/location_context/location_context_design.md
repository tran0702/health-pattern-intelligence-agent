# Location Context — Design (Tier 1, track cô lập `notebooks/location_context/`)

> Chốt với user 2026-07-17: port hướng **location-cluster + reverse-geocode** từ
> pipeline cũ (`weather_location_workout_analysis.ipynb`) thành **module dùng chung**
> để bảo trì dễ, **không** sửa File 1–4. Phần "nối weather" của Tier 1 **đã xong sẵn**
> (episodes phủ weather 100% — File 2 đã được vá kể từ khi File 4 viết dòng
> "never joined weather"), nên Tier 1 rút gọn còn đúng phần **địa điểm**.

---

## 1. Vấn đề module này lấp

File 2 đã cluster GPS workout (DBSCAN/haversine, eps=100m) nhưng gán nhãn cluster
bằng **rule cứng theo giờ** (`home` = cluster nhiều điểm 0–5h; `work` = 9–17h ngày
thường; còn lại `outdoor`). Hệ quả trên data thật:

| location_type | số episode (File 2 rule) |
|---|---|
| unknown | 104.042 (99,5%) |
| work | 283 |
| outdoor | 184 |
| home | 56 |

Hai điểm yếu:
1. **Không có danh tính nơi chốn** — chỉ 4 nhãn thô, không biết đó là bãi biển, công
   viên, hồ chèo thuyền hay phòng gym. File 2 tự đánh dấu chỗ này `TODO(user-input)`.
2. **Nhãn dựa vào giờ mong manh** — GPS chỉ tồn tại **trong lúc workout**, nên "home =
   cluster nhiều điểm 0–5h" gần như không có bằng chứng (ít ai workout lúc 3h sáng).

`context_baseline.SubjectContext` cũng cần `home_climate` — thiết kế Task 1 ghi rõ
"derive từ location (geocode)" — nhưng **chưa có nguồn geocode** nào cấp.

## 2. Module làm gì (không đụng File 1–4)

`location_context.py`:
1. **cluster** GPS **y hệt File 2** (DBSCAN/haversine) → giữ nguyên ngữ nghĩa cluster;
2. **reverse-geocode centroid mỗi cluster đúng 1 lần** (Nominatim, cache đĩa, throttle
   ≤1 req/s, UA thật — port thiết kế từ `enrichment_experiment/ee_geocode.py` nhưng
   **self-contained**, không phụ thuộc chéo track enrichment);
3. **phân loại** tag OSM → **vocab location_type có kiểm soát**, minh bạch & audit được;
4. suy **`home_climate`** (suburb nhà + band khí hậu từ phân bố nhiệt độ) cho
   `SubjectContext`.

### Vocab location_type (mở rộng, tương thích ngược)
`home / work / gym / park / water / outdoor / unknown`

Giữ nguyên 4 nhãn cũ của File 2; **thêm `park` và `water`**. `water` đặc biệt có nghĩa
với subject này — họ **chèo thuyền (Rowing) rất nhiều**, diễn ra trên mặt nước, nên
`water` giải thích ngữ cảnh HR tốt hơn hẳn nhãn `outdoor` chung chung. Tên nơi thật
luôn được giữ ở cột **`location_place`** (vd `"West Lakes, Adelaide"`) để minh bạch,
không phá schema cũ.

### Phân loại OSM → vocab (ưu tiên đặc-trưng-trước)
`water` → `gym` → `park` → `work` → `home` → (đường/`highway` →) `outdoor`.
Điểm workout geocode được nhưng không rơi vào nhóm nào → mặc định `outdoor` (GPS
workout vốn ở ngoài trời). Không geocode được → `unknown`. Bảng ánh xạ đầy đủ nằm
trong `classify_place()` — mỗi nhánh 1 dòng, đọc là hiểu.

## 3. Guardrail (đồng bộ với các track khác)

- **Cô lập:** chỉ đọc `data/apple_health_export/workout-routes/*.gpx` +
  `data/processed/*.parquet`; chỉ ghi `data/location_context/geocode_cache/` +
  `results/location_context/`. Không import/sửa 01–04 / 03b / enrichment_experiment.
- **Robust khi thiếu:** offline / mất mạng / lỗi HTTP → dùng cache nếu có, không thì
  trả `unknown`; **không bao giờ crash** (đúng nguyên tắc repo). Chạy
  `python demo_location_context.py --offline` để kiểm chứng đường degrade.
- **Chính sách OSM:** ≤1 req/s, UA định danh app + email liên hệ, cache theo coord
  làm tròn ~11m → rerun không gọi lại mạng, tái lập hoàn toàn từ cache.
- **Không hard-code coord→nhãn** ngoài bảng OSM minh bạch; tên nơi thật giữ nguyên ở
  `location_place` để audit.

## 4. Kết quả trên DATA THẬT (chạy online, geocode cache z16)

- GPS: 5.147 điểm (177 file GPX, subsample 1pt/60s) → **47 cluster**; geocode 47
  centroid (cache vĩnh viễn sau lần đầu).
- **Phát hiện:** GPX trải nhiều thành phố/quốc gia — Adelaide (Panorama, Belair,
  Henley/Goolwa/Grange Beach, Morialta), **Brisbane, London, Tallinn, Copenhagen,
  Stockholm** → subject có đi công tác/du lịch. home region (cluster dày nhất,
  2.859 điểm) vẫn ra đúng **Panorama, Adelaide** (-35.005, 138.597).
- `home_climate`: place `"Gamma Crescent, Panorama"`, band **temperate**
  (median 14.3°C, p10 9.0, p90 24.8).

**Before/after `location_type` (chế độ cộng thêm — không hồi quy):**

| location_type | File 2 rule | + geocode |
|---|---|---|
| home | 56 | 56 |
| work | 283 | 283 |
| outdoor | 184 | 161 |
| **park** | 0 | **10** |
| **water** | 0 | **13** |
| unknown | 104.042 | 104.042 |

`water` = bãi biển/ven biển geocode được (Henley/Goolwa/Grange Beach, Twin Creek);
`park` = trail trong Morialta Conservation Park (First/Second Falls, Yurrebilla,
Pretty Corner Trail). **495 episode** giờ mang tên nơi thật ở `location_place`.

**Bài học kỹ thuật (đã kiểm chứng bằng chạy thật, không đoán):**
- zoom 18 trả vi-đối-tượng gần nhất (ghế đá, bãi đỗ xe) → phân loại sai; **zoom 16**
  trả vùng bao quanh tốt hơn, nhưng centroid nằm trên đường nên category hầu hết là
  `highway` → **cue từ khóa trên tên nơi** (beach/falls/trail…) mới là thứ phân loại
  ổn định, category OSM chỉ bổ trợ.
- Geocode workout **không** suy được nhà/chỗ làm (đó là tín hiệu theo giờ của File 2)
  → module **chỉ cộng thêm** `park/water/gym` + tên nơi, **không đè** home/work/outdoor.

## 5. Đã wire vào pipeline (2026-07-17) ✅

- **File 2** (`02_context_semantic.ipynb`) — chèn 1 cell **sau Step 4** (id `7770dfc4`,
  KHÔNG đụng Step 3 `CONTEXT_RULES`): gọi `build_location_table` + `attach_location`
  → cộng thêm `park/water/gym` + `location_place`, ghi `home_climate.json`. Cell dựng
  `behavioral_episodes` (id `c34a447f`) thêm cột `location_place` (guard nếu vắng).
  **Đã chạy nbconvert --inplace SẠCH:** episodes materialize water=16, park=12,
  outdoor 184→156, home/work giữ 56/283; 523 episode có `location_place`. edge/node
  table + bkg_graph tái tạo nhất quán trong cùng lần chạy.
- **`context_providers.py`** — `load_frames` đọc artifact
  `results/location_context/home_climate.json` vào `frames["home_geo"]`;
  `predict_home_climate` dùng nó grounding evidence (`home=Gamma Crescent, Panorama`)
  + fallback dùng band geocode khi vắng weather. **Coupling qua file, KHÔNG import
  chéo track.** File 3 gọi `build_subject_context(load_frames(), ...)` là tự hưởng —
  không phải sửa File 3.
- **`graph_model.build_node_features`** — `location_type` giàu hơn (thêm `park`/`water`)
  tự one-hot thành GCN node feature (`loc_park`, `loc_water`) khi File 3 chạy lại.

**Còn lại (không trong phạm vi lần này):** re-run File 3/4 để park/water chảy vào GCN
+ audit (episodes đã sẵn sàng; results/ hiện tính trên episodes cũ nên đang stale).

## 6. Files & chạy
```
notebooks/location_context/
  location_context.py          # core: parse GPX + cluster + geocode + classify + home_climate
  demo_location_context.py      # end-to-end trên data thật; --offline để degrade an toàn
  location_context_design.md    # (file này)
data/location_context/geocode_cache/    # cache JSON theo coord làm tròn
results/location_context/               # cluster_locations.csv, episodes_with_place.parquet, home_climate.json
```
```bash
cd notebooks/location_context
python demo_location_context.py           # geocode centroid live (cache sau lần đầu)
python demo_location_context.py --offline # cache-only, không mạng
```
