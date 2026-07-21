# Context Vocabulary — generated report

- source: **llm**  
- model: `gemini-3.1-flash-lite`  
- dimensions: 10, terms: 70

## demographic  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_provided, missing, unspecified | Baseline physiological norms cannot be determined without demographic context. |
| `child_female` | girl, female_pediatric | Higher resting heart rate and lower HRV compared to adults due to autonomic maturation. |
| `child_male` | boy, male_pediatric | Higher resting heart rate and lower HRV compared to adults due to autonomic maturation. |
| `adult_female_young` | woman_18_35, female_young_adult | Higher resting heart rate than males; HRV influenced by menstrual cycle phases. |
| `adult_male_young` | man_18_35, male_young_adult | Lower resting heart rate and higher stroke volume compared to females. |
| `adult_female_middle` | woman_36_55, female_middle_aged | HRV may show increased variability or decline during perimenopause. |
| `adult_male_middle` | man_36_55, male_middle_aged | Gradual decline in maximum heart rate and HRV typically observed. |
| `adult_female_senior` | woman_55_plus, female_elderly | Lower resting heart rate and reduced HRV complexity due to age-related autonomic changes. |
| `adult_male_senior` | man_55_plus, male_elderly | Lower resting heart rate and reduced HRV complexity due to age-related autonomic changes. |

## heart_health  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_available, missing_data, unclassified | Baseline data insufficient to establish a cardiovascular fitness profile. |
| `elite_athletic` | high_performance, cardiac_athlete, superior_fitness | Characterized by very low resting heart rate and high vagal tone, often showing rapid heart rate recovery. |
| `healthy_active` | fit, good_cardiovascular_health, physically_active | Normal resting heart rate range with consistent HRV patterns indicating efficient autonomic nervous system regulation. |
| `average_sedentary` | baseline, standard, moderate_fitness | Typical resting heart rate and HRV values for a non-athletic population; serves as the standard reference point. |
| `at_risk` | elevated_cardiac_risk, poor_fitness, deconditioned | Often associated with higher resting heart rates and reduced HRV, potentially indicating autonomic imbalance or cardiovascular strain. |

## medical_conditions  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `none` | healthy, no_known_conditions, asymptomatic | Baseline physiological parameters follow standard age-adjusted normative ranges. |
| `hypertension` | high_blood_pressure, essential_hypertension | Often associated with reduced HRV and elevated resting heart rate due to increased sympathetic tone. |
| `arrhythmia` | atrial_fibrillation, afib, tachycardia, bradycardia, irregular_heartbeat | Directly disrupts R-R interval regularity, rendering standard HRV metrics unreliable. |
| `diabetes_mellitus` | type_1_diabetes, type_2_diabetes, insulin_dependent | Frequently leads to autonomic neuropathy, resulting in blunted heart rate variability and elevated resting heart rate. |
| `pregnancy` | gestational_period, expecting | Characterized by a progressive increase in resting heart rate and significant shifts in autonomic balance throughout trimesters. |
| `thyroid_disorder` | hyperthyroidism, hypothyroidism, graves_disease, hashimotos | Hyperthyroidism typically elevates resting heart rate, while hypothyroidism may induce bradycardia. |
| `cardiac_history` | post_myocardial_infarction, heart_failure, coronary_artery_disease | Structural heart changes often result in reduced cardiac reserve and altered autonomic regulation. |
| `unknown` | not_provided, missing_data, unspecified | Data insufficient to determine physiological baseline adjustments. |

## occupation  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not specified, missing, n/a | Baseline metrics are treated as population averages due to lack of specific activity context. |
| `sedentary_office` | desk job, white collar, corporate | Characterized by low daily energy expenditure and potential for postural-related HRV suppression. |
| `manual_labor` | blue collar, construction, physical trade | High physical load throughout the day leads to elevated resting heart rate and potential cumulative fatigue. |
| `shift_worker` | night shift, rotating schedule, irregular hours | Circadian misalignment often results in blunted HRV and elevated heart rate during typical sleep windows. |
| `professional_athlete` | elite athlete, pro sports | High vagal tone typically results in a significantly lower resting heart rate and higher baseline HRV. |
| `healthcare_professional` | nurse, doctor, clinical staff | High cognitive and emotional stress combined with irregular physical activity often leads to increased sympathetic dominance. |
| `student` | academic, university | Irregular sleep patterns and high cognitive load can cause significant fluctuations in daily HRV. |
| `retired` | senior, non-working | Baseline metrics are generally more stable but may reflect age-related changes in autonomic nervous system responsiveness. |

## geographic_location  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_specified, missing, n/a | Baseline physiological parameters are indeterminate without environmental context. |
| `temperate` | moderate, mild_climate, continental | Serves as the standard reference for autonomic nervous system regulation without extreme thermal stress. |
| `hot_arid` | desert, dry_heat, xeric | Increased resting heart rate due to peripheral vasodilation and fluid loss for evaporative cooling. |
| `hot_humid` | tropical, equatorial, muggy | Elevated cardiovascular strain as high humidity limits sweat evaporation, increasing core temperature and heart rate. |
| `cold` | arctic, polar, subarctic | Initial increase in heart rate due to sympathetic activation and peripheral vasoconstriction to conserve core heat. |
| `high_altitude` | mountainous, alpine, hypoxic | Persistent elevation in resting heart rate and reduced HRV due to chronic hypoxic stress and increased sympathetic tone. |

## weather  (episode/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_recorded, missing_data, n/a | Baseline physiological state cannot be adjusted for environmental thermal stress. |
| `thermoneutral` | comfortable, mild, temperate, optimal | Minimal thermoregulatory demand; heart rate and HRV reflect resting metabolic baseline. |
| `cold_stress` | chilly, freezing, cold_exposure | Peripheral vasoconstriction and shivering thermogenesis increase metabolic rate and heart rate. |
| `heat_stress` | hot, warm, high_temperature | Vasodilation and increased skin blood flow elevate heart rate to maintain cardiac output and cooling. |
| `high_humidity` | muggy, humid, damp | Reduced evaporative cooling efficiency forces higher heart rate to manage core temperature elevation. |
| `extreme_heatwave` | heat_index_extreme, dangerous_heat | Significant cardiovascular strain; heart rate is elevated due to sustained thermoregulatory demand and potential dehydration. |

## workout_type  (episode/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_recorded, missing, n/a | Baseline physiological state is undefined; no specific heart rate expectation. |
| `steady_state_cardio` | run, jog, cycle, row, elliptical | Characterized by a gradual rise to a stable heart rate plateau with predictable HRV suppression. |
| `high_intensity_interval` | hiit, sprints, circuit_training | Rapid heart rate spikes followed by incomplete recovery periods; significant autonomic nervous system strain. |
| `resistance_training` | weightlifting, strength, bodybuilding, powerlifting | Heart rate response is intermittent and influenced by Valsalva maneuvers and peripheral resistance. |
| `low_impact_steady` | walk, hiking, leisure_cycling | Maintains heart rate near aerobic threshold with minimal HRV volatility compared to high-intensity efforts. |
| `mind_body` | yoga, pilates, stretching, meditation | Typically lowers heart rate and increases HRV through parasympathetic activation and controlled breathing. |
| `aquatic_exercise` | swimming, water_aerobics | Hydrostatic pressure and horizontal body position result in lower resting heart rates compared to land-based equivalents. |

## workout_duration  (episode/numeric_bucketed)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not_recorded, missing_data, undefined | Duration is unavailable; baseline heart rate variability cannot be adjusted for exercise load. |
| `very_short` | quick_session, warm_up_only, brief | Duration is 1-15 minutes; minimal impact on long-term HRV recovery. |
| `short` | standard_short, light_workout | Duration is 16-30 minutes; moderate sympathetic activation with rapid return to baseline. |
| `moderate` | typical_workout, average_session | Duration is 31-60 minutes; significant metabolic demand requiring standard post-exercise recovery. |
| `long` | extended_session, heavy_workout | Duration is 61-120 minutes; sustained sympathetic dominance likely to suppress HRV for several hours. |
| `endurance` | marathon, ultra, prolonged_effort | Duration is 121+ minutes; high risk of autonomic nervous system fatigue and prolonged recovery requirements. |

## workout_location  (episode/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | not specified, n/a, missing | Baseline physiological expectations remain neutral without environmental context. |
| `indoor_gym` | fitness center, weight room, health club | Controlled climate typically minimizes thermoregulatory stress on heart rate. |
| `home_workout` | living room, garage gym, home studio | Familiar environment often correlates with lower psychological stress markers. |
| `outdoor_road` | street, pavement, urban running | Exposure to traffic and uneven surfaces may increase sympathetic nervous system activation. |
| `outdoor_trail` | hiking path, off-road, mountain trail | Variable terrain and elevation changes significantly increase heart rate variability demands. |
| `park` | public green space, track, field | Open air environments generally support stable heart rate recovery compared to urban heat islands. |
| `pool` | swimming pool, natatorium | Hydrostatic pressure and horizontal body position lower resting heart rate compared to land-based exercise. |
| `open_water` | lake, ocean, sea | Cold water immersion and currents increase metabolic cost and heart rate response. |
| `studio` | yoga studio, pilates studio, group class | Structured environments often involve guided breathing which can modulate HRV. |

## sleep  (subject/categorical)  — llm

| value | aliases | physio_note |
|---|---|---|
| `unknown` | missing, no_data, untracked | Baseline metrics cannot be established due to lack of overnight sensor coverage. |
| `restful` | deep_sleep, high_quality, restorative | Characterized by a pronounced nocturnal heart rate dip and high HRV stability, indicating optimal autonomic recovery. |
| `adequate` | normal, sufficient, standard | Expected physiological baseline with a moderate heart rate decline and typical HRV fluctuations throughout the night. |
| `short` | insufficient, sleep_deprived, under_rested | Reduced duration limits the time available for parasympathetic dominance, often resulting in a higher average nocturnal heart rate. |
| `fragmented` | interrupted, restless, broken | Frequent heart rate spikes and HRV drops suggest sympathetic nervous system arousal during the sleep cycle. |
| `irregular` | inconsistent, circadian_mismatch, shift_work | Variable sleep timing prevents the stabilization of circadian rhythms, leading to unpredictable heart rate and HRV baseline shifts. |
