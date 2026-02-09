"""
Health Pattern Intelligence Agent
Complete agentic system for lifestyle analysis from Apple Health data.
Uses Gemini 2.5 Flash Lite for cost-efficient analysis of 360K+ HR records.

Key features:
- Batch statistical processing 
- Location/weather-aware anomaly detection
- GNN-compatible anomaly CSV export for model comparison
- Context-aware prompts handling 21% missing location/weather data
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-genai not installed. Run: pip install google-genai")


class DataProcessor:
    """
    Pre-processes 360K+ HR records into compact statistical summaries.
    Gemini never sees raw data - only aggregated insights.
    This keeps API costs low and responses fast.
    """

    def __init__(self, hr_df: pd.DataFrame, workout_df: Optional[pd.DataFrame] = None):
        self.hr_df = hr_df.copy()
        self.workout_df = workout_df.copy() if workout_df is not None else None
        self._preprocess()

    def _preprocess(self):
        """Parse dates and add derived columns."""
        # Ensure datetime
        for col in ['startDate', 'endDate']:
            if col in self.hr_df.columns and not pd.api.types.is_datetime64_any_dtype(self.hr_df[col]):
                self.hr_df[col] = pd.to_datetime(self.hr_df[col], dayfirst=True, errors='coerce')

        if self.workout_df is not None:
            for col in ['startDate', 'endDate']:
                if col in self.workout_df.columns:
                    self.workout_df[col] = pd.to_datetime(
                        self.workout_df[col], dayfirst=True, errors='coerce'
                    )

        # Derived columns
        df = self.hr_df
        df['hour'] = df['startDate'].dt.hour
        df['date'] = df['startDate'].dt.date
        df['day_of_week'] = df['startDate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['month'] = df['startDate'].dt.to_period('M')
        df['year_month'] = df['startDate'].dt.strftime('%Y-%m')

        # Time period
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[-1, 5, 11, 16, 20, 24],
            labels=['night', 'morning', 'afternoon', 'evening', 'late_night']
        )

        # Activity context (heuristic based on HR zones)
        df['activity_context'] = 'resting'
        df.loc[df['value'] > 100, 'activity_context'] = 'active'
        df.loc[df['value'] > 130, 'activity_context'] = 'exercise'
        df.loc[df['value'] > 160, 'activity_context'] = 'intense'

        # HR zones
        df['hr_zone'] = pd.cut(
            df['value'],
            bins=[0, 60, 80, 100, 130, 160, 250],
            labels=['very_low', 'low', 'moderate', 'elevated', 'high', 'very_high']
        )

        # Temperature bins (where available)
        if 'temperature_c' in df.columns:
            df['temp_bin'] = pd.cut(
                df['temperature_c'],
                bins=[-20, 5, 15, 25, 35, 50],
                labels=['cold', 'cool', 'mild', 'warm', 'hot']
            )

    def get_data_quality_summary(self) -> Dict:
        """Compute data quality metrics for prompt context."""
        df = self.hr_df
        total = len(df)

        quality = {
            'total_records': total,
            'date_range': f"{df['startDate'].min().strftime('%Y-%m-%d')} to {df['startDate'].max().strftime('%Y-%m-%d')}",
            'location_coverage': round(df['location_name'].notna().sum() / total * 100, 1) if 'location_name' in df.columns else 0,
            'weather_coverage': round(df['temperature_c'].notna().sum() / total * 100, 1) if 'temperature_c' in df.columns else 0,
            'gps_confirmed': round(
                (df['location_confidence'] == 'gps_confirmed').sum() / total * 100, 1
            ) if 'location_confidence' in df.columns else 0,
            'records_missing_location': df['location_name'].isna().sum() if 'location_name' in df.columns else total,
            'records_missing_weather': df['temperature_c'].isna().sum() if 'temperature_c' in df.columns else total,
        }
        return quality

    def get_overall_summary(self) -> str:
        """Compact overall HR statistics."""
        df = self.hr_df
        summary_lines = [
            f"Total HR records: {len(df):,}",
            f"Date range: {df['startDate'].min().strftime('%Y-%m-%d')} → {df['startDate'].max().strftime('%Y-%m-%d')}",
            f"Overall HR: mean={df['value'].mean():.1f}, median={df['value'].median():.1f}, "
            f"std={df['value'].std():.1f}, min={df['value'].min():.0f}, max={df['value'].max():.0f}",
            f"\nHR by time period:",
        ]

        for period in ['night', 'morning', 'afternoon', 'evening']:
            subset = df[df['time_period'] == period]
            if len(subset) > 0:
                summary_lines.append(
                    f"  {period}: mean={subset['value'].mean():.1f}, "
                    f"std={subset['value'].std():.1f}, n={len(subset):,}"
                )

        summary_lines.append(f"\nHR by activity context:")
        for ctx in ['resting', 'active', 'exercise', 'intense']:
            subset = df[df['activity_context'] == ctx]
            if len(subset) > 0:
                summary_lines.append(
                    f"  {ctx}: mean={subset['value'].mean():.1f}, n={len(subset):,} "
                    f"({len(subset)/len(df)*100:.1f}%)"
                )

        # Weekend vs weekday
        wd = df[~df['is_weekend']]['value']
        we = df[df['is_weekend']]['value']
        summary_lines.append(
            f"\nWeekday mean: {wd.mean():.1f} | Weekend mean: {we.mean():.1f} "
            f"(diff: {we.mean() - wd.mean():+.1f})"
        )

        return "\n".join(summary_lines)

    def get_location_summary(self) -> str:
        """HR statistics stratified by location."""
        df = self.hr_df
        if 'location_name' not in df.columns:
            return "No location data available."

        lines = ["## HR by Location\n"]

        # Top locations
        loc_counts = df['location_name'].value_counts().head(10)
        for loc, count in loc_counts.items():
            subset = df[df['location_name'] == loc]
            pct = count / len(df) * 100

            loc_line = (
                f"**{loc}** (n={count:,}, {pct:.1f}%): "
                f"mean={subset['value'].mean():.1f}, median={subset['value'].median():.1f}, "
                f"std={subset['value'].std():.1f}"
            )

            # Add weather context if available
            if 'temperature_c' in subset.columns and subset['temperature_c'].notna().sum() > 0:
                temp_mean = subset['temperature_c'].mean()
                humidity_mean = subset['humidity_pct'].mean() if 'humidity_pct' in subset.columns else None
                weather_str = f", avg temp={temp_mean:.1f}°C"
                if humidity_mean and not np.isnan(humidity_mean):
                    weather_str += f", humidity={humidity_mean:.0f}%"
                loc_line += weather_str

            lines.append(loc_line)

        # Resting HR by location (HR < 85 and not during typical exercise times)
        resting = df[(df['value'] < 85) & (df['activity_context'] == 'resting')]
        if len(resting) > 0 and 'location_name' in resting.columns:
            lines.append("\n## Resting HR by Location (HR < 85, non-exercise)")
            for loc in resting['location_name'].value_counts().head(5).index:
                sub = resting[resting['location_name'] == loc]
                lines.append(
                    f"  {loc}: resting mean={sub['value'].mean():.1f}, n={len(sub):,}"
                )

        # Missing location stats
        missing = df['location_name'].isna().sum()
        lines.append(f"\n⚠️ Records without location: {missing:,} ({missing/len(df)*100:.1f}%)")

        return "\n".join(lines)

    def get_weather_summary(self) -> str:
        """HR statistics stratified by weather conditions."""
        df = self.hr_df
        if 'temperature_c' not in df.columns or df['temperature_c'].notna().sum() == 0:
            return "No weather data available."

        has_weather = df[df['temperature_c'].notna()]
        lines = [f"## HR by Weather (n={len(has_weather):,} records with weather)\n"]

        # By temperature bin
        if 'temp_bin' in has_weather.columns:
            lines.append("### By Temperature Range:")
            for temp_bin in ['cold', 'cool', 'mild', 'warm', 'hot']:
                subset = has_weather[has_weather['temp_bin'] == temp_bin]
                if len(subset) > 0:
                    lines.append(
                        f"  {temp_bin}: HR mean={subset['value'].mean():.1f}, "
                        f"std={subset['value'].std():.1f}, n={len(subset):,}"
                    )

        # Resting HR by temperature
        resting_weather = has_weather[has_weather['activity_context'] == 'resting']
        if len(resting_weather) > 0 and 'temp_bin' in resting_weather.columns:
            lines.append("\n### Resting HR by Temperature:")
            for temp_bin in ['cold', 'cool', 'mild', 'warm', 'hot']:
                subset = resting_weather[resting_weather['temp_bin'] == temp_bin]
                if len(subset) > 0:
                    lines.append(
                        f"  {temp_bin}: resting HR={subset['value'].mean():.1f}, n={len(subset):,}"
                    )

        # By weather description (top 5)
        if 'weather_description' in has_weather.columns:
            top_weather = has_weather['weather_description'].value_counts().head(5)
            lines.append("\n### By Weather Condition:")
            for cond, count in top_weather.items():
                if pd.notna(cond) and cond:
                    subset = has_weather[has_weather['weather_description'] == cond]
                    lines.append(
                        f"  {cond}: HR mean={subset['value'].mean():.1f}, n={count:,}"
                    )

        missing = df['temperature_c'].isna().sum()
        lines.append(f"\n⚠️ Records without weather: {missing:,} ({missing/len(df)*100:.1f}%)")

        return "\n".join(lines)

    def get_workout_summary(self) -> str:
        """Workout statistics."""
        if self.workout_df is None or len(self.workout_df) == 0:
            return "No workout data available."

        wk = self.workout_df
        lines = [
            f"## Workout Summary (n={len(wk):,})\n",
            f"Date range: {wk['startDate'].min().strftime('%Y-%m-%d')} → {wk['startDate'].max().strftime('%Y-%m-%d')}"
        ]

        # By type
        for wtype in wk['workoutType'].value_counts().head(8).index:
            subset = wk[wk['workoutType'] == wtype]
            duration_avg = subset['duration'].mean()
            lines.append(
                f"  {wtype}: n={len(subset)}, avg duration={duration_avg:.0f}min, "
                f"avg distance={subset['totalDistance'].mean():.1f}, "
                f"avg calories={subset['totalEnergyBurned'].mean():.0f}"
            )

        return "\n".join(lines)

    def get_temporal_trends(self) -> str:
        """Monthly trends for pattern detection."""
        df = self.hr_df
        monthly = df.groupby('year_month').agg(
            hr_mean=('value', 'mean'),
            hr_std=('value', 'std'),
            hr_median=('value', 'median'),
            count=('value', 'count')
        ).reset_index()

        lines = ["## Monthly HR Trends\n"]
        for _, row in monthly.iterrows():
            lines.append(
                f"  {row['year_month']}: mean={row['hr_mean']:.1f}, "
                f"std={row['hr_std']:.1f}, n={row['count']:,}"
            )

        return "\n".join(lines)

    def _tag_workout_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag records that fall within workout time windows.
        These are excluded from anomaly detection (elevated HR is expected).
        """
        df['is_during_workout'] = False

        if self.workout_df is None or len(self.workout_df) == 0:
            return df

        # For each workout, tag HR records within [start - 5min, end + 10min]
        # Buffer accounts for warmup/cooldown HR readings
        for _, wk in self.workout_df.iterrows():
            wk_start = wk['startDate'] - pd.Timedelta(minutes=5)
            wk_end = wk['endDate'] + pd.Timedelta(minutes=10)
            mask = (df['startDate'] >= wk_start) & (df['startDate'] <= wk_end)
            df.loc[mask, 'is_during_workout'] = True

        n_workout = df['is_during_workout'].sum()
        print(f"  🏋️ Workout-period records tagged: {n_workout:,} "
              f"({n_workout/len(df)*100:.1f}%) — excluded from anomaly scoring")
        return df

    def _compute_context_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute context-stratified baselines.
        Each record is scored against peers in the same (time_period × activity_context).
        Optional: further stratify by location if coverage is sufficient.
        """
        # Primary context: time_period × activity_context
        # This separates "resting at night" from "active in afternoon"
        context_col = 'context_group'
        df[context_col] = df['time_period'].astype(str) + '_' + df['activity_context'].astype(str)

        # Compute group stats (exclude workout records for clean baselines)
        non_workout = df[~df['is_during_workout']]
        group_stats = non_workout.groupby(context_col)['value'].agg(
            ['mean', 'std', 'median', 'count']
        ).rename(columns={'mean': 'ctx_mean', 'std': 'ctx_std',
                          'median': 'ctx_median', 'count': 'ctx_count'})

        # Require minimum 50 records for reliable stats
        group_stats.loc[group_stats['ctx_count'] < 50, 'ctx_std'] = np.nan

        df = df.merge(group_stats[['ctx_mean', 'ctx_std']], left_on=context_col,
                       right_index=True, how='left')

        # Context Z-score: how unusual is this HR for this context?
        df['z_context'] = np.where(
            df['ctx_std'].notna() & (df['ctx_std'] > 0),
            (df['value'] - df['ctx_mean']) / df['ctx_std'],
            np.nan
        )

        # Location-stratified Z-score (where location data exists)
        df['z_location'] = np.nan
        if 'location_name' in df.columns:
            loc_context = df['time_period'].astype(str) + '_' + df['location_name'].astype(str)
            loc_stats = non_workout[non_workout['location_name'].notna()].groupby(
                non_workout['time_period'].astype(str) + '_' + non_workout['location_name'].astype(str)
            )['value'].agg(['mean', 'std', 'count'])

            loc_stats = loc_stats[loc_stats['count'] >= 30]  # min samples

            for grp in loc_stats.index:
                mask = (loc_context == grp) & (~df['is_during_workout'])
                if mask.sum() > 0 and loc_stats.loc[grp, 'std'] > 0:
                    df.loc[mask, 'z_location'] = (
                        (df.loc[mask, 'value'] - loc_stats.loc[grp, 'mean']) /
                        loc_stats.loc[grp, 'std']
                    )

        # Rolling personal baseline (30-day window per record)
        df = df.sort_values('startDate')
        rolling_mean = df['value'].rolling(window=500, min_periods=50, center=True).mean()
        rolling_std = df['value'].rolling(window=500, min_periods=50, center=True).std()
        df['z_rolling'] = np.where(
            rolling_std > 0,
            (df['value'] - rolling_mean) / rolling_std,
            np.nan
        )

        # Cleanup temp columns
        df.drop(columns=[context_col, 'ctx_mean', 'ctx_std'], inplace=True)

        return df

    def get_record_level_anomalies(self, z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Record-level anomaly detection with:
        1. Workout exclusion (don't flag exercise HR)
        2. Context-stratified Z-scores (time_period × activity)
        3. Location-aware baselines
        4. Rolling personal baseline
        5. Bradycardia-sensitive scoring

        Target anomaly rate: ~2-5%
        """
        print("\n⚡ Computing record-level anomalies...")
        df = self.hr_df.copy()

        # Step 1: Tag workout periods
        df = self._tag_workout_periods(df)

        # Step 2: Compute context baselines and Z-scores
        print("  📊 Computing context-stratified baselines...")
        df = self._compute_context_baselines(df)

        # Step 3: Composite anomaly score
        # Weighted combination of available Z-scores
        z_cols = ['z_context', 'z_location', 'z_rolling']
        weights = [0.45, 0.25, 0.30]  # context heaviest, location bonus

        z_matrix = df[z_cols].copy()
        w_matrix = pd.DataFrame(
            np.where(z_matrix.notna(), weights, 0),
            columns=z_cols, index=df.index
        )

        # Signed raw score (negative = low HR, positive = high HR)
        weighted_sum = (z_matrix.fillna(0) * w_matrix).sum(axis=1)
        weight_total = w_matrix.sum(axis=1).replace(0, np.nan)
        df['z_composite'] = weighted_sum / weight_total

        # Absolute anomaly score normalized to 0-1
        abs_composite = df['z_composite'].abs()
        p99 = abs_composite.quantile(0.99)
        df['anomaly_score'] = (abs_composite / p99).clip(0, 1).fillna(0)

        # Step 4: Bradycardia boost
        # Low HR is clinically more significant — boost score for very low HR
        bradycardia_mask = (df['value'] < 55) & (~df['is_during_workout'])
        df.loc[bradycardia_mask, 'anomaly_score'] = df.loc[bradycardia_mask, 'anomaly_score'].clip(lower=0.6)

        # Step 5: Binary anomaly flag
        # Workout records are NEVER flagged as anomalies
        df['is_anomaly'] = 0

        # Flag non-workout records exceeding threshold
        non_workout_mask = ~df['is_during_workout']
        df.loc[non_workout_mask, 'is_anomaly'] = (
            (df.loc[non_workout_mask, 'z_composite'].abs() > z_threshold) |
            (df.loc[non_workout_mask, 'value'] < 50)  # Absolute bradycardia
        ).astype(int)

        # Step 6: Infer causes
        df['likely_causes'] = df.apply(self._infer_anomaly_cause, axis=1)

        # Stats
        total_anomalies = df['is_anomaly'].sum()
        anomaly_rate = total_anomalies / len(df) * 100
        print(f"  ✅ Anomaly detection complete:")
        print(f"     Total anomalies: {total_anomalies:,} / {len(df):,} ({anomaly_rate:.2f}%)")
        print(f"     Low HR anomalies: {(df['is_anomaly'] == 1).sum() - ((df['z_composite'] > z_threshold) & non_workout_mask).sum():,}")
        print(f"     High HR anomalies (non-workout): {((df['z_composite'] > z_threshold) & non_workout_mask).sum():,}")

        self._record_anomalies = df
        return df

    def get_daily_anomaly_candidates(self, z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Daily aggregation of record-level anomalies (for Gemini summary context).
        Uses record-level scores, not daily means.
        """
        # Ensure record-level anomalies are computed
        if not hasattr(self, '_record_anomalies') or self._record_anomalies is None:
            self.get_record_level_anomalies(z_threshold)

        df = self._record_anomalies

        daily = df.groupby('date').agg(
            hr_mean=('value', 'mean'),
            hr_std=('value', 'std'),
            hr_median=('value', 'median'),
            hr_min=('value', 'min'),
            hr_max=('value', 'max'),
            hr_range=('value', lambda x: x.max() - x.min()),
            record_count=('value', 'count'),
            anomaly_count=('is_anomaly', 'sum'),
            anomaly_rate=('is_anomaly', 'mean'),
            anomaly_score_mean=('anomaly_score', 'mean'),
            anomaly_score_max=('anomaly_score', 'max'),
            workout_records=('is_during_workout', 'sum'),
        ).reset_index()

        daily['is_anomaly'] = (daily['anomaly_rate'] > 0.15).astype(int)  # >15% of daily records flagged

        # Location context
        if 'location_name' in df.columns:
            daily_loc = df.groupby('date')['location_name'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            )
            daily['primary_location'] = daily['date'].map(daily_loc)

        if 'temperature_c' in df.columns:
            daily['avg_temp_c'] = df.groupby('date')['temperature_c'].mean()

        if 'humidity_pct' in df.columns:
            daily['avg_humidity'] = df.groupby('date')['humidity_pct'].mean()

        daily['has_workout'] = daily['workout_records'] > 0

        return daily

    def _infer_anomaly_cause(self, row) -> str:
        """Record-level heuristic cause inference."""
        if row.get('is_anomaly', 0) == 0:
            return ''

        causes = []

        # HR direction
        z = row.get('z_composite', 0)
        hr = row.get('value', 0)

        if hr < 50:
            causes.append('severe_bradycardia')
        elif hr < 55:
            causes.append('bradycardia')
        elif z < -2.5:
            causes.append('unusually_low_hr')

        if z > 2.5 and not row.get('is_during_workout', False):
            causes.append('elevated_hr_at_rest')

        # Time context
        tp = row.get('time_period', '')
        if tp == 'night' and hr > 90:
            causes.append('elevated_nocturnal_hr')
        if tp == 'night' and hr < 45:
            causes.append('nocturnal_bradycardia')

        # Temperature context
        temp = row.get('temperature_c', np.nan) if 'temperature_c' in row.index else np.nan
        if pd.notna(temp):
            if temp > 35 and z > 2:
                causes.append('heat_stress')
            elif temp < 5 and z < -2:
                causes.append('cold_induced')

        # Location anomaly
        z_loc = row.get('z_location', np.nan)
        if pd.notna(z_loc) and abs(z_loc) > 2.5:
            causes.append('unusual_for_location')

        return '; '.join(causes) if causes else 'context_deviation'

    def export_anomalies_gnn_format(self, output_path: str) -> pd.DataFrame:
        """
        Export RECORD-LEVEL anomaly data in GNN-compatible schema.
        Each record has its own anomaly_score (not daily aggregate).
        """
        # Ensure record-level anomalies are computed
        if not hasattr(self, '_record_anomalies') or self._record_anomalies is None:
            self.get_record_level_anomalies()

        df = self._record_anomalies

        gnn_df = pd.DataFrame({
            'timestamp': df['startDate'].dt.strftime('%d/%m/%Y %H:%M'),
            'hr_value': df['value'],
            'predicted_anomaly': df['is_anomaly'].astype(int),
            'anomaly_score': df['anomaly_score'].round(6),
            'activity_context': df['activity_context'],
            'hr_zone': df['hr_zone'].astype(str),
            'time_period': df['time_period'].astype(str),
            'hour': df['hour'],
            'day_of_week': df['day_of_week'],
            'is_weekend': df['is_weekend'],
            'type': df['type'] if 'type' in df.columns else 'HeartRate',
        })

        gnn_df.to_csv(output_path, index=False)

        # Print summary
        n_anom = gnn_df['predicted_anomaly'].sum()
        rate = n_anom / len(gnn_df) * 100
        print(f"✅ Exported GNN-compatible anomalies → {output_path}")
        print(f"   Records: {len(gnn_df):,} | Anomalies: {n_anom:,} ({rate:.2f}%)")
        return gnn_df

    def build_analysis_context(self, analysis_type: str) -> str:
        """
        Build compact data context for Gemini prompt.
        This is the key batch-processing step: 360K records → ~2K tokens of summary.
        """
        quality = self.get_data_quality_summary()

        sections = {
            'baseline': [
                self.get_overall_summary(),
                self.get_location_summary(),
                self.get_weather_summary(),
                self.get_temporal_trends(),
            ],
            'patterns': [
                self.get_overall_summary(),
                self.get_location_summary(),
                self.get_temporal_trends(),
            ],
            'workout_performance': [
                self.get_workout_summary(),
                self.get_weather_summary(),
                self.get_location_summary(),
            ],
            'recommendations': [
                self.get_overall_summary(),
                self.get_location_summary(),
                self.get_weather_summary(),
                self.get_workout_summary(),
            ],
            'anomalies': [
                self._get_anomaly_summary(),
                self.get_location_summary(),
                self.get_weather_summary(),
            ],
        }

        selected = sections.get(analysis_type, sections['baseline'])
        data_context = "\n\n".join(selected)

        # Data quality disclaimer
        quality_note = (
            f"\n## Data Quality Notes\n"
            f"- Total records: {quality['total_records']:,}\n"
            f"- Date range: {quality['date_range']}\n"
            f"- Location coverage: {quality['location_coverage']}% "
            f"({quality['records_missing_location']:,} records missing)\n"
            f"- Weather coverage: {quality['weather_coverage']}% "
            f"({quality['records_missing_weather']:,} records missing)\n"
            f"- GPS confirmed: {quality['gps_confirmed']}%\n"
            f"- ⚠️ ~21% of data lacks location/weather context. "
            f"Conclusions involving location/weather should note this limitation."
        )

        return data_context + quality_note

    def _get_anomaly_summary(self) -> str:
        """Summary of detected anomalies for Gemini context (record-level)."""
        if not hasattr(self, '_record_anomalies') or self._record_anomalies is None:
            self.get_record_level_anomalies()

        df = self._record_anomalies
        anomalies = df[df['is_anomaly'] == 1]
        non_workout = df[~df['is_during_workout']]

        lines = [
            f"## Anomaly Detection Summary (Record-Level, Workout-Excluded)",
            f"Total records analyzed: {len(df):,}",
            f"Workout-period records (excluded): {df['is_during_workout'].sum():,}",
            f"Non-workout records scored: {len(non_workout):,}",
            f"Anomalous records: {len(anomalies):,} ({len(anomalies)/len(df)*100:.2f}%)\n",
        ]

        if len(anomalies) > 0:
            # HR distribution of anomalies
            lines.append("### Anomaly HR Distribution:")
            lines.append(
                f"  Mean={anomalies['value'].mean():.1f}, Median={anomalies['value'].median():.1f}, "
                f"Std={anomalies['value'].std():.1f}"
            )
            lines.append(f"  Low HR anomalies (< 60): {(anomalies['value'] < 60).sum():,}")
            lines.append(f"  High HR anomalies (> 100, non-workout): {(anomalies['value'] > 100).sum():,}")

            # By time period
            lines.append("\n### Anomalies by Time Period:")
            tp_counts = anomalies['time_period'].value_counts()
            for tp, cnt in tp_counts.items():
                lines.append(f"  {tp}: {cnt:,}")

            # Top daily anomaly concentrations
            daily = self.get_daily_anomaly_candidates()
            top_days = daily.nlargest(10, 'anomaly_count')
            lines.append("\n### Top 10 Most Anomalous Days:")
            for _, row in top_days.iterrows():
                loc = row.get('primary_location', 'unknown')
                temp = f"{row['avg_temp_c']:.0f}°C" if pd.notna(row.get('avg_temp_c')) else 'N/A'
                lines.append(
                    f"  {row['date']}: {row['anomaly_count']:.0f} anomalies / {row['record_count']:.0f} records "
                    f"({row['anomaly_rate']*100:.0f}%), HR mean={row['hr_mean']:.1f}, "
                    f"location={loc}, temp={temp}"
                )

            # Cause distribution
            all_causes = '; '.join(anomalies['likely_causes'].dropna()).split('; ')
            cause_counts = pd.Series(all_causes).value_counts()
            lines.append("\n### Anomaly Cause Distribution:")
            for cause, count in cause_counts.head(10).items():
                if cause:
                    lines.append(f"  {cause}: {count:,}")

        return "\n".join(lines)


class HealthPatternAgent:
    """
    Main agent class. Orchestrates data processing and Gemini analysis.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package required. Run: pip install google-genai")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.processor: Optional[DataProcessor] = None
        self.hr_df: Optional[pd.DataFrame] = None
        self.workout_df: Optional[pd.DataFrame] = None
        self.anomaly_df: Optional[pd.DataFrame] = None

        # Import prompts
        from agent_prompts_enhanced import SYSTEM_PROMPT, ANALYSIS_PROMPTS
        self.system_prompt = SYSTEM_PROMPT
        self.analysis_prompts = ANALYSIS_PROMPTS

        print(f"✅ Agent initialized with {model}")

    def load_data(self, hr_file: str, workout_file: str = None, anomaly_file: str = None):
        """Load CSV data files."""
        print("\n⏳ Loading data...")

        # Heart rate
        self.hr_df = pd.read_csv(hr_file)
        self.hr_df['startDate'] = pd.to_datetime(self.hr_df['startDate'], dayfirst=True, errors='coerce')
        self.hr_df['endDate'] = pd.to_datetime(self.hr_df['endDate'], dayfirst=True, errors='coerce')
        print(f"  ❤️ HR data: {len(self.hr_df):,} records")

        # Workouts
        if workout_file and os.path.exists(workout_file):
            self.workout_df = pd.read_csv(workout_file)
            self.workout_df['startDate'] = pd.to_datetime(
                self.workout_df['startDate'], dayfirst=True, errors='coerce'
            )
            self.workout_df['endDate'] = pd.to_datetime(
                self.workout_df['endDate'], dayfirst=True, errors='coerce'
            )
            print(f"  🏋️ Workout data: {len(self.workout_df):,} records")

        # Anomaly (pre-existing)
        if anomaly_file and os.path.exists(anomaly_file):
            self.anomaly_df = pd.read_csv(anomaly_file)
            print(f"  ⚡ Anomaly data: {len(self.anomaly_df):,} records")

        # Initialize processor
        self.processor = DataProcessor(self.hr_df, self.workout_df)
        print("✅ Data loaded and processor initialized")

    def _call_gemini(self, prompt: str, max_tokens: int = 8192) -> str:
        """Call Gemini API with retry logic."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=0.3,  # Lower for analytical consistency
                ),
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            # Retry once with smaller context
            try:
                print("🔄 Retrying with condensed context...")
                condensed = prompt[:len(prompt)//2] + "\n\n[Context truncated for token limit]\n\n" + prompt[-2000:]
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=condensed,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                    ),
                )
                return response.text
            except Exception as e2:
                return f"❌ API failed after retry: {e2}"

    def analyze_with_gemini(self, analysis_type: str) -> str:
        """
        Run a specific analysis type through Gemini.
        Data is pre-aggregated by DataProcessor to keep costs low.
        """
        if self.processor is None:
            print("❌ No data loaded. Call load_data() first.")
            return ""

        print(f"\n🔍 Running {analysis_type} analysis...")
        print("  📊 Aggregating data summaries (batch processing)...")

        # Build compact context from 360K records
        data_context = self.processor.build_analysis_context(analysis_type)

        # Get analysis prompt template
        analysis_prompt = self.analysis_prompts.get(analysis_type, self.analysis_prompts['baseline'])

        # Combine
        full_prompt = f"""
# DATA CONTEXT (Pre-aggregated from {len(self.hr_df):,} records)

{data_context}

---

{analysis_prompt}
"""

        print(f"  📤 Sending to Gemini ({self.model})...")
        print(f"  📝 Prompt size: ~{len(full_prompt)} chars")

        result = self._call_gemini(full_prompt)

        print(f"\n{'='*80}")
        print(f"📋 {analysis_type.upper()} ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(result)
        print(f"{'='*80}\n")

        return result

    def analyze_anomalies(self) -> str:
        """Run anomaly-specific analysis with root cause investigation."""
        if self.processor is None:
            print("❌ No data loaded.")
            return ""

        print("\n🔍 Running anomaly analysis with root cause investigation...")

        # Get anomaly context
        data_context = self.processor.build_analysis_context('anomalies')

        anomaly_prompt = """
# Anomaly Root Cause Analysis

You are investigating heart rate anomalies detected in the user's Apple Health data.
For each anomalous period, provide:

## 1. ANOMALY CLASSIFICATION
- Statistical anomalies (unusual HR for time/context)
- Physiological anomalies (abnormal patterns suggesting health events)
- Environmental anomalies (weather/location-driven HR changes)
- Behavioral anomalies (lifestyle changes, travel, stress)

## 2. ROOT CAUSE ANALYSIS
For each major anomaly cluster:
- What was the likely trigger?
- Was it location/weather related?
- Was it workout-related?
- Could it indicate a health concern?
- Is it a data quality issue (sensor artifact)?

## 3. COMPARISON FRAMEWORK
Structure your findings so they can be compared with a GNN-based anomaly detection model:
- Which anomalies are likely TRUE positives (real health events)?
- Which are likely FALSE positives (sensor noise, expected variation)?
- What patterns might a GNN model miss that statistical methods catch?
- What patterns might GNN catch that statistics miss?

## 4. ACTIONABLE INSIGHTS
- Anomalies requiring medical attention
- Anomalies suggesting lifestyle adjustments
- Anomalies that are benign (expected given context)
- Data quality improvements to reduce false positives

Provide specific dates, HR values, and context for each finding.
"""

        full_prompt = f"""
# DATA CONTEXT

{data_context}

---

{anomaly_prompt}
"""

        result = self._call_gemini(full_prompt)

        print(f"\n{'='*80}")
        print(f"⚡ ANOMALY ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(result)
        print(f"{'='*80}\n")

        return result

    def ask_question(self, question: str) -> str:
        """Free-form question with full data context."""
        if self.processor is None:
            print("❌ No data loaded.")
            return ""

        print(f"\n💬 Processing question: {question[:80]}...")

        # Use baseline context for general questions
        data_context = self.processor.build_analysis_context('baseline')

        prompt = f"""
# DATA CONTEXT

{data_context}

---

# USER QUESTION
{question}

Provide a data-driven answer using the statistics above.
Cite specific numbers and patterns. Note data quality limitations where relevant.
"""

        result = self._call_gemini(prompt)

        print(f"\n{'='*80}")
        print(f"💬 ANSWER")
        print(f"{'='*80}")
        print(result)
        print(f"{'='*80}\n")

        return result

    def generate_comprehensive_report(self, output_dir: str = "reports") -> str:
        """Generate full analysis report across all dimensions."""
        if self.processor is None:
            print("❌ No data loaded.")
            return ""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print("\n🔬 GENERATING COMPREHENSIVE HEALTH REPORT")
        print("This will make 4-5 Gemini API calls...\n")

        report_sections = {}
        for section in ['baseline', 'patterns', 'workout_performance', 'recommendations']:
            print(f"  📊 Analyzing: {section}...")
            report_sections[section] = self.analyze_with_gemini(section)

        # Anomaly analysis
        print(f"  ⚡ Analyzing: anomalies...")
        report_sections['anomalies'] = self.analyze_anomalies()

        # Compile report
        report = f"""# 🏥 Health Pattern Intelligence Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data: {len(self.hr_df):,} heart rate records

---

## 1. Baseline Health Profile
{report_sections.get('baseline', 'N/A')}

---

## 2. Behavioral Patterns
{report_sections.get('patterns', 'N/A')}

---

## 3. Workout Performance
{report_sections.get('workout_performance', 'N/A')}

---

## 4. Anomaly Investigation
{report_sections.get('anomalies', 'N/A')}

---

## 5. Personalized Recommendations
{report_sections.get('recommendations', 'N/A')}

---

*Report generated by Health Pattern Intelligence Agent*
*Model: {self.model}*
"""

        # Save
        report_file = os.path.join(output_dir, f"health_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n💾 Report saved: {report_file}")

        return report

    def export_anomalies_for_comparison(self, output_path: str = "anomaly_agent_output.csv"):
        """Export anomalies in GNN-compatible format for model comparison."""
        if self.processor is None:
            print("❌ No data loaded.")
            return

        # Auto-create parent directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print("\n📤 Exporting anomalies in GNN-compatible format...")
        self.processor.export_anomalies_gnn_format(output_path)

        # Also export daily summary
        daily_path = output_path.replace('.csv', '_daily.csv')
        daily = self.processor.get_daily_anomaly_candidates()
        daily.to_csv(daily_path, index=False)
        print(f"✅ Daily anomaly summary → {daily_path} ({len(daily):,} days)")