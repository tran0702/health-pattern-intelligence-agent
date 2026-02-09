"""
Gemini-Powered Autonomous Anomaly Detection Agent
3-Stage: Data Cleaning → Baseline Learning → Window Detection → Precise Identification
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Install: pip install google-generativeai")
    exit(1)


class GeminiAnomalyAgent:
    """
    Autonomous anomaly detection using Gemini AI.
    No hardcoded anomaly rules - learns baseline and identifies outliers autonomously.
    """
    
    def __init__(self, gemini_api_key: str = None):
        self.api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        self.hr_df = None
        self.workout_df = None
        self.baseline = None
        self.anomaly_windows = []
        self.anomalies = []
        
        print("✅ Gemini Anomaly Agent initialized")
    
    # ============================================================================
    # DATA LOADING & CLEANING
    # ============================================================================
    
    def load_and_clean_data(self, hr_csv: str, workout_csv: str):
        """Load and clean HR + workout data."""
        print("\n📂 Loading data...")
        
        hr_raw = pd.read_csv(hr_csv, parse_dates=['startDate', 'endDate'])
        workout_raw = pd.read_csv(workout_csv, parse_dates=['startDate', 'endDate'])
        
        print(f"   • HR raw: {len(hr_raw):,} records")
        print(f"   • Workouts raw: {len(workout_raw):,} sessions")
        
        # Clean data
        self.hr_df = self._clean_heart_rate(hr_raw)
        self.workout_df = self._clean_workouts(workout_raw)
        
        print(f"\n✅ Data cleaned")
        print(f"   • HR final: {len(self.hr_df):,} ({len(hr_raw)-len(self.hr_df):,} removed)")
        print(f"   • Workouts final: {len(self.workout_df):,} ({len(workout_raw)-len(self.workout_df):,} removed)")
        print(f"   • Range: {self.hr_df['startDate'].min().date()} to {self.hr_df['startDate'].max().date()}")
        
        return self
    
    def _clean_heart_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean HR data: remove outliers, duplicates, flag statistical anomalies."""
        print("\n🧹 Cleaning HR...")
        
        initial = len(df)
        
        # 1. Physiological outliers (<40, >200 bpm)
        df = df[(df['value'] >= 40) & (df['value'] <= 200)].copy()
        print(f"   • Removed {initial - len(df):,} physiological outliers")
        
        # 2. Duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['startDate', 'value'], keep='first')
        print(f"   • Removed {before_dedup - len(df):,} duplicates")
        
        # 3. Statistical outliers (3×IQR method)
        Q1, Q3 = df['value'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        
        df['is_statistical_outlier'] = (df['value'] < lower) | (df['value'] > upper)
        n_outliers = df['is_statistical_outlier'].sum()
        print(f"   • Flagged {n_outliers:,} statistical outliers (3×IQR: [{lower:.1f}, {upper:.1f}])")
        
        return df.sort_values('startDate').reset_index(drop=True)
    
    def _clean_workouts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean workout data: remove short/invalid workouts, duplicates."""
        print("\n🧹 Cleaning workouts...")
        
        initial = len(df)
        
        # 1. Remove <5 minutes
        df = df[df['duration'] >= 300].copy()
        print(f"   • Removed {initial - len(df):,} workouts <5 min")
        
        # 2. Flag zero-distance for distance activities
        distance_activities = ['Running', 'Walking', 'Cycling', 'Hiking', 'Swimming']
        df['is_distance_activity'] = df['workoutType'].isin(distance_activities)
        df['zero_distance_flag'] = (df['is_distance_activity']) & (df['totalDistance'] == 0)
        n_zero = df['zero_distance_flag'].sum()
        print(f"   • Flagged {n_zero:,} distance activities with zero distance")
        
        # 3. Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['startDate', 'workoutType', 'duration'], keep='first')
        print(f"   • Removed {before_dedup - len(df):,} duplicates")
        
        # 4. Flag suspicious energy/distance ratios
        df['energy_per_km'] = np.where(
            df['totalDistance'] > 0,
            df['totalEnergyBurned'] / (df['totalDistance'] / 1000),
            np.nan
        )
        suspicious = ((df['energy_per_km'] < 30) | (df['energy_per_km'] > 200)) & df['is_distance_activity']
        df['suspicious_energy_ratio'] = suspicious
        print(f"   • Flagged {suspicious.sum():,} suspicious energy/distance ratios")
        
        return df.sort_values('startDate').reset_index(drop=True)
    
    # ============================================================================
    # GEMINI API
    # ============================================================================
    
    def _call_gemini(self, prompt: str, temperature: float = 0.3) -> str:
        """Call Gemini with error handling."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'top_p': 0.95,
                    'max_output_tokens': 8192,
                }
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return None
    
    # ============================================================================
    # STAGE 1: BASELINE LEARNING
    # ============================================================================
    
    def learn_baseline(self) -> Dict:
        """Stage 1: Learn user's normal HR patterns and lifestyle baseline."""
        print("\n" + "="*80)
        print("🧠 STAGE 1: LEARNING PERSONAL BASELINE")
        print("="*80)
        
        stats = self._prepare_baseline_stats()
        
        prompt = f"""You are analyzing Apple Watch data to learn a person's NORMAL cardiovascular baseline.

**DATA:**
{json.dumps(stats, indent=2)}

**TASK:** Create baseline profile for anomaly detection.

**ANALYZE:**
1. Normal HR ranges (resting, active, exercise)
2. Activity patterns (workout frequency, types, intensity)
3. Temporal patterns (daily/weekly rhythms)
4. Fitness indicators (cardio fitness, recovery, stability)

**OUTPUT JSON (no markdown):**
{{
  "resting_hr": {{"range": [min, max], "mean": X, "context": "..."}},
  "active_hr": {{"range": [min, max], "mean": X, "context": "..."}},
  "exercise_hr": {{"range": [min, max], "mean": X, "by_type": {{}}, "context": "..."}},
  "activity_profile": {{
    "workout_frequency_per_week": X,
    "preferred_activities": [],
    "typical_duration_min": X,
    "rest_pattern": "..."
  }},
  "temporal_patterns": {{
    "daily_rhythm": "...",
    "peak_hr_hours": [],
    "low_hr_hours": []
  }},
  "fitness_level": {{"cardio_fitness": "low/moderate/high", "recovery": "slow/normal/fast", "stability": "stable/variable"}},
  "anomaly_thresholds": {{
    "resting_hr_upper_limit": X,
    "unexpected_elevation_criteria": "...",
    "sustained_elevation_minutes": X
  }}
}}

Use specific numbers from the data.
"""

        print("\n🤖 Analyzing with Gemini...")
        response = self._call_gemini(prompt, temperature=0.2)
        
        if response:
            try:
                import re
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', response, re.DOTALL)
                baseline_json = json_match.group(1) if json_match else response
                baseline_json = baseline_json.strip()
                
                if '{' in baseline_json:
                    start = baseline_json.index('{')
                    end = baseline_json.rindex('}') + 1
                    baseline_json = baseline_json[start:end]
                
                self.baseline = json.loads(baseline_json)
                
                print("\n✅ Baseline learned")
                print(f"   • Resting HR: {self.baseline['resting_hr']['range']} bpm")
                print(f"   • Fitness: {self.baseline['fitness_level']['cardio_fitness']}")
                print(f"   • Workout freq: {self.baseline['activity_profile']['workout_frequency_per_week']}/week")
                
                with open('baseline_profile.json', 'w') as f:
                    json.dump(self.baseline, f, indent=2)
                print(f"   • Saved: baseline_profile.json")
                
                return self.baseline
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"\nResponse:\n{response[:500]}...")
                return None
        
        return None
    
    def _prepare_baseline_stats(self) -> Dict:
        """Calculate comprehensive statistics for baseline learning."""
        
        hr_stats = {
            'total_records': len(self.hr_df),
            'days_of_data': (self.hr_df['startDate'].max() - self.hr_df['startDate'].min()).days,
            'overall': {
                'mean': round(self.hr_df['value'].mean(), 1),
                'median': round(self.hr_df['value'].median(), 1),
                'std': round(self.hr_df['value'].std(), 1),
                'min': int(self.hr_df['value'].min()),
                'max': int(self.hr_df['value'].max()),
                'percentiles': {
                    'p5': round(self.hr_df['value'].quantile(0.05), 1),
                    'p25': round(self.hr_df['value'].quantile(0.25), 1),
                    'p50': round(self.hr_df['value'].quantile(0.50), 1),
                    'p75': round(self.hr_df['value'].quantile(0.75), 1),
                    'p95': round(self.hr_df['value'].quantile(0.95), 1)
                }
            }
        }
        
        # By HR type
        if 'type' in self.hr_df.columns:
            by_type = {}
            for hr_type in self.hr_df['type'].unique():
                subset = self.hr_df[self.hr_df['type'] == hr_type]['value']
                by_type[hr_type] = {
                    'count': len(subset),
                    'mean': round(subset.mean(), 1),
                    'range': [int(subset.min()), int(subset.max())]
                }
            hr_stats['by_type'] = by_type
        
        # Hourly patterns
        self.hr_df['hour'] = self.hr_df['startDate'].dt.hour
        hourly = self.hr_df.groupby('hour')['value'].mean().round(1).to_dict()
        hr_stats['hourly_mean'] = hourly
        
        # Day of week
        self.hr_df['dow'] = self.hr_df['startDate'].dt.dayofweek
        dow = self.hr_df.groupby('dow')['value'].mean().round(1)
        hr_stats['day_of_week_mean'] = {
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i]: v 
            for i, v in dow.items()
        }
        
        # Workout statistics
        workout_stats = {
            'total_workouts': len(self.workout_df),
            'types': self.workout_df['workoutType'].value_counts().to_dict(),
            'duration_minutes': {
                'mean': round(self.workout_df['duration'].mean() / 60, 1),
                'median': round(self.workout_df['duration'].median() / 60, 1)
            },
            'frequency_per_week': round(
                len(self.workout_df) / (hr_stats['days_of_data'] / 7), 1
            ),
            'common_hours': self.workout_df['startDate'].dt.hour.value_counts().head(3).to_dict()
        }
        
        return {
            'heart_rate': hr_stats,
            'workouts': workout_stats
        }
    
    # ============================================================================
    # STAGE 2: ANOMALY WINDOW DETECTION
    # ============================================================================
    
    def detect_anomaly_windows(self, window_days: int = 7, sample_rate: float = 0.2) -> List[Dict]:
        """
        Stage 2: Identify time windows with anomalous patterns.
        
        Args:
            window_days: Size of each window in days
            sample_rate: Fraction of windows to analyze (0.0-1.0). Use <1.0 to reduce API costs.
        """
        print("\n" + "="*80)
        print("🔍 STAGE 2: DETECTING ANOMALY WINDOWS")
        print("="*80)
        
        if not self.baseline:
            print("❌ Run learn_baseline() first")
            return []
        
        windows = self._create_time_windows(window_days)
        
        # Sample windows if requested
        if sample_rate < 1.0:
            import random
            sample_size = max(1, int(len(windows) * sample_rate))
            windows = random.sample(windows, sample_size)
            print(f"\n📊 Randomly sampling {len(windows)} of total windows (sample_rate={sample_rate})")
        else:
            print(f"\n📊 Analyzing all {len(windows)} windows of {window_days} days...")
        
        anomaly_windows = []
        
        for i, window in enumerate(windows, 1):
            print(f"\n[{i}/{len(windows)}] {window['start_date']} → {window['end_date']}")
            
            window_stats = self._get_window_stats(window)
            
            if window_stats['hr_records'] < 100:
                print(f"   ⊘ Insufficient data")
                continue
            
            is_anomalous, analysis = self._analyze_window(window_stats)
            
            if is_anomalous:
                print(f"   ⚠️ ANOMALY")
                anomaly_windows.append({
                    'window_id': i,
                    'start_date': window['start_date'],
                    'end_date': window['end_date'],
                    'analysis': analysis,
                    'stats': window_stats
                })
            else:
                print(f"   ✓ Normal")
            
            if i < len(windows):
                import time
                time.sleep(1.5)
        
        self.anomaly_windows = anomaly_windows
        print(f"\n✅ Found {len(anomaly_windows)} anomalous windows")
        
        return anomaly_windows
    
    def _create_time_windows(self, window_days: int) -> List[Dict]:
        """Create overlapping time windows."""
        min_date = self.hr_df['startDate'].min()
        max_date = self.hr_df['startDate'].max()
        
        windows = []
        current = min_date
        
        while current < max_date:
            end = min(current + timedelta(days=window_days), max_date)
            
            windows.append({
                'start_date': current.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'start_dt': current,
                'end_dt': end
            })
            
            current += timedelta(days=window_days // 2)
        
        return windows
    
    def _get_window_stats(self, window: Dict) -> Dict:
        """Calculate statistics for a time window."""
        mask_hr = (self.hr_df['startDate'] >= window['start_dt']) & (self.hr_df['startDate'] < window['end_dt'])
        mask_wo = (self.workout_df['startDate'] >= window['start_dt']) & (self.workout_df['startDate'] < window['end_dt'])
        
        window_hr = self.hr_df[mask_hr]
        window_wo = self.workout_df[mask_wo]
        
        daily = window_hr.copy()
        daily['date'] = daily['startDate'].dt.date
        daily_stats = daily.groupby('date')['value'].agg(['mean', 'max', 'count']).round(1)
        
        # Convert date keys to strings for JSON serialization
        daily_means = {str(k): v for k, v in daily_stats['mean'].items()}
        daily_maxes = {str(k): v for k, v in daily_stats['max'].items()}
        
        return {
            'period': f"{window['start_date']} to {window['end_date']}",
            'hr_records': len(window_hr),
            'workouts': len(window_wo),
            'hr_mean': round(window_hr['value'].mean(), 1) if len(window_hr) > 0 else 0,
            'hr_median': round(window_hr['value'].median(), 1) if len(window_hr) > 0 else 0,
            'hr_std': round(window_hr['value'].std(), 1) if len(window_hr) > 0 else 0,
            'hr_max': int(window_hr['value'].max()) if len(window_hr) > 0 else 0,
            'hr_95th': round(window_hr['value'].quantile(0.95), 1) if len(window_hr) > 0 else 0,
            'daily_means': daily_means,
            'daily_maxes': daily_maxes,
            'workout_types': window_wo['workoutType'].value_counts().to_dict() if len(window_wo) > 0 else {}
        }
    
    def _analyze_window(self, window_stats: Dict) -> Tuple[bool, str]:
        """Ask Gemini to compare window against baseline."""
        
        prompt = f"""You are reviewing a 7-day window of heart rate data that has ALREADY been cleaned (outliers >200 bpm removed).

**ESTABLISHED BASELINE (normal patterns):**
{json.dumps(self.baseline, indent=2)}

**THIS WINDOW:**
{json.dumps(window_stats, indent=2)}

**TASK:** Determine if this window shows CLEAR, SIGNIFICANT deviation from baseline.

**STRICT CRITERIA (ALL must apply for anomaly):**
1. Window's mean HR is >15 bpm above baseline resting HR upper limit
2. Multiple days (≥3) show sustained elevation
3. Elevation occurs during typical resting hours (not exercise times)
4. NO corresponding increase in workout activity to explain it

**NOT ANOMALOUS if:**
- Window has normal workout activity matching elevated HR
- HR values still within baseline exercise range
- Only 1-2 days show elevation
- Elevation is minor (<10 bpm above baseline mean)
- Daily patterns match baseline temporal rhythms

**OUTPUT JSON only:**
{{
  "is_anomalous": true/false,
  "confidence": "low/medium/high",
  "reasons": ["specific reason 1", "specific reason 2"],
  "specific_dates_of_concern": ["YYYY-MM-DD"],
  "summary": "1-sentence explanation"
}}

CRITICAL: Be VERY conservative. Most windows should be normal. Only flag truly unusual patterns with strong evidence.
"""
        
        response = self._call_gemini(prompt, temperature=0.3)
        
        if response:
            try:
                import re
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', response, re.DOTALL)
                result_json = json_match.group(1) if json_match else response
                result_json = result_json.strip()
                
                if '{' in result_json:
                    start = result_json.index('{')
                    end = result_json.rindex('}') + 1
                    result_json = result_json[start:end]
                
                result = json.loads(result_json)
                return result.get('is_anomalous', False), result
                
            except:
                return False, {"error": "Parse failed"}
        
        return False, {"error": "No response"}
    
    # ============================================================================
    # STAGE 3: PRECISE ANOMALY IDENTIFICATION
    # ============================================================================
    
    def identify_precise_anomalies(self) -> List[Dict]:
        """Stage 3: Deep dive to find exact dates/times."""
        print("\n" + "="*80)
        print("🎯 STAGE 3: IDENTIFYING PRECISE ANOMALIES")
        print("="*80)
        
        if not self.anomaly_windows:
            print("❌ No anomalous windows. Run detect_anomaly_windows() first.")
            return []
        
        all_anomalies = []
        
        for i, window in enumerate(self.anomaly_windows, 1):
            print(f"\n[{i}/{len(self.anomaly_windows)}] {window['start_date']} → {window['end_date']}")
            
            detailed_data = self._get_detailed_window_data(window)
            anomalies = self._identify_specific_anomalies(window, detailed_data)
            
            if anomalies:
                all_anomalies.extend(anomalies)
                print(f"   → Found {len(anomalies)} anomalies")
            
            if i < len(self.anomaly_windows):
                import time
                time.sleep(2.0)
        
        self.anomalies = all_anomalies
        
        print(f"\n✅ Total anomalies: {len(all_anomalies)}")
        
        if all_anomalies:
            df = pd.DataFrame(all_anomalies)
            df.to_csv('detected_anomalies.csv', index=False)
            print(f"   • Saved: detected_anomalies.csv")
        
        return all_anomalies
    
    def _get_detailed_window_data(self, window: Dict) -> Dict:
        """Get day-by-day breakdown for window."""
        start = pd.to_datetime(window['start_date'])
        end = pd.to_datetime(window['end_date'])
        
        mask_hr = (self.hr_df['startDate'] >= start) & (self.hr_df['startDate'] < end)
        mask_wo = (self.workout_df['startDate'] >= start) & (self.workout_df['startDate'] < end)
        
        window_hr = self.hr_df[mask_hr].copy()
        window_wo = self.workout_df[mask_wo].copy()
        
        window_hr['date'] = window_hr['startDate'].dt.date
        window_hr['hour'] = window_hr['startDate'].dt.hour
        
        daily_breakdown = []
        
        for date in window_hr['date'].unique():
            day_hr = window_hr[window_hr['date'] == date]
            day_wo = window_wo[window_wo['startDate'].dt.date == date]
            
            hourly_stats = day_hr.groupby('hour')['value'].agg(['mean', 'max', 'count'])
            
            daily_breakdown.append({
                'date': str(date),
                'hr_mean': round(day_hr['value'].mean(), 1),
                'hr_median': round(day_hr['value'].median(), 1),
                'hr_max': int(day_hr['value'].max()),
                'hr_records': len(day_hr),
                'workouts': len(day_wo),
                'workout_types': day_wo['workoutType'].tolist() if len(day_wo) > 0 else [],
                'hourly_mean': {int(k): round(v, 1) for k, v in hourly_stats['mean'].items()},
                'hourly_max': {int(k): int(v) for k, v in hourly_stats['max'].items()}
            })
        
        return {'daily_breakdown': daily_breakdown}
    
    def _identify_specific_anomalies(self, window: Dict, detailed_data: Dict) -> List[Dict]:
        """Ask Gemini to pinpoint exact dates/times."""
        
        prompt = f"""Identify EXACT dates/times of anomalous HR in this window.

**BASELINE:**
{json.dumps(self.baseline, indent=2)}

**WINDOW ANALYSIS:**
{json.dumps(window['analysis'], indent=2)}

**DETAILED DATA:**
{json.dumps(detailed_data, indent=2)}

**CRITERIA:**
1. HR consistently >110 bpm during non-workout periods
2. HR elevated beyond baseline without activity
3. Sustained elevation (>30 min) in resting hours
4. Unusual patterns vs baseline

**OUTPUT JSON array only:**
[
  {{
    "date": "YYYY-MM-DD",
    "hours": [10, 11, 14],
    "hr_range": [min, max],
    "reason": "specific explanation",
    "confidence": "low/medium/high",
    "has_workout": true/false,
    "likely_cause": "best guess"
  }}
]

Only clear anomalies. Be specific about WHY.
"""
        
        response = self._call_gemini(prompt, temperature=0.3)
        
        if response:
            try:
                import re
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', response, re.DOTALL)
                anomalies_json = json_match.group(1) if json_match else response
                anomalies_json = anomalies_json.strip()
                
                if '[' in anomalies_json:
                    start = anomalies_json.index('[')
                    end = anomalies_json.rindex(']') + 1
                    anomalies_json = anomalies_json[start:end]
                
                anomalies = json.loads(anomalies_json)
                return anomalies if isinstance(anomalies, list) else []
                
            except Exception as e:
                print(f"   ⚠️ Parse error: {e}")
                return []
        
        return []
    
    # ============================================================================
    # REPORT GENERATION
    # ============================================================================
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive anomaly report."""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'anomaly_report_{timestamp}.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GEMINI AUTONOMOUS ANOMALY DETECTION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Gemini 2.5 Flash Lite\n")
            f.write(f"Data: {self.hr_df['startDate'].min().date()} to {self.hr_df['startDate'].max().date()}\n")
            f.write("="*80 + "\n\n")
            
            f.write("LEARNED BASELINE\n")
            f.write("-"*80 + "\n")
            f.write(json.dumps(self.baseline, indent=2))
            f.write("\n\n")
            
            f.write(f"ANOMALOUS WINDOWS ({len(self.anomaly_windows)})\n")
            f.write("-"*80 + "\n")
            for w in self.anomaly_windows:
                f.write(f"\nWindow: {w['start_date']} → {w['end_date']}\n")
                f.write(json.dumps(w['analysis'], indent=2))
                f.write("\n")
            f.write("\n")
            
            f.write(f"PRECISE ANOMALIES ({len(self.anomalies)})\n")
            f.write("-"*80 + "\n")
            for a in self.anomalies:
                f.write(f"\nDate: {a['date']}\n")
                f.write(f"  Hours: {a.get('hours', 'N/A')}\n")
                f.write(f"  HR: {a.get('hr_range', 'N/A')} bpm\n")
                f.write(f"  Reason: {a.get('reason', 'N/A')}\n")
                f.write(f"  Cause: {a.get('likely_cause', 'Unknown')}\n")
                f.write(f"  Confidence: {a.get('confidence', 'N/A')}\n")
        
        print(f"\n✅ Report: {output_file}")
        return output_file


# ============================================================================
# CLI
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🤖 GEMINI AUTONOMOUS ANOMALY DETECTION")
    print("="*80)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ GEMINI_API_KEY not set")
        print("Set: export GEMINI_API_KEY='your-key'")
        return
    
    agent = GeminiAnomalyAgent(api_key)
    
    # Update paths
    HR_FILE = "agent_data/heart_rate_data.csv"
    WORKOUT_FILE = "agent_data/workout_data.csv"
    
    if not os.path.exists(HR_FILE) or not os.path.exists(WORKOUT_FILE):
        print(f"\n❌ Files not found:")
        print(f"   • {HR_FILE}")
        print(f"   • {WORKOUT_FILE}")
        return
    
    agent.load_and_clean_data(HR_FILE, WORKOUT_FILE)
    
    while True:
        print("\n" + "="*80)
        print("OPTIONS")
        print("="*80)
        print("1. Learn Baseline (Stage 1)")
        print("2. Detect Anomaly Windows (Stage 2)")
        print("3. Identify Precise Anomalies (Stage 3)")
        print("4. Run Full Pipeline")
        print("5. Generate Report")
        print("6. Exit")
        print("="*80)
        
        choice = input("\nSelect (1-6): ").strip()
        
        if choice == '1':
            agent.learn_baseline()
        elif choice == '2':
            days = input("Window days (default 7): ").strip()
            days = int(days) if days.isdigit() else 7
            
            sample = input("Sample rate 0.0-1.0 (default 0.2 = 20%): ").strip()
            try:
                sample_rate = float(sample) if sample else 0.2
                sample_rate = max(0.01, min(1.0, sample_rate))
            except:
                sample_rate = 0.2
            
            agent.detect_anomaly_windows(window_days=days, sample_rate=sample_rate)
        elif choice == '3':
            agent.identify_precise_anomalies()
        elif choice == '4':
            print("\n⚡ Running full pipeline with 20% sampling (faster, cost-effective)")
            agent.learn_baseline()
            agent.detect_anomaly_windows(sample_rate=0.2)
            agent.identify_precise_anomalies()
            agent.generate_report()
        elif choice == '5':
            agent.generate_report()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()