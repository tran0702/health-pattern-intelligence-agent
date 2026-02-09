"""
Enhanced Data Extractor - With Location/Weather Forward-Fill
Intelligently fills missing location/weather data for comprehensive analysis
"""

import pandas as pd
import xml.etree.ElementTree as ET
import os
import warnings
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import time

warnings.filterwarnings('ignore')


class DataExtractor:
    """Extract and prepare CSV files from Apple Health XML with location/weather enrichment."""
    
    def __init__(self, xml_path: str, output_dir: str = "agent_data", 
                 visual_crossing_api_key: Optional[str] = None):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.api_key = visual_crossing_api_key
        os.makedirs(output_dir, exist_ok=True)
        
        # Cache for weather data to avoid duplicate API calls
        self.weather_cache: Dict[Tuple[str, str], Dict] = {}
    
    def load_existing_location_weather(self, hr_location_weather_path: str) -> pd.DataFrame:
        """Load existing hr_with_location_weather.csv as base data."""
        print("\n⏳ Loading existing location/weather data...")
        
        if not os.path.exists(hr_location_weather_path):
            raise FileNotFoundError(f"File not found: {hr_location_weather_path}")
        
        df = pd.read_csv(hr_location_weather_path)
        
        # Parse datetime columns
        df['startDate'] = pd.to_datetime(df['startDate'], format='%d/%m/%Y %H:%M')
        df['endDate'] = pd.to_datetime(df['endDate'], format='%d/%m/%Y %H:%M')
        if 'datetime_hour' in df.columns:
            df['datetime_hour'] = pd.to_datetime(df['datetime_hour'], format='%d/%m/%Y %H:%M')
        
        print(f"✅ Loaded {len(df):,} records with location/weather data")
        print(f"   - Records with location: {df['location_name'].notna().sum():,}")
        print(f"   - Records with weather: {df['temperature_c'].notna().sum():,}")
        
        return df
    
    def extract_location_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract timeline of known locations from workout GPS data."""
        location_history = df[df['location_name'].notna()][
            ['startDate', 'location_name', 'location_cluster']
        ].copy()
        location_history = location_history.sort_values('startDate').drop_duplicates()
        
        print(f"\n📍 Location History: {len(location_history)} GPS checkpoints")
        return location_history
    
    def forward_fill_location(self, df: pd.DataFrame, max_days_stale: int = 14) -> pd.DataFrame:
        """Forward-fill location from last known GPS workout."""
        print("\n⏳ Forward-filling missing locations...")
        
        df = df.sort_values('startDate').copy()
        
        # Track last known location
        last_location = None
        last_location_date = None
        location_filled = 0
        location_uncertain = 0
        
        for idx, row in df.iterrows():
            if pd.notna(row['location_name']):
                # Update known location from GPS
                last_location = row['location_name']
                last_location_date = row['startDate']
                df.at[idx, 'location_confidence'] = 'gps_confirmed'
            else:
                # Fill with last known location
                if last_location is not None:
                    days_since_gps = (row['startDate'] - last_location_date).days
                    
                    if days_since_gps <= max_days_stale:
                        df.at[idx, 'location_name'] = last_location
                        df.at[idx, 'location_confidence'] = f'filled_{days_since_gps}d'
                        location_filled += 1
                    else:
                        df.at[idx, 'location_confidence'] = 'uncertain_stale'
                        location_uncertain += 1
                else:
                    df.at[idx, 'location_confidence'] = 'unknown'
        
        print(f"✅ Location forward-fill complete:")
        print(f"   - Filled: {location_filled:,} records")
        print(f"   - Uncertain (stale): {location_uncertain:,} records")
        print(f"   - Unknown (no GPS history): {(df['location_confidence'] == 'unknown').sum():,} records")
        
        return df
    
    def get_weather_for_location_date(self, location: str, date: datetime) -> Optional[Dict]:
        """
        Get historical weather data from Visual Crossing API.
        Adelaide coordinates: -34.9285, 138.6007
        """
        if not self.api_key:
            return None
        
        # Cache key
        date_key = date.strftime('%Y-%m-%d')
        cache_key = (location, date_key)
        
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]
        
        # Adelaide coordinates (default - expand this mapping)
        location_coords = {
            'Panorama': (-34.9285, 138.6007),
            'Adelaide': (-34.9285, 138.6007),
            'Adelaide CBD': (-34.9285, 138.6007),
            # Add more locations as needed
        }
        
        if location not in location_coords:
            print(f"⚠️  Unknown location coordinates: {location}")
            return None
        
        lat, lon = location_coords[location]
        
        try:
            # Visual Crossing API
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_key}"
            params = {
                'key': self.api_key,
                'unitGroup': 'metric',
                'include': 'hours',
                'contentType': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract hourly data
            if 'days' in data and len(data['days']) > 0:
                day_data = data['days'][0]
                
                weather_data = {
                    'temperature_c': day_data.get('temp'),
                    'feels_like_c': day_data.get('feelslike'),
                    'humidity_pct': day_data.get('humidity'),
                    'weather_description': day_data.get('conditions', ''),
                    'hours': day_data.get('hours', [])  # Hourly breakdown
                }
                
                self.weather_cache[cache_key] = weather_data
                time.sleep(0.1)  # Rate limiting
                return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Weather API error for {location} on {date_key}: {e}")
            return None
        
        return None
    
    def get_hourly_weather(self, weather_data: Dict, hour: int) -> Dict:
        """Extract specific hour from daily weather data."""
        if not weather_data or 'hours' not in weather_data:
            return {}
        
        for hour_data in weather_data['hours']:
            if hour_data.get('datetime', '').startswith(f"{hour:02d}:"):
                return {
                    'temperature_c': hour_data.get('temp'),
                    'feels_like_c': hour_data.get('feelslike'),
                    'humidity_pct': hour_data.get('humidity'),
                    'weather_description': hour_data.get('conditions', '')
                }
        
        # Fallback to daily averages
        return {
            'temperature_c': weather_data.get('temperature_c'),
            'feels_like_c': weather_data.get('feels_like_c'),
            'humidity_pct': weather_data.get('humidity_pct'),
            'weather_description': weather_data.get('weather_description')
        }
    
    def backfill_weather(self, df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """Backfill weather data for records with location but no weather."""
        if not self.api_key:
            print("⚠️  No Visual Crossing API key provided - skipping weather backfill")
            print("   Set API key: extractor = DataExtractor(xml_path, visual_crossing_api_key='YOUR_KEY')")
            return df
        
        print("\n⏳ Backfilling missing weather data...")
        
        missing_weather = df[
            (df['location_name'].notna()) & 
            (df['temperature_c'].isna())
        ]
        
        if len(missing_weather) == 0:
            print("✅ No weather backfill needed")
            return df
        
        print(f"   - Records to backfill: {len(missing_weather):,}")
        
        # Group by (location, date) to minimize API calls
        unique_location_dates = missing_weather.groupby([
            'location_name',
            pd.Grouper(key='startDate', freq='D')
        ]).size().reset_index()[['location_name', 'startDate']]
        
        print(f"   - Unique API calls needed: {len(unique_location_dates)}")
        
        filled_count = 0
        
        for i in range(0, len(unique_location_dates), batch_size):
            batch = unique_location_dates.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                location = row['location_name']
                date = row['startDate']
                
                # Get daily weather
                weather_data = self.get_weather_for_location_date(location, date)
                
                if weather_data:
                    # Apply to all records on this (location, date)
                    mask = (
                        (df['location_name'] == location) &
                        (df['startDate'].dt.date == date.date()) &
                        (df['temperature_c'].isna())
                    )
                    
                    # Apply hourly-specific weather if available
                    for idx in df[mask].index:
                        hour = df.at[idx, 'startDate'].hour
                        hourly_weather = self.get_hourly_weather(weather_data, hour)
                        
                        df.at[idx, 'temperature_c'] = hourly_weather.get('temperature_c')
                        df.at[idx, 'feels_like_c'] = hourly_weather.get('feels_like_c')
                        df.at[idx, 'humidity_pct'] = hourly_weather.get('humidity_pct')
                        df.at[idx, 'weather_description'] = hourly_weather.get('weather_description')
                        df.at[idx, 'weather_confidence'] = 'api_backfilled'
                        
                        filled_count += 1
            
            print(f"   - Progress: {min(i+batch_size, len(unique_location_dates))}/{len(unique_location_dates)} batches")
        
        print(f"✅ Weather backfill complete: {filled_count:,} records filled")
        
        return df
    
    def enrich_heart_rate_data(self, hr_location_weather_path: str,
                               enable_location_fill: bool = True,
                               enable_weather_fill: bool = True) -> pd.DataFrame:
        """
        Main method: Load existing data and enrich with forward-fill logic.
        """
        print("\n" + "="*80)
        print("🌡️  ENRICHING HEART RATE DATA WITH LOCATION & WEATHER")
        print("="*80)
        
        # Load existing data
        df = self.load_existing_location_weather(hr_location_weather_path)
        
        # Forward-fill locations
        if enable_location_fill:
            df = self.forward_fill_location(df)
        
        # Backfill weather
        if enable_weather_fill:
            df = self.backfill_weather(df)
        
        # Save enriched data
        output_file = f"{self.output_dir}/heart_rate_enriched.csv"
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ ENRICHMENT COMPLETE")
        print("="*80)
        print(f"📊 Final Statistics:")
        print(f"   - Total records: {len(df):,}")
        print(f"   - With location: {df['location_name'].notna().sum():,} ({df['location_name'].notna().sum()/len(df)*100:.1f}%)")
        print(f"   - With weather: {df['temperature_c'].notna().sum():,} ({df['temperature_c'].notna().sum()/len(df)*100:.1f}%)")
        print(f"\n💾 Saved to: {output_file}")
        
        return df
    
    def extract_heart_rate(self) -> pd.DataFrame:
        """Extract HR data efficiently (legacy method)."""
        print("\n⏳ Extracting Heart Rate data...")
        
        hr_types = [
            "HKQuantityTypeIdentifierHeartRate",
            "HKQuantityTypeIdentifierRestingHeartRate",
            "HKQuantityTypeIdentifierWalkingHeartRateAverage"
        ]
        
        records = []
        for event, elem in ET.iterparse(self.xml_path, events=('end',)):
            if elem.tag == 'Record' and elem.get('type') in hr_types:
                records.append({
                    'type': elem.get('type', '').replace('HKQuantityTypeIdentifier', ''),
                    'sourceName': elem.get('sourceName', ''),
                    'value': float(elem.get('value', 0)),
                    'unit': elem.get('unit', ''),
                    'startDate': elem.get('startDate', ''),
                    'endDate': elem.get('endDate', ''),
                })
                elem.clear()
        
        df = pd.DataFrame(records)
        df['startDate'] = pd.to_datetime(df['startDate'].str.replace(r' [+-]\d{4}$', '', regex=True))
        df['endDate'] = pd.to_datetime(df['endDate'].str.replace(r' [+-]\d{4}$', '', regex=True))
        
        output_file = f"{self.output_dir}/heart_rate_data.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(df):,} HR records → {output_file}")
        return df
    
    def extract_workouts(self) -> pd.DataFrame:
        """Extract workout data."""
        print("\n⏳ Extracting Workout data...")
        
        workouts = []
        for event, elem in ET.iterparse(self.xml_path, events=('end',)):
            if elem.tag == 'Workout':
                workouts.append({
                    'workoutType': elem.get('workoutActivityType', '').replace('HKWorkoutActivityType', ''),
                    'duration': float(elem.get('duration', 0)),
                    'durationUnit': elem.get('durationUnit', ''),
                    'totalDistance': float(elem.get('totalDistance', 0) or 0),
                    'totalDistanceUnit': elem.get('totalDistanceUnit', ''),
                    'totalEnergyBurned': float(elem.get('totalEnergyBurned', 0) or 0),
                    'sourceName': elem.get('sourceName', ''),
                    'startDate': elem.get('startDate', ''),
                    'endDate': elem.get('endDate', ''),
                })
                elem.clear()
        
        df = pd.DataFrame(workouts)
        df['startDate'] = pd.to_datetime(df['startDate'].str.replace(r' [+-]\d{4}$', '', regex=True))
        df['endDate'] = pd.to_datetime(df['endDate'].str.replace(r' [+-]\d{4}$', '', regex=True))
        
        output_file = f"{self.output_dir}/workout_data.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(df):,} workouts → {output_file}")
        return df
    
    def copy_anomaly_file(self, source_file: str):
        """Copy existing anomaly analysis file."""
        if os.path.exists(source_file):
            import shutil
            dest = f"{self.output_dir}/anomaly_days_deep_dive.csv"
            shutil.copy(source_file, dest)
            print(f"✅ Copied anomaly data → {dest}")
        else:
            print(f"⚠️  Anomaly file not found: {source_file}")
    
    def extract_all(self, anomaly_source: str = None, hr_location_weather_path: str = None):
        """Extract all data needed for agent."""
        print("\n" + "="*60)
        print("📦 EXTRACTING DATA FOR HEALTH PATTERN AGENT")
        print("="*60)
        
        # Use enriched HR data if path provided
        if hr_location_weather_path and os.path.exists(hr_location_weather_path):
            self.enrich_heart_rate_data(hr_location_weather_path)
        else:
            self.extract_heart_rate()
        
        self.extract_workouts()
        
        if anomaly_source:
            self.copy_anomaly_file(anomaly_source)
        
        print("\n" + "="*60)
        print("✅ DATA EXTRACTION COMPLETE")
        print("="*60)
        print(f"Files saved to: {self.output_dir}/")
        print("\nReady to run: python run_health_agent.py")


if __name__ == "__main__":
    import sys
    
    # Configuration
    XML_PATH = r"C:\Project\Apple Health Data\data\apple_health_export\export_cleaned.xml"
    HR_LOCATION_WEATHER = r"C:\Project\Apple Health Data\output\hr_with_location_weather.csv"
    
    # Get Visual Crossing API key from environment
    VISUAL_CROSSING_API_KEY = os.getenv('VISUAL_CROSSING_API_KEY')
    
    if len(sys.argv) > 1:
        XML_PATH = sys.argv[1]
    
    # Initialize extractor with API key
    extractor = DataExtractor(
        XML_PATH,
        visual_crossing_api_key=VISUAL_CROSSING_API_KEY
    )
    
    # Extract with enrichment
    extractor.extract_all(
        anomaly_source=ANOMALY_CSV,
        hr_location_weather_path=HR_LOCATION_WEATHER
    )