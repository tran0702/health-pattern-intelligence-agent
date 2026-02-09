"""
Run Health Pattern Intelligence Agent
Interactive CLI for lifestyle analysis from Apple Health data.
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from health_pattern_agent import HealthPatternAgent


def main():
    print("\n" + "=" * 80)
    print("🏥 HEALTH PATTERN INTELLIGENCE AGENT")
    print("=" * 80)

    # --- Configuration ---
    # Data paths (adjust these to your setup)
    # Note: Update these paths to point to your processed data files
    HR_FILE = r"../../results/data/hr_with_location_weather_enriched.csv"
    WORKOUT_FILE = r"../../results/data/workout_routes.csv"

    # Output paths (relative to src/agents/)
    REPORT_DIR = "../../results/reports/agent_outputs"
    ANOMALY_EXPORT = "../../results/data/anomaly_agent_output.csv"

    # --- API Key ---
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found!")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        api_key = input("Or enter your API key now: ").strip()
        if not api_key:
            return
        os.environ['GEMINI_API_KEY'] = api_key

    # --- Initialize Agent ---
    try:
        agent = HealthPatternAgent(api_key, model="gemini-2.5-flash-lite")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # --- Load Data ---
    if not os.path.exists(HR_FILE):
        print(f"\n❌ HR data not found: {HR_FILE}")
        print("Please update HR_FILE path in this script.")
        return

    workout_file = WORKOUT_FILE if os.path.exists(WORKOUT_FILE) else None
    anomaly_file = ANOMALY_FILE if os.path.exists(ANOMALY_FILE) else None

    agent.load_data(HR_FILE, workout_file, anomaly_file)

    # --- Interactive Menu ---
    while True:
        print("\n" + "=" * 80)
        print("📊 ANALYSIS OPTIONS")
        print("=" * 80)
        print("1. Baseline Profile     - Lifestyle HR by location/weather")
        print("2. Pattern Analysis     - Behavioral patterns across contexts")
        print("3. Workout Performance  - Climate-aware training optimization")
        print("4. Anomaly Analysis     - Detect & investigate unusual patterns")
        print("5. Recommendations      - Location/weather-aware advice")
        print("6. Comprehensive Report - Full analysis (4-5 API calls)")
        print("7. Export Anomalies     - GNN-compatible CSV for model comparison")
        print("8. Ask Custom Question  - Free-form query")
        print("9. Exit")
        print("=" * 80)

        choice = input("\nSelect option (1-9): ").strip()

        if choice == '1':
            agent.analyze_with_gemini('baseline')
        elif choice == '2':
            agent.analyze_with_gemini('patterns')
        elif choice == '3':
            agent.analyze_with_gemini('workout_performance')
        elif choice == '4':
            agent.analyze_anomalies()
        elif choice == '5':
            agent.analyze_with_gemini('recommendations')
        elif choice == '6':
            agent.generate_comprehensive_report(REPORT_DIR)
        elif choice == '7':
            export_path = input(f"Export path [{ANOMALY_EXPORT}]: ").strip() or ANOMALY_EXPORT
            agent.export_anomalies_for_comparison(export_path)
            print(f"\n📤 Anomaly CSV exported. Compare with your GNN model output.")
            print("   Schema: timestamp, hr_value, predicted_anomaly, anomaly_score,")
            print("           activity_context, hr_zone, time_period, hour, day_of_week, is_weekend, type")
        elif choice == '8':
            question = input("\nYour question: ").strip()
            if question:
                agent.ask_question(question)
        elif choice == '9':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
