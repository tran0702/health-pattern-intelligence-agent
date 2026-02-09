# Apple Health Data Analysis Workflow

## Analysis Pipeline

This document describes the complete analysis workflow and dependencies between files.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Data Loading & Preprocessing                      │
│  File: notebooks/load_ecg_data.ipynb                        │
│  ─────────────────────────────────────────────────────────  │
│  Input:  data/apple_health_export/export.xml (DELETED)     │
│  Output: Heart rate data, ECG data, Workout data in memory │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Weather & Location Analysis                       │
│  File: notebooks/weather_location_workout_analysis.ipynb    │
│  ─────────────────────────────────────────────────────────  │
│  Input:  Heart rate and workout data from Step 1           │
│  Output: results/data/hr_with_location_weather.csv         │
│          results/data/workout_routes.csv                    │
│          results/data/location_summary.csv                  │
│          results/visualizations/weather_location/*.png      │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴──────────┐
                    ↓                    ↓
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Step 3A: GNN Anomaly        │  │  Step 3B: Gemini Agent       │
│  File: notebooks/            │  │  Files: src/agents/          │
│        gnn_anomaly_          │  │    - health_pattern_agent.py │
│        detection.ipynb       │  │    - gemini_anomaly_agent.py │
│  ───────────────────────────  │  │    - run_health_agent.py     │
│  Input:  HR + weather data   │  │  ───────────────────────────  │
│  Output: results/data/       │  │  Input:  HR + weather data   │
│          anomaly_predictions │  │  Output: results/reports/    │
│          .csv                │  │          agent_outputs/*.txt │
│          results/models/     │  │          results/data/       │
│          gnn_anomaly_model   │  │          anomaly_agent_      │
│          .pth                │  │          output-update.csv   │
└──────────────────────────────┘  └──────────────────────────────┘
                    └─────────┬──────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Model Comparison & Validation                     │
│  File: notebooks/anomaly_model_comparison.ipynb             │
│  ─────────────────────────────────────────────────────────  │
│  Input:  results/data/anomaly_predictions.csv (GNN)         │
│          results/data/anomaly_agent_output-update.csv       │
│  Output: results/visualizations/model_comparison/*.png      │
│          results/data/model_comparison_full.csv             │
│          results/data/model_disagreements.csv               │
└─────────────────────────────────────────────────────────────┘
```

## File Dependencies

### Notebooks (in execution order)

1. **load_ecg_data.ipynb**
   - **Purpose**: Initial data extraction from Apple Health export
   - **Inputs**: `data/apple_health_export/export.xml` (now DELETED)
   - **Outputs**: In-memory dataframes for next steps
   - **Note**: Original data has been removed; notebook provided as reference

2. **weather_location_workout_analysis.ipynb**
   - **Purpose**: Environmental context integration
   - **Inputs**: Heart rate and workout data
   - **Outputs**:
     - `results/data/hr_with_location_weather.csv`
     - `results/data/hr_with_location_weather_enriched.csv`
     - `results/data/workout_routes.csv`
     - `results/data/location_summary.csv`
     - `results/visualizations/weather_location/*.png`

3. **gnn_anomaly_detection.ipynb**
   - **Purpose**: Graph Neural Network anomaly detection
   - **Inputs**: `results/data/hr_with_location_weather_enriched.csv`
   - **Outputs**:
     - `results/data/anomaly_predictions.csv`
     - `results/data/gnn_predictions.csv`
     - `results/models/gnn_anomaly_model.pth`
     - `results/visualizations/anomaly_detection/*.png`

4. **anomaly_model_comparison.ipynb**
   - **Purpose**: Compare GNN vs Gemini agent anomaly detection
   - **Inputs**:
     - `results/data/anomaly_agent_output-update.csv` (Gemini)
     - `results/data/anomaly_predictions.csv` (GNN)
   - **Outputs**:
     - `results/data/model_comparison_full.csv`
     - `results/data/model_disagreements.csv`
     - `results/visualizations/model_comparison/*.png`

### Python Scripts

1. **src/agents/health_pattern_agent.py**
   - Main agent class for Gemini-powered analysis

2. **src/agents/gemini_anomaly_agent.py**
   - Specialized anomaly detection agent

3. **src/agents/agent_prompts_enhanced.py**
   - Prompt templates for LLM interactions

4. **src/agents/run_health_agent.py**
   - Interactive CLI for running health agents
   - **Inputs**: Heart rate and workout CSV files (user-configured)
   - **Outputs**:
     - `results/reports/agent_outputs/*.md`
     - `results/data/anomaly_agent_output.csv`

5. **src/utils/data_extractor.py**
   - Utility functions for data extraction

## Path Configuration Notes

### For Running Notebooks

When running notebooks from the `notebooks/` directory, use relative paths:

- **Reading from results**: `../results/data/filename.csv`
- **Writing to results**: `../results/visualizations/category/plot.png`
- **Original data** (DELETED): Reference only, not available

### For Running Python Scripts

When running scripts from `src/agents/`, use relative paths:

- **Reading data**: `../../results/data/filename.csv`
- **Writing reports**: `../../results/reports/agent_outputs/report.md`

### Running from Repository Root

If running from root directory:

- **Notebooks**: `jupyter notebook notebooks/filename.ipynb`
- **Scripts**: `python src/agents/run_health_agent.py`

## Data Files Locations

### Input Data (Original - DELETED)
- `data/apple_health_export/` - No longer exists, removed for privacy

### Intermediate Results
- `results/data/hr_with_location_weather.csv` - Enriched heart rate data
- `results/data/workout_routes.csv` - Workout GPS routes processed

### Model Outputs
- `results/data/anomaly_predictions.csv` - GNN model predictions
- `results/data/anomaly_agent_output-update.csv` - Gemini agent predictions
- `results/models/gnn_anomaly_model.pth` - Trained GNN model

### Comparison Results
- `results/data/model_comparison_full.csv` - Full model comparison
- `results/data/model_disagreements.csv` - Cases where models disagree

### Visualizations
- `results/visualizations/anomaly_detection/` - Anomaly detection plots
- `results/visualizations/heart_rate_analysis/` - HR analysis charts
- `results/visualizations/model_comparison/` - Model comparison plots
- `results/visualizations/weather_location/` - Weather/location visualizations

### Reports
- `results/reports/agent_outputs/` - AI-generated analysis reports
- `results/reports/statistical_summaries/` - Statistical summary reports

## Usage Instructions

### To Replicate Analysis (with your own data)

1. Export your Apple Health data
2. Place export in a local `data/` folder (gitignored)
3. Run notebooks in order:
   ```bash
   cd notebooks
   jupyter notebook load_ecg_data.ipynb  # Extract your data
   jupyter notebook weather_location_workout_analysis.ipynb
   jupyter notebook gnn_anomaly_detection.ipynb
   ```
4. Configure and run Gemini agent:
   ```bash
   cd ../src/agents
   export GEMINI_API_KEY="your-key"
   python run_health_agent.py
   ```
5. Compare models:
   ```bash
   cd ../../notebooks
   jupyter notebook anomaly_model_comparison.ipynb
   ```

### To Review Existing Results

All analysis outputs are in `results/` directory:
- View visualizations in `results/visualizations/`
- Explore data files in `results/data/`
- Read reports in `results/reports/`

---

*Last Updated: February 2026*
