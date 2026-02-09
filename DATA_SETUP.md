# Data Setup Guide

## Important: Original Data Has Been Removed

This repository does NOT contain any original Apple Health export data. All personal health data has been removed for privacy. Only processed results, trained models, and analysis code are included.

## Using This Repository

### Option 1: View Existing Results (No Setup Required)

All analysis outputs are available in the `results/` directory:

- **Visualizations**: `results/visualizations/` (21 charts organized by category)
- **Data Files**: `results/data/` (22 CSV files with processed results)
- **Models**: `results/models/` (2 trained ML models)
- **Reports**: `results/reports/` (11 analysis reports)

### Option 2: Run Analysis with Your Own Data

If you want to run the analysis with your own Apple Health data:

#### Step 1: Export Your Apple Health Data

1. Open the Health app on your iPhone
2. Tap your profile icon (top right)
3. Scroll down and tap "Export All Health Data"
4. Wait for export to complete (may take several minutes)
5. Share the export.zip file to your computer
6. Extract the zip file

#### Step 2: Create Data Directory

```bash
# In the repository root
mkdir -p data/apple_health_export
```

#### Step 3: Copy Your Export Files

```bash
# Copy the export to the data directory
cp /path/to/your/export/export.xml data/apple_health_export/
cp -r /path/to/your/export/workout-routes data/apple_health_export/
cp -r /path/to/your/export/electrocardiograms data/apple_health_export/
```

#### Step 4: Configure Paths (Optional)

The notebooks are already configured with relative paths. If running from the repository root, paths will work automatically:

**Notebooks** (executed from `notebooks/` directory):
- Input: `../data/apple_health_export/export.xml`
- Output: `../results/data/`, `../results/visualizations/`, etc.

**Python Scripts** (executed from `src/agents/` directory):
- Input: `../../results/data/hr_with_location_weather_enriched.csv`
- Output: `../../results/reports/agent_outputs/`

#### Step 5: Set Up API Keys (For AI Agents)

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"
```

#### Step 6: Run Analysis Pipeline

```bash
# 1. Extract and load data
cd notebooks
jupyter notebook load_ecg_data.ipynb

# 2. Analyze weather and location patterns
jupyter notebook weather_location_workout_analysis.ipynb

# 3. Run GNN anomaly detection
jupyter notebook gnn_anomaly_detection.ipynb

# 4. Run Gemini agent analysis
cd ../src/agents
python run_health_agent.py

# 5. Compare models
cd ../../notebooks
jupyter notebook anomaly_model_comparison.ipynb
```

## File Path Structure

### Current Repository Structure

```
Apple-Health-Data/
├── data/                          ← NOT INCLUDED (gitignored)
│   └── apple_health_export/       ← Create this if using your own data
│       ├── export.xml
│       ├── workout-routes/
│       └── electrocardiograms/
│
├── notebooks/                     ← Analysis notebooks
├── src/agents/                    ← Python scripts
│
└── results/                       ← All outputs (INCLUDED)
    ├── visualizations/
    ├── data/
    ├── models/
    └── reports/
```

### Path References in Code

#### From Notebooks (notebooks/*.ipynb)

```python
# Reading original data (if you have it)
xml_path = "../data/apple_health_export/export.xml"

# Reading processed results
df = pd.read_csv("../results/data/hr_with_location_weather.csv")

# Writing new results
plt.savefig("../results/visualizations/weather_location/my_plot.png")
df.to_csv("../results/data/my_output.csv")
```

#### From Python Scripts (src/agents/*.py)

```python
# Reading processed data
HR_FILE = "../../results/data/hr_with_location_weather_enriched.csv"

# Writing reports
REPORT_DIR = "../../results/reports/agent_outputs"
```

## Troubleshooting

### "File not found" errors

**Problem**: Notebook can't find data files

**Solution**:
1. Check you're running the notebook from the `notebooks/` directory
2. Verify the file exists in the expected location
3. Use relative paths (`../results/data/...`) not absolute paths

### Import errors in Python scripts

**Problem**: `ModuleNotFoundError` when running scripts

**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run from the correct directory
cd src/agents
python run_health_agent.py
```

### Gemini API errors

**Problem**: API key not recognized

**Solution**:
```bash
# Set environment variable
export GEMINI_API_KEY="your-key"

# Or create .env file in repository root
echo "GEMINI_API_KEY=your-key" > .env
```

## Privacy Note

The `data/` directory is listed in `.gitignore` to prevent accidentally committing personal health data. Always keep your Apple Health export private and never commit it to version control.

## Need Help?

- Review the [workflow documentation](docs/workflow.md) for the complete analysis pipeline
- Check the [methodology document](docs/methodology.md) for technical details
- See the main [README](README.md) for project overview
