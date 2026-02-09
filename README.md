# Apple Health Data Analysis

A comprehensive analysis project for Apple Health data using machine learning, graph neural networks, and AI agents to detect anomalies and patterns in health metrics.

## Project Overview

This project analyzes Apple Health export data to:
- Detect anomalies in heart rate and activity patterns
- Analyze relationships between weather, location, and workout performance
- Compare multiple machine learning models for health pattern recognition
- Use AI agents for automated health insights
- Apply Graph Neural Networks (GNN) for temporal pattern detection

## Features

### 1. Anomaly Detection
- Multiple anomaly detection algorithms (Isolation Forest, LOF, One-Class SVM)
- Graph Neural Network-based temporal anomaly detection
- Ensemble model comparison and consensus analysis
- Automated anomaly reporting with contextual insights

### 2. Health Pattern Analysis
- Heart rate analysis across different contexts
- Workout performance correlation with environmental factors
- Resting heart rate (RHR) and heart rate recovery (HRR) tracking
- Training load estimation and fatigue monitoring

### 3. Environmental Context Integration
- Weather data integration for workout analysis
- Location-based heart rate pattern detection
- Temporal pattern analysis (hourly, daily, seasonal)
- Geographic visualization of workout routes

### 4. AI-Powered Insights
- Gemini-enhanced anomaly agent for natural language insights
- Pattern recognition agent for health trend analysis
- Automated health report generation
- Contextual interpretation of detected anomalies

## Repository Structure

```
.
├── notebooks/                              # Jupyter notebooks for analysis
│   ├── anomaly_model_comparison.ipynb     # Model comparison and evaluation
│   ├── gnn_anomaly_detection.ipynb        # Graph Neural Network implementation
│   ├── load_ecg_data.ipynb                # ECG data loading and preprocessing
│   └── weather_location_workout_analysis.ipynb  # Environmental analysis
│
├── src/                                    # Source code
│   ├── agents/                            # AI agent implementations
│   │   ├── health_pattern_agent.py       # Main health analysis agent
│   │   ├── gemini_anomaly_agent.py       # Gemini-powered anomaly detection
│   │   ├── agent_prompts_enhanced.py     # LLM prompts and templates
│   │   └── run_health_agent.py           # Agent execution script
│   └── utils/                             # Utility functions
│       └── data_extractor.py             # Data extraction utilities
│
├── results/                                # Analysis outputs
│   ├── visualizations/                    # Generated charts and graphs
│   │   ├── anomaly_detection/            # Anomaly detection plots
│   │   ├── heart_rate_analysis/          # HR analysis visualizations
│   │   ├── model_comparison/             # Model performance comparisons
│   │   └── weather_location/             # Environmental analysis plots
│   ├── data/                             # Processed result datasets
│   ├── models/                           # Trained ML models
│   └── reports/                          # Analysis reports
│       ├── agent_outputs/                # AI agent generated reports
│       └── statistical_summaries/        # Statistical analysis summaries
│
└── docs/                                  # Additional documentation
```

## Technology Stack

### Core Libraries
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, PyTorch
- **Deep Learning**: PyTorch Geometric (for GNN)
- **Visualization**: matplotlib, seaborn, plotly
- **Geospatial**: geopandas, folium
- **AI/LLM**: Google Gemini API

### Key Algorithms
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM
- **Graph Neural Networks**: Temporal Graph Networks for time-series anomaly detection
- **Clustering**: DBSCAN, K-Means for pattern grouping
- **Statistical Analysis**: Z-score, IQR-based outlier detection

## Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda for package management
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apple-health-data-analysis.git
cd apple-health-data-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up API keys for AI agents:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Quick Start

**📖 Important Documentation:**
- **[DATA_SETUP.md](DATA_SETUP.md)** - Setup guide for running with your own data
- **[docs/workflow.md](docs/workflow.md)** - Complete analysis pipeline and file dependencies
- **[docs/methodology.md](docs/methodology.md)** - Technical methodology and algorithms

#### Option 1: View Existing Results (No Setup)

All analysis results are in the `results/` directory:
```bash
# View visualizations
open results/visualizations/

# Explore data outputs
ls results/data/

# Read reports
cat results/reports/agent_outputs/*.txt
```

#### Option 2: Run Analysis with Your Own Data

1. **Export your Apple Health data** (see [DATA_SETUP.md](DATA_SETUP.md))
2. **Place data in local `data/` directory** (gitignored)
3. **Run the analysis pipeline**:

```bash
# Step 1: Load and extract data
cd notebooks
jupyter notebook load_ecg_data.ipynb

# Step 2: Weather and location analysis
jupyter notebook weather_location_workout_analysis.ipynb

# Step 3: GNN anomaly detection
jupyter notebook gnn_anomaly_detection.ipynb

# Step 4: AI agent analysis
cd ../src/agents
python run_health_agent.py

# Step 5: Model comparison
cd ../../notebooks
jupyter notebook anomaly_model_comparison.ipynb
```

For detailed workflow and dependencies, see [docs/workflow.md](docs/workflow.md).

## Key Results

### Anomaly Detection Performance
- **Models Evaluated**: 3 (Isolation Forest, LOF, One-Class SVM)
- **Consensus Detection**: Ensemble voting for robust anomaly identification
- **Visualization**: 21 comprehensive charts across 4 analysis domains

### Insights Generated
- Detected anomalies with contextual explanations
- Weather-performance correlations
- Location-based heart rate pattern differences
- Temporal trend analysis with seasonal variations

## Data Privacy

⚠️ **Important**: This repository does NOT contain any original health data. All personal Apple Health export data has been removed. Only analysis code, processed results, and trained models are included.

To use this project with your own data:
1. Export your Apple Health data (Settings > Health > Profile > Export Health Data)
2. Place the export in a local `data/` directory (not tracked by git)
3. Run the analysis notebooks with your data

## Project Highlights

### 1. Multi-Model Anomaly Detection
Compares three different anomaly detection algorithms and uses ensemble voting to identify consensus anomalies with high confidence.

### 2. Graph Neural Network Implementation
Custom PyTorch Geometric implementation for temporal health pattern analysis, capturing complex dependencies in time-series health data.

### 3. AI-Powered Interpretation
Leverages large language models (Gemini) to provide natural language interpretations of detected patterns and anomalies.

### 4. Environmental Context
Unique integration of weather and location data to understand how environmental factors affect health metrics.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- Apple Health for providing comprehensive health data export capabilities
- PyTorch Geometric team for GNN implementation tools
- Google Gemini for AI-powered insights
- The open-source data science community

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This project is for personal health insights and should not replace professional medical advice. Always consult healthcare providers for medical decisions.
