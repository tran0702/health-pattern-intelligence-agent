# Analysis Methodology

## Overview

This document describes the analytical approaches, algorithms, and methodologies used in the Apple Health Data Analysis project.

## Data Sources

### Primary Data
- **Apple Health Export**: XML format containing health metrics, workouts, and device data
- **ECG Data**: Raw electrocardiogram measurements in CSV format
- **Workout Routes**: GPX files with GPS coordinates and elevation data

### External Data
- **Weather Data**: Historical weather information for workout analysis
- **Location Data**: Geographic context for activity patterns

## Analysis Pipeline

### 1. Data Extraction and Preprocessing

#### XML Parsing
- Parse Apple Health export XML file
- Extract relevant health records (heart rate, steps, workouts, etc.)
- Convert timestamps to uniform timezone-aware datetime objects
- Handle missing values and data quality issues

#### Data Cleaning
- Remove duplicate records
- Filter out invalid measurements (e.g., impossible heart rates)
- Standardize units across different device sources
- Interpolate missing values where appropriate

### 2. Feature Engineering

#### Temporal Features
- Hour of day, day of week, month, season
- Time since last workout
- Moving averages and rolling statistics
- Lag features for time-series analysis

#### Contextual Features
- Workout type and intensity
- Weather conditions (temperature, humidity, precipitation)
- Location (indoor/outdoor, elevation, geographic region)
- Device source (Apple Watch model, iPhone)

#### Aggregated Metrics
- Daily resting heart rate (RHR)
- Heart rate recovery (HRR) post-workout
- Training load estimation
- Weekly and monthly activity summaries

### 3. Anomaly Detection Approaches

#### Method 1: Isolation Forest
- **Algorithm**: Ensemble of isolation trees
- **Parameters**: 100 estimators, contamination=0.1
- **Rationale**: Effective for high-dimensional data, doesn't assume data distribution
- **Output**: Anomaly scores and binary labels

#### Method 2: Local Outlier Factor (LOF)
- **Algorithm**: Density-based outlier detection
- **Parameters**: n_neighbors=20, contamination=0.1
- **Rationale**: Captures local density deviations
- **Output**: Negative outlier factor scores

#### Method 3: One-Class SVM
- **Algorithm**: Support Vector Machine for novelty detection
- **Parameters**: nu=0.1, kernel='rbf', gamma='scale'
- **Rationale**: Learns decision boundary of normal behavior
- **Output**: Distance to decision boundary

#### Ensemble Approach
- Combine predictions from all three methods
- Use consensus voting (2+ models agree = high confidence anomaly)
- Weight by model-specific performance metrics
- Generate confidence scores for each detection

### 4. Graph Neural Network (GNN) Implementation

#### Architecture
- **Model**: Temporal Graph Convolutional Network
- **Layers**: 2 graph convolution layers + fully connected layers
- **Input**: Time-series health metrics as graph nodes
- **Edges**: Temporal connections and feature correlations

#### Training Process
- **Loss Function**: Binary cross-entropy for anomaly classification
- **Optimizer**: Adam with learning rate 0.001
- **Training Data**: 80/20 train-validation split
- **Epochs**: 100 with early stopping
- **Regularization**: Dropout (p=0.3) to prevent overfitting

#### Temporal Modeling
- Sliding window approach (7-day windows)
- Node features: Daily aggregated health metrics
- Edge features: Temporal distance and correlation strength
- Graph construction: K-nearest temporal neighbors

### 5. Statistical Analysis

#### Descriptive Statistics
- Mean, median, standard deviation for all metrics
- Percentile analysis (25th, 50th, 75th, 95th)
- Distribution shape assessment (skewness, kurtosis)

#### Hypothesis Testing
- T-tests for comparing groups (e.g., indoor vs outdoor workouts)
- ANOVA for multi-group comparisons
- Chi-square tests for categorical associations
- Significance level: α = 0.05

#### Correlation Analysis
- Pearson correlation for linear relationships
- Spearman correlation for monotonic relationships
- Cross-correlation for time-lagged relationships
- Partial correlation controlling for confounders

### 6. Environmental Context Integration

#### Weather Data Integration
- Match workout timestamps to nearest weather observation
- Calculate derived metrics (heat index, wind chill, feels-like temp)
- Analyze impact on heart rate and performance

#### Location Analysis
- Geocode workout routes using GPS coordinates
- Classify locations (home, work, park, gym, etc.)
- Calculate elevation gain and terrain difficulty
- Identify location-specific heart rate patterns

### 7. AI Agent Implementation

#### Pattern Recognition Agent
- **Purpose**: Automated insight generation
- **Approach**: Rule-based pattern detection + LLM interpretation
- **Input**: Detected anomalies and statistical summaries
- **Output**: Natural language health insights

#### Gemini-Enhanced Agent
- **Purpose**: Contextual anomaly interpretation
- **Approach**: LLM-powered analysis of detected patterns
- **Prompts**: Structured templates with health context
- **Output**: Detailed explanations and recommendations

#### Agent Workflow
1. Extract relevant health data
2. Run anomaly detection pipeline
3. Identify top anomalies by severity
4. Generate context for each anomaly
5. Use LLM to interpret patterns
6. Produce comprehensive health report

## Evaluation Metrics

### Anomaly Detection
- **Precision**: Proportion of detected anomalies that are true anomalies
- **Recall**: Proportion of true anomalies that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Model Comparison
- **Agreement Score**: Percentage of consensus predictions across models
- **Disagreement Analysis**: Cases where models differ
- **Confidence Calibration**: Alignment of predicted confidence with accuracy

### Clinical Relevance (Manual Review)
- False positive rate for known normal patterns
- True positive rate for documented health events
- Actionability of generated insights

## Validation Approach

### Cross-Validation
- Time-series aware splitting (no data leakage)
- Rolling window validation for temporal models
- Stratified sampling to maintain class balance

### Robustness Checks
- Sensitivity analysis for hyperparameters
- Performance on different time periods
- Consistency across different activity types

### Manual Verification
- Expert review of high-confidence anomalies
- Comparison with known health events
- User feedback on insight relevance

## Limitations and Considerations

### Data Quality
- Self-reported data may be incomplete
- Device measurement errors and inconsistencies
- Missing data during periods of non-wear

### Algorithmic Limitations
- Anomaly detection is unsupervised (no ground truth labels)
- Model performance depends on data quality and completeness
- Interpretability vs. performance trade-offs

### Privacy and Ethics
- All personal data removed from public repository
- No medical diagnosis or treatment recommendations
- Analysis for personal insights only

### Generalizability
- Models trained on individual data may not generalize
- Population-level patterns may differ significantly
- Environmental and lifestyle factors highly personal

## Future Improvements

- Incorporate additional health metrics (sleep, nutrition, stress)
- Develop personalized baseline models
- Real-time anomaly detection and alerting
- Integration with medical knowledge graphs
- Federated learning for privacy-preserving population analysis
- Causal inference for intervention recommendations

## References

- Isolation Forest: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
- Local Outlier Factor: Breunig, M. M., et al. (2000)
- Graph Neural Networks: Hamilton, W. L., Ying, R., & Leskovec, J. (2017)
- Time-Series Anomaly Detection: Ahmad, S., et al. (2017)

---

*Last Updated: February 2026*
