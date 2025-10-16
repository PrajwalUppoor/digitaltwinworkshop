# Digital Twin Workshop: From Prerequisites to Implementation
## A Hands-on Journey with AI, ML & Python

---

# Slide 1: Workshop Overview

## What You'll Learn Today
- ✅ Prerequisites: AI, ML, Digital Twin concepts
- ✅ Python techniques for Digital Twin development
- ✅ Step-by-step methodology to build a Digital Twin
- ✅ **Hands-on Project**: Industrial Equipment Predictive Maintenance
- ✅ Real Kaggle dataset implementation in Google Colab

## Duration: 3-4 Hours
**Format**: Theory (40%) + Hands-on Coding (60%)

---

# Slide 2: What is AI vs ML vs Digital Twin?

## The Technology Stack

```
┌─────────────────────────────────────────┐
│     Artificial Intelligence (AI)        │
│  ┌─────────────────────────────────┐   │
│  │   Machine Learning (ML)          │   │
│  │  ┌─────────────────────────┐    │   │
│  │  │  Deep Learning (DL)      │    │   │
│  │  │  - Neural Networks       │    │   │
│  │  │  - CNNs, RNNs, LSTMs     │    │   │
│  │  └─────────────────────────┘    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

         Applied to ↓

┌─────────────────────────────────────────┐
│         DIGITAL TWIN                     │
│  Virtual replica synchronized with       │
│  physical asset using AI/ML for:         │
│  • Monitoring • Prediction • Optimization│
└─────────────────────────────────────────┘
```

---

# Slide 3: Prerequisites - Knowledge Foundation

## Required Background

### 1️⃣ Python Programming
- ✓ Variables, loops, functions, classes
- ✓ List comprehensions, lambda functions
- ✓ File I/O, error handling

### 2️⃣ Mathematics (Basic)
- ✓ Statistics: Mean, median, std deviation
- ✓ Linear algebra: Vectors, matrices (conceptual)
- ✓ Calculus: Derivatives (why models learn)

### 3️⃣ Data Science Basics
- ✓ Pandas DataFrames
- ✓ Data cleaning & transformation
- ✓ Visualization (Matplotlib, Seaborn)

**Don't worry!** We'll reinforce these as we code.

---

# Slide 4: Prerequisites - Technical Setup

## Environment Setup (5 minutes)

### Option 1: Google Colab (Recommended for Workshop) ⭐
```
✓ No installation needed
✓ Free GPU access
✓ Pre-installed libraries
✓ Easy sharing
```
**Link**: https://colab.research.google.com

### Option 2: Local Jupyter
```bash
pip install jupyter numpy pandas scikit-learn matplotlib seaborn simpy
jupyter notebook
```

---

# Slide 5: What is a Digital Twin?

## Definition
> A **Digital Twin** is a virtual representation of a physical object or system that:
> - Mirrors the real-world asset in real-time
> - Uses data from sensors and IoT devices
> - Employs AI/ML for predictions and optimization
> - Enables "what-if" scenario testing

## Types of Digital Twins
1. **Component Twin** - Single part (motor, sensor)
2. **Asset Twin** - Complete machine (pump, conveyor)
3. **System Twin** - Entire facility (warehouse, factory)
4. **Process Twin** - Business workflow

---

# Slide 6: Digital Twin Architecture

```
Physical World              Digital World
─────────────              ──────────────

┌──────────────┐          ┌──────────────┐
│   Physical   │          │   Digital    │
│    Asset     │◄────────►│     Twin     │
│  (Machine)   │  IoT     │   (Model)    │
└──────────────┘  Data    └──────────────┘
      │                           │
      │ Sensors                   │ ML Models
      │ (Temp, Vibration,         │ Predictions
      │  Pressure, etc.)          │ Alerts
      │                           │
      └──────────┬────────────────┘
                 │
         ┌───────▼────────┐
         │  Data Pipeline │
         │  • Collection  │
         │  • Processing  │
         │  • Storage     │
         └────────────────┘
```

---

# Slide 7: Why Digital Twins Matter

## Business Value

| Benefit | Impact | Example |
|---------|--------|---------|
| 🔮 **Predictive Maintenance** | 30-50% ↓ downtime | Predict motor failure 2 weeks ahead |
| 💰 **Cost Reduction** | 20-40% ↓ costs | Optimize energy consumption |
| ⚡ **Performance Optimization** | 15-25% ↑ efficiency | Tune production parameters |
| 🎯 **Risk Mitigation** | 50-70% ↓ accidents | Test dangerous scenarios safely |
| 🚀 **Innovation** | Faster R&D | Test designs before building |

---

# Slide 8: Digital Twin Development Methodology

## 7-Step Framework

```
1. Define Objectives       → What problem are we solving?
2. Select Physical Asset   → Which system to model?
3. Identify Data Sources   → What sensors/data do we have?
4. Build Data Pipeline     → How to collect & store data?
5. Create Virtual Model    → How to represent the system?
6. Integrate ML/AI         → What predictions do we need?
7. Deploy & Monitor        → How to operationalize?
```

**Today's Focus**: Steps 1-6 with a complete example!

---

# Slide 9: Python Techniques for Digital Twins

## Core Libraries

### Data Handling
```python
import pandas as pd           # DataFrames
import numpy as np            # Numerical operations
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Interactive plots
```

### Machine Learning
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
```

### Simulation
```python
import simpy                  # Discrete-event simulation
```

---

# Slide 10: Our Hands-On Project

## 🏭 Industrial Equipment Predictive Maintenance

### The Scenario
- **Asset**: Industrial rotating machinery (pumps, motors, bearings)
- **Problem**: Unexpected failures cause costly downtime
- **Solution**: Build a Digital Twin that predicts failures

### Dataset
**Source**: Kaggle - Predictive Maintenance Dataset
- **Features**: Temperature, vibration, pressure, RPM, age
- **Target**: Machine failure (0=Normal, 1=Failure)
- **Size**: ~10,000 samples with 6 sensors

**We'll build this together in Google Colab!**

---

# Slide 11: Step 1 - Define Objectives

## Our Digital Twin Goals

### Primary Objective
🎯 Predict equipment failure 24-48 hours in advance

### Success Metrics
- ✅ **Accuracy**: >85% prediction accuracy
- ✅ **Precision**: Minimize false alarms (<10%)
- ✅ **Recall**: Catch >90% of actual failures
- ✅ **Latency**: Predictions within 5 seconds

### Business Impact
- 💰 Reduce unplanned downtime by 40%
- 🔧 Enable scheduled maintenance
- 📊 Improve asset utilization

---

# Slide 12: Step 2 - Select Physical Asset

## Asset Profile: Industrial Motor System

### Physical Characteristics
```
Motor Specification:
├── Type: 3-Phase Induction Motor
├── Power: 50 HP
├── Operating Speed: 1750 RPM
└── Environment: Continuous Operation

Critical Components:
├── Bearings (most failure-prone)
├── Windings (temperature sensitive)
└── Shaft (vibration monitoring)
```

### Failure Modes
1. **Bearing wear** → Increased vibration
2. **Overheating** → Temperature spikes
3. **Misalignment** → Abnormal pressure
4. **Overload** → Increased torque

---

# Slide 13: Step 3 - Identify Data Sources

## Sensor Configuration

| Sensor Type | Measurement | Frequency | Criticality |
|-------------|-------------|-----------|-------------|
| 🌡️ Temperature | °C | 1 Hz | High |
| 📊 Vibration | mm/s | 10 Hz | Critical |
| 🎚️ Pressure | PSI | 1 Hz | Medium |
| ⚙️ RPM | Revolutions | 1 Hz | High |
| ⏰ Operating Hours | Cumulative | On change | High |
| ⚡ Power Draw | Watts | 1 Hz | Medium |

### Data Quality Requirements
- ✓ 99%+ uptime for sensors
- ✓ <5% missing values
- ✓ Calibration every 6 months

---

# Slide 14: Step 4 - Build Data Pipeline

## Data Flow Architecture

```python
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Sensors   │─────►│  Edge Device │─────►│   Cloud DB  │
│  (IoT Tags) │ MQTT │  (Filtering) │ HTTP │ (TimeSeries)│
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  Data Lake      │
                                           │  (Parquet/CSV)  │
                                           └─────────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  ML Pipeline    │
                                           │  (Training)     │
                                           └─────────────────┘
```

### Data Processing Steps
1. Ingest → 2. Validate → 3. Clean → 4. Transform → 5. Store

---

# Slide 15: Step 5 - Create Virtual Model

## Digital Twin Components

### 1. State Representation
```python
class MachineState:
    temperature: float      # Current temp
    vibration: float        # Vibration level
    pressure: float         # Pressure reading
    rpm: float             # Rotation speed
    operating_hours: int   # Total runtime
    health_score: float    # 0-100 scale
    failure_probability: float  # ML prediction
```

### 2. Simulation Logic
- Physics-based model (optional)
- Statistical behavior model
- Time-series evolution

### 3. Visualization Dashboard
- Real-time gauges
- Trend charts
- Alert indicators

---

# Slide 16: Step 6 - Integrate ML/AI

## Machine Learning Pipeline

```
Historical Data
      ↓
Feature Engineering
├── Rolling averages
├── Rate of change
├── Statistical moments
└── Fourier features
      ↓
Model Training
├── Random Forest
├── XGBoost
└── LSTM (if time-series)
      ↓
Model Evaluation
├── Cross-validation
├── Confusion matrix
└── ROC-AUC score
      ↓
Deployment
└── Real-time inference
```

---

# Slide 17: ML Algorithms for Digital Twins

## Algorithm Selection Guide

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Random Forest** | Classification | Robust, interpretable | Memory intensive |
| **XGBoost** | High performance | Best accuracy | Needs tuning |
| **LSTM** | Time-series | Captures temporal patterns | Requires more data |
| **Isolation Forest** | Anomaly detection | Unsupervised | Less precise |
| **Prophet** | Forecasting | Handles seasonality | Regression only |

**Today**: We'll use Random Forest for interpretability

---

# Slide 18: Feature Engineering for Digital Twins

## Creating Predictive Features

### Raw Sensor Data
```
temperature = 85.3°C
vibration = 0.45 mm/s
```

### Engineered Features
```python
# Statistical
temp_rolling_mean_10min = df['temp'].rolling(600).mean()
vibration_std_1hr = df['vibration'].rolling(3600).std()

# Rates of Change
temp_delta = df['temp'].diff()
vibration_acceleration = df['vibration'].diff(2)

# Domain Knowledge
temp_above_threshold = (df['temp'] > 90).astype(int)
vibration_alarm_level = (df['vibration'] > 0.5).astype(int)

# Interaction Features
temp_vibration_product = df['temp'] * df['vibration']
```

---

# Slide 19: Model Evaluation Metrics

## How to Measure Success

### Confusion Matrix
```
                Predicted
              Normal  Failure
Actual Normal    TN      FP
       Failure   FN      TP
```

### Key Metrics
```python
Accuracy  = (TP + TN) / Total        # Overall correctness
Precision = TP / (TP + FP)           # Of predicted failures, how many are real?
Recall    = TP / (TP + FN)           # Of actual failures, how many did we catch?
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

### For Maintenance:
- 🎯 **High Recall** = Don't miss actual failures (safety)
- 🎯 **High Precision** = Don't cry wolf (cost)

---

# Slide 20: Step 7 - Deploy & Monitor

## Production Considerations

### Deployment Architecture
```
┌─────────────────────────────────────┐
│         Production System            │
├─────────────────────────────────────┤
│  ┌──────────┐      ┌─────────────┐ │
│  │  FastAPI │◄────►│ ML Model    │ │
│  │  Service │      │ (.pkl file) │ │
│  └──────────┘      └─────────────┘ │
│       │                             │
│       ▼                             │
│  ┌──────────┐      ┌─────────────┐ │
│  │ Database │      │  Alert      │ │
│  │ (Postgres)│     │  System     │ │
│  └──────────┘      └─────────────┘ │
└─────────────────────────────────────┘
```

### Monitoring Checklist
- ✅ Model drift detection
- ✅ Prediction latency
- ✅ API uptime
- ✅ Alert accuracy

---

# Slide 21: Hands-On Exercise Overview

## What We'll Code Together

### Part 1: Data Preparation (15 min)
1. Load Kaggle dataset
2. Explore data structure
3. Clean and validate

### Part 2: Feature Engineering (15 min)
4. Create rolling statistics
5. Add domain features
6. Split train/test sets

### Part 3: Model Development (20 min)
7. Train Random Forest
8. Evaluate performance
9. Feature importance analysis

### Part 4: Digital Twin Simulation (20 min)
10. Build virtual machine class
11. Real-time prediction loop
12. Visualization dashboard

---

# Slide 22: Coding Time! 🚀

## Let's Jump to Google Colab

### Workshop Notebook URL
```
https://colab.research.google.com/drive/[YOUR_NOTEBOOK_ID]
```

### What to Have Ready
1. ✅ Google account logged in
2. ✅ Kaggle account (for dataset download)
3. ✅ Positive attitude 😊

### If You Get Stuck
- 🙋 Raise hand for help
- 💬 Check chat for common issues
- 👥 Pair with neighbor

**Let's build our Digital Twin!**

---

# Slide 23: Code Walkthrough - Data Loading

```python
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load dataset from Kaggle
# Dataset: AI4I 2020 Predictive Maintenance Dataset
url = "https://raw.githubusercontent.com/..../predictive_maintenance.csv"
df = pd.read_csv(url)

# Step 3: Initial exploration
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFailure distribution:\n{df['Failure'].value_counts()}")
```

---

# Slide 24: Code Walkthrough - EDA

```python
# Exploratory Data Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Temperature distribution
sns.histplot(data=df, x='Temperature', hue='Failure', ax=axes[0,0])
axes[0,0].set_title('Temperature Distribution by Failure')

# Vibration distribution
sns.histplot(data=df, x='Vibration', hue='Failure', ax=axes[0,1])
axes[0,1].set_title('Vibration Distribution by Failure')

# Correlation heatmap
correlation = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation, annot=True, fmt='.2f', ax=axes[0,2])
axes[0,2].set_title('Feature Correlations')

# Box plots for outlier detection
sns.boxplot(data=df, x='Failure', y='Temperature', ax=axes[1,0])
sns.boxplot(data=df, x='Failure', y='Vibration', ax=axes[1,1])
sns.boxplot(data=df, x='Failure', y='Pressure', ax=axes[1,2])

plt.tight_layout()
plt.show()
```

---

# Slide 25: Code Walkthrough - Feature Engineering

```python
# Create advanced features
def engineer_features(df):
    df = df.copy()
    
    # Rolling statistics (last 10 samples)
    df['temp_rolling_mean'] = df['Temperature'].rolling(10, min_periods=1).mean()
    df['temp_rolling_std'] = df['Temperature'].rolling(10, min_periods=1).std()
    df['vib_rolling_mean'] = df['Vibration'].rolling(10, min_periods=1).mean()
    df['vib_rolling_std'] = df['Vibration'].rolling(10, min_periods=1).std()
    
    # Rate of change
    df['temp_delta'] = df['Temperature'].diff().fillna(0)
    df['vib_delta'] = df['Vibration'].diff().fillna(0)
    
    # Threshold crossings
    df['temp_alarm'] = (df['Temperature'] > df['Temperature'].quantile(0.9)).astype(int)
    df['vib_alarm'] = (df['Vibration'] > df['Vibration'].quantile(0.9)).astype(int)
    
    # Interaction features
    df['temp_vib_interaction'] = df['Temperature'] * df['Vibration']
    
    # Fill any remaining NaNs
    df = df.fillna(method='bfill').fillna(0)
    
    return df

df_engineered = engineer_features(df)
print(f"New feature count: {df_engineered.shape[1]}")
```

---

# Slide 26: Code Walkthrough - Model Training

```python
# Prepare data for ML
feature_cols = ['Temperature', 'Vibration', 'Pressure', 'RPM', 
                'Operating_Hours', 'temp_rolling_mean', 'temp_rolling_std',
                'vib_rolling_mean', 'vib_rolling_std', 'temp_delta', 
                'vib_delta', 'temp_alarm', 'vib_alarm', 'temp_vib_interaction']

X = df_engineered[feature_cols]
y = df_engineered['Failure']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced data
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("Model training complete!")
```

---

# Slide 27: Code Walkthrough - Model Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"""
Model Performance:
{'='*50}
Accuracy:  {accuracy:.2%}
Precision: {precision:.2%}  (Of predicted failures, {precision:.0%} were real)
Recall:    {recall:.2%}  (Caught {recall:.0%} of actual failures)
F1-Score:  {f1:.3f}
ROC-AUC:   {roc_auc:.3f}
""")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))
```

---

# Slide 28: Code Walkthrough - Feature Importance

```python
# Feature importance analysis
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 5 Predictive Features:")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"  {row['Feature']:.<30} {row['Importance']:.4f}")
```

---

# Slide 29: Code Walkthrough - Digital Twin Class

```python
import time
from datetime import datetime

class DigitalTwinMachine:
    """Virtual representation of physical machine"""
    
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        self.state = {
            'temperature': 75.0,
            'vibration': 0.3,
            'pressure': 100.0,
            'rpm': 1750,
            'operating_hours': 0,
            'health_score': 100.0,
            'failure_probability': 0.0,
            'status': 'NORMAL',
            'alerts': []
        }
        self.history = []
    
    def update_sensors(self, sensor_data):
        """Receive new sensor readings"""
        self.state['temperature'] = sensor_data.get('temperature', self.state['temperature'])
        self.state['vibration'] = sensor_data.get('vibration', self.state['vibration'])
        self.state['pressure'] = sensor_data.get('pressure', self.state['pressure'])
        self.state['rpm'] = sensor_data.get('rpm', self.state['rpm'])
        self.state['operating_hours'] += 1
        
        # Predict failure probability
        self._predict_failure()
        self._check_alerts()
        self._update_health_score()
        
        # Store history
        self.history.append({
            'timestamp': datetime.now(),
            **self.state
        })
```

---

# Slide 30: Code Walkthrough - Digital Twin (cont.)

```python
    def _predict_failure(self):
        """Run ML model prediction"""
        # Prepare features (simplified - in production, use engineered features)
        features = np.array([[
            self.state['temperature'],
            self.state['vibration'],
            self.state['pressure'],
            self.state['rpm'],
            self.state['operating_hours'],
            # Add engineered features here
            0, 0, 0, 0, 0, 0, 0, 0, 0  # Placeholders
        ]])
        
        # Get prediction
        proba = self.model.predict_proba(features)[0][1]
        self.state['failure_probability'] = proba
        
        # Update status
        if proba > 0.7:
            self.state['status'] = 'CRITICAL'
        elif proba > 0.4:
            self.state['status'] = 'WARNING'
        else:
            self.state['status'] = 'NORMAL'
    
    def _check_alerts(self):
        """Generate maintenance alerts"""
        self.state['alerts'] = []
        
        if self.state['temperature'] > 90:
            self.state['alerts'].append('HIGH_TEMPERATURE')
        if self.state['vibration'] > 0.6:
            self.state['alerts'].append('HIGH_VIBRATION')
        if self.state['failure_probability'] > 0.7:
            self.state['alerts'].append('FAILURE_IMMINENT')
```

---

# Slide 31: Code Walkthrough - Simulation Loop

```python
    def _update_health_score(self):
        """Calculate overall health (0-100)"""
        # Invert failure probability to health score
        self.health_score = (1 - self.state['failure_probability']) * 100
    
    def get_dashboard(self):
        """Return dashboard view"""
        return {
            'Machine Status': self.state['status'],
            'Health Score': f"{self.state['health_score']:.1f}%",
            'Failure Risk': f"{self.state['failure_probability']:.1%}",
            'Temperature': f"{self.state['temperature']:.1f}°C",
            'Vibration': f"{self.state['vibration']:.2f} mm/s",
            'Operating Hours': self.state['operating_hours'],
            'Active Alerts': ', '.join(self.state['alerts']) or 'None'
        }

# Initialize Digital Twin
twin = DigitalTwinMachine(rf_model, feature_cols)

# Simulate real-time updates
print("Digital Twin Simulation Running...\n")
for i in range(10):
    # Simulate sensor readings (in production, this comes from IoT)
    sensor_data = {
        'temperature': np.random.normal(80, 5),
        'vibration': np.random.normal(0.4, 0.1),
        'pressure': np.random.normal(100, 3),
        'rpm': np.random.normal(1750, 20)
    }
    
    twin.update_sensors(sensor_data)
    dashboard = twin.get_dashboard()
    
    print(f"Update {i+1}:")
    for key, value in dashboard.items():
        print(f"  {key}: {value}")
    print()
    time.sleep(1)
```

---

# Slide 32: Code Walkthrough - Visualization Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(twin):
    """Create interactive dashboard"""
    history_df = pd.DataFrame(twin.history)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature', 'Vibration', 'Health Score', 
                       'Failure Probability', 'Operating Hours', 'Status'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "indicator"}]]
    )
    
    # Temperature
    fig.add_trace(go.Scatter(y=history_df['temperature'], name='Temp'), row=1, col=1)
    
    # Vibration
    fig.add_trace(go.Scatter(y=history_df['vibration'], name='Vib'), row=1, col=2)
    
    # Health Score
    fig.add_trace(go.Scatter(y=history_df['health_score'], name='Health'), row=2, col=1)
    
    # Failure Probability
    fig.add_trace(go.Scatter(y=history_df['failure_probability'], name='Risk'), row=2, col=2)
    
    # Operating Hours
    fig.add_trace(go.Scatter(y=history_df['operating_hours'], name='Hours'), row=3, col=1)
    
    # Current Status Indicator
    current_health = twin.state['health_score']
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_health,
        title={'text': "Current Health"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkgreen" if current_health > 70 else "orange" if current_health > 40 else "red"}}
    ), row=3, col=2)
    
    fig.update_layout(height=900, showlegend=False, title_text="Digital Twin Dashboard")
    fig.show()

create_dashboard(twin)
```

---

# Slide 33: Real-World Integration Points

## From Colab to Production

### 1. Data Integration
```python
# Development (CSV)
df = pd.read_csv('data.csv')

# Production (Time-series DB)
from sqlalchemy import create_engine
engine = create_engine('postgresql://...')
df = pd.read_sql("SELECT * FROM sensor_data WHERE ts > NOW() - INTERVAL '1 hour'", engine)
```

### 2. Model Deployment
```python
# Save trained model
import joblib
joblib.dump(rf_model, 'digital_twin_model.pkl')

# Load in production
model = joblib.load('digital_twin_model.pkl')
```

### 3. API Endpoint
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(sensor_data: SensorData):
    features = prepare_features(sensor_data)
    prediction = model.predict_proba([features])[0][1]
    return {"failure_probability": prediction}
```

---

# Slide 34: Advanced Digital Twin Capabilities

## Beyond Prediction

### 1. What-If Simulation
```python
# Test scenario: What if temperature rises 10°C?
twin_scenario = twin.copy()
twin_scenario.state['temperature'] += 10
twin_scenario._predict_failure()
print(f"New failure risk: {twin_scenario.state['failure_probability']:.1%}")
```

### 2. Optimization
```python
# Find optimal operating parameters
from scipy.optimize import minimize

def cost_function(params):
    temp, rpm = params
    # Minimize cost = downtime_cost * failure_prob + energy_cost * rpm
    failure_prob = model.predict_proba([[temp, 0.4, 100, rpm, 1000, ...]])[0][1]
    return 10000 * failure_prob + 0.1 * rpm

result = minimize(cost_function, x0=[80, 1750], bounds=[(70, 95), (1500, 2000)])
print(f"Optimal: Temp={result.x[0]:.1f}°C, RPM={result.x[1]:.0f}")
```

### 3. Anomaly Detection
```python
from sklearn.ensemble import IsolationForest
anomaly_detector = IsolationForest(contamination=0.1)
anomalies = anomaly_detector.fit_predict(df[feature_cols])
```

---

# Slide 35: Common Pitfalls & Solutions

## Lessons from Production

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Data Drift** | Model accuracy drops over time | Monitor distributions, retrain quarterly |
| **Class Imbalance** | Model predicts all "Normal" | Use SMOTE, class weights, or threshold tuning |
| **Sensor Failures** | Missing/erratic data | Implement fallback logic, interpolation |
| **Latency** | Predictions take >5 seconds | Cache feature engineering, use model quantization |
| **False Alarms** | Too many warnings | Tune threshold, add confirmation window |
| **Missed Failures** | Failures not predicted | Add more features, collect more failure data |

---

# Slide 36: Model Monitoring in Production

## Continuous Validation

### Metrics to Track
```python
# 1. Prediction Distribution Shift
current_predictions = model.predict_proba(X_current)[:, 1]
baseline_predictions = model.predict_proba(X_baseline)[:, 1]

from scipy.stats import ks_2samp
statistic, p_value = ks_2samp(baseline_predictions, current_predictions)
if p_value < 0.05:
    print("⚠️ Prediction drift detected!")

# 2. Feature Distribution Shift
from scipy.spatial.distance import jensenshannon
for col in feature_cols:
    hist_current, _ = np.histogram(X_current[col], bins=50, density=True)
    hist_baseline, _ = np.histogram(X_baseline[col], bins=50, density=True)
    js_divergence = jensenshannon(hist_current, hist_baseline)
    if js_divergence > 0.1:
        print(f"⚠️ Feature '{col}' has drifted (JS={js_divergence:.3f})")

# 3. Model Performance (when labels arrive)
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
# Alert if precision or recall drops below threshold
```

---

# Slide 37: Scaling Considerations

## From 1 Machine to 1000 Machines

### Architecture Evolution

#### Phase 1: Single Machine (Today)
- Monolithic notebook/script
- Local CSV files
- Manual monitoring

#### Phase 2: Fleet Monitoring
- Centralized database (InfluxDB, TimescaleDB)
- Batch prediction pipeline (Airflow)
- Alerting system (PagerDuty)

#### Phase 3: Enterprise Platform
- Microservices (Docker/Kubernetes)
- Streaming predictions (Kafka + Flink)
- ML Ops platform (MLflow, Kubeflow)
- Multi-tenant Digital Twin service

---

# Slide 38: Cost-Benefit Analysis

## ROI of Digital Twin Implementation

### Investment
```
Year 1 Costs:
├── Development: $150K (2 engineers × 6 months)
├── Infrastructure: $50K (cloud, sensors)
├── Training: $20K
└── TOTAL: $220K
```

### Returns (Annual)
```
Downtime Reduction:
├── Before: 10 failures/year × 24 hrs × $5K/hr = $1.2M
├── After:  3 failures/year × 24 hrs × $5K/hr = $360K
└── Savings: $840K

Maintenance Optimization:
├── Reduced overtime: $100K
├── Parts optimization: $50K
└── Savings: $150K

TOTAL ANNUAL BENEFIT: $990K
ROI: 350% (payback in 3 months!)
```

---

# Slide 39: Next Steps - Your Learning Path

## After This Workshop

### Week 1-2: Reinforcement
- ✅ Re-run the Colab notebook with different parameters
- ✅ Try another Kaggle dataset (HVAC, Wind Turbines)
- ✅ Add new features (FFT for vibration)

### Month 1: Intermediate
- 📚 Study time-series models (LSTM, Prophet)
- 🔧 Build a FastAPI endpoint for your model
- 📊 Create a Streamlit dashboard

### Month 2-3: Advanced
- 🏗️ Deploy to cloud (AWS SageMaker, Azure ML)
- 🔄 Implement CI/CD for model updates
- 📡 Integrate with real IoT data (if available)

### Month 4+: Production
- 🚀 Build a multi-asset digital twin platform
- 📈 Implement reinforcement learning for optimization
- 🎯 Present case study at your organization

---

# Slide 40: Resources & Community

## Continue Your Journey

### 📚 Recommended Reading
- **Books**:
  - "Hands-On Machine Learning" by Aurélien Géron
  - "Digital Twin: The Ultimate Guide" by Michael Grieves
  - "Python for Data Analysis" by Wes McKinney

### 🌐 Online Resources
- Kaggle Learn: https://www.kaggle.com/learn
- Fast.ai Courses: https://www.fast.ai
- Google Colab Examples: https://colab.research.google.com

### 👥 Communities
- r/MachineLearning (Reddit)
- Towards Data Science (Medium)
- MLOps Community (Slack)

### 🎓 Certifications
- Google Professional ML Engineer
- AWS Certified Machine Learning
- Microsoft Azure AI Engineer

---

# Slide 41: Q&A - Common Questions

## Frequently Asked Questions

**Q: How much data do I need to train a model?**
A: Minimum 1000 samples, ideally 10K+. For rare failures, use synthetic data (SMOTE) or transfer learning.

**Q: Can I use this for non-industrial applications?**
A: Yes! Digital Twins work for: Healthcare (patient vitals), Smart cities (traffic), Buildings (HVAC), Agriculture (crop health).

**Q: What if I don't have real sensor data?**
A: Start with simulated data (SimPy), public datasets (Kaggle, UCI), or small IoT kit (Raspberry Pi + sensors ~$50).

**Q: Is cloud required or can I run locally?**
A: Local works for prototypes. Cloud needed for: Multiple assets, real-time processing, team collaboration, high availability.

**Q: How often should I retrain models?**
A: Monitor drift. Typical: Quarterly scheduled + triggered retraining if accuracy drops >5%.

---

# Slide 42: Hands-On Exercise Recap

## What We Built Together ✅

### Achievements
1. ✅ Loaded & explored real-world sensor data
2. ✅ Engineered 14 predictive features
3. ✅ Trained Random Forest with 85%+ accuracy
4. ✅ Built Python class for Digital Twin
5. ✅ Simulated real-time predictions
6. ✅ Created interactive dashboard

### Skills Gained
- Data preprocessing & validation
- Feature engineering techniques
- ML model training & evaluation
- Object-oriented programming for twins
- Time-series visualization

### Your Turn
Customize the code with your own:
- Thresholds
- Features
- Algorithms
- Visualizations

---

# Slide 43: Challenge Assignment 🏆

## Take It Further

### Challenge 1: Enhanced Features (Beginner)
Add 3 new features:
- Frequency domain features (FFT)
- Lag features (previous 5 time steps)
- Categorical encoding (machine type, shift)

**Goal**: Improve accuracy by 2%

### Challenge 2: LSTM Model (Intermediate)
Replace Random Forest with LSTM:
- Input: Sequences of 50 time steps
- Output: Failure probability
- Compare performance to RF

### Challenge 3: Multi-Asset Fleet (Advanced)
Extend to 10 machines:
- Train one model or individual models?
- Build centralized dashboard
- Implement priority-based alerting

**Share your results in the Discord/Slack!**

---

# Slide 44: Digital Twin Maturity Model

## Where Are You on the Journey?

```
Level 1: BASIC MONITORING
├── Real-time dashboards
├── Threshold alerts
└── Historical trends

Level 2: PREDICTIVE
├── ML-based predictions
├── Anomaly detection
└── Failure forecasting

Level 3: PRESCRIPTIVE  ← Today's Workshop
├── What-if simulations
├── Optimization recommendations
└── Automated decision support

Level 4: AUTONOMOUS
├── Self-healing systems
├── Closed-loop control
└── AI-driven operations

Level 5: COGNITIVE
├── Multi-domain reasoning
├── Continuous learning
└── Emergent intelligence
```

**You're now at Level 3! 🎉**

---

# Slide 45: Case Studies - Real World Success

## Digital Twins in Action

### 1. General Electric - Wind Turbines
- **Asset**: 50,000+ turbines worldwide
- **Twin**: Predicts failures 2 weeks ahead
- **Result**: 20% ↑ uptime, $100M+ savings

### 2. Siemens - Gas Turbines
- **Asset**: Power generation turbines
- **Twin**: Optimizes fuel efficiency
- **Result**: 10% ↓ fuel consumption

### 3. Tesla - Vehicle Fleet
- **Asset**: 1M+ electric vehicles
- **Twin**: Battery degradation prediction
- **Result**: Improved warranty planning

### 4. Walmart - Cold Chain (Our Next Module!)
- **Asset**: Refrigeration systems
- **Twin**: Prevents food spoilage
- **Result**: 30% ↓ waste, better compliance

---

# Slide 46: Ethics & Responsible AI

## Important Considerations

### Data Privacy
```python
# Always anonymize sensitive data
df['machine_id'] = df['machine_id'].apply(hash)  # One-way hash

# Never train on PII
df = df.drop(['operator_name', 'location_gps'], axis=1)
```

### Bias & Fairness
- ⚠️ Training only on new machines → bias against old equipment
- ✅ Solution: Stratified sampling across age groups

### Explainability
```python
# Use SHAP for model interpretation
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Safety-Critical Systems
- 🔴 Never fully automate critical decisions
- ✅ Human-in-the-loop for final actions
- ✅ Extensive testing & validation

---

# Slide 47: Workshop Summary

## Key Takeaways 🎯

### Conceptual
1. Digital Twin = Virtual + Physical + Data + AI
2. Predictive > Reactive maintenance
3. Feature engineering is 80% of success
4. Monitor, monitor, monitor in production

### Technical
1. Random Forest: Great starting algorithm
2. Class imbalance: Use balanced weights
3. Feature importance: Guides domain understanding
4. Object-oriented twins: Scalable architecture

### Practical
1. Start with Colab, graduate to production
2. Kaggle datasets: Perfect for learning
3. Visualizations: Critical for stakeholder buy-in
4. ROI: Digital Twins pay for themselves quickly

---

# Slide 48: Evaluation & Feedback

## Help Us Improve

### Quick Poll (1 minute)
1. Rate workshop content (1-5): _____
2. Pace: Too Fast / Just Right / Too Slow
3. Hands-on time: More / Enough / Less
4. Most valuable part: _______________
5. What to add: _______________

### Share Your Project
- Upload your Colab notebook to GitHub
- Tag: #DigitalTwinWorkshop
- We'll feature the best ones!

### Stay Connected
- LinkedIn: [Your Profile]
- GitHub: [Workshop Repo]
- Email: workshop@example.com
- Discord: [Community Link]

---

# Slide 49: Certificate of Completion 🏆

```
┌────────────────────────────────────────────┐
│                                            │
│       CERTIFICATE OF COMPLETION            │
│                                            │
│         Digital Twin Workshop              │
│   AI/ML for Predictive Maintenance         │
│                                            │
│         Awarded to: [Your Name]            │
│                                            │
│         Date: October 13, 2025             │
│                                            │
│  Skills Demonstrated:                      │
│  ✓ Data preparation & feature engineering  │
│  ✓ ML model development & evaluation       │
│  ✓ Digital Twin architecture design        │
│  ✓ Real-time simulation & visualization    │
│                                            │
│         Instructor: [Instructor Name]      │
│                                            │
└────────────────────────────────────────────┘
```

**To receive**: Email screenshot of your working Colab notebook!

---

# Slide 50: Thank You! 🙏

## You're Now a Digital Twin Developer!

### What You've Achieved Today
- ✅ Built end-to-end ML pipeline
- ✅ Created functional Digital Twin
- ✅ Deployed predictive model
- ✅ Gained production-ready skills

### Your Next Steps
1. Complete challenge assignments
2. Explore additional datasets
3. Share your learnings
4. Join our community

### Remember
> "The best way to predict the future is to create it."  
> – Peter Drucker

**Now go build amazing Digital Twins! 🚀**

---

## Contact & Resources

📧 Email: digital.twin.workshop@example.com  
💻 GitHub: github.com/your-repo/digital-twin-workshop  
📱 Twitter: @DigitalTwinEdu  
💬 Discord: discord.gg/digitaltwin  

**Workshop Materials**:  
- Slides: [Link]  
- Colab Notebook: [Link]  
- Dataset: [Kaggle Link]  
- Additional Resources: [GitHub]

---

# END OF PRESENTATION
**Questions? Let's discuss!**
