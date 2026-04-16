# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

### CAP5771 — Introduction to Data Science | University of Florida

# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

### CAP5771 — Introduction to Data Science | University of Florida

---

## 🚀 Streamlit Deployment

This project includes a **Streamlit dashboard (`app.py`)** for direct deployment.

### Deployment Settings

- **Main file:** `app.py`
- **Python version (Advanced Settings):** `Python 3.12`
- **Start command:**

```bash
streamlit run app.py
```

### Required Files (same directory as `app.py`)

- `batting_clean.csv`
- `bowling_clean.csv`
- All `.pkl` model files
- `bat_features.pkl`
- `bowl_features.pkl`
- `requirements.txt`
- `runtime.txt`

### Notes

- Uses relative paths (`BASE_DIR`) → no path changes needed
- No database required at runtime
- No API keys needed
- Models are pre-trained → inference only

------

## Project Overview

This project analyzes how competitive pressure influences player performance in the Indian Premier League (IPL) and builds regression models to predict future performance under similar match conditions.

Competitive pressure is operationalized using tournament stage:

- **League** (baseline)
- **Playoffs**
- **Final**

The dataset spans IPL seasons from **2008–2024** and is derived entirely from structured ball-by-ball match records provided by [Cricsheet](https://cricsheet.org).

This project integrates:
- Pressure-based performance analysis
- Risk and variability measurement
- Structured database design
- Data wrangling and feature engineering
- Player-level regression modeling
- Interactive dashboard for result communication

All analysis is fully reproducible using cricket-derived data only.

---

## Research Objectives

### 1. Descriptive Objective — Performance Under Pressure

Do IPL players exhibit higher performance risk and variability in pressure matches compared to league matches?

The analysis evaluates stage-based shifts in:

**Batting**
- Average runs per innings
- Variance of runs
- Probability of low score (runs < 10)
- Stage-based Pressure Impact Index

**Bowling**
- Economy rate
- Variance of economy
- Probability of high-cost spell
- Wickets per match
- Stage-based Pressure Impact Index

Pressure matches are defined as **Playoffs + Finals**, with League matches serving as the baseline.

---

### 2. Predictive Objective — Player-Level Regression

Two separate modeling pipelines are constructed:

**Batting Regression Model** — Target: Runs scored

Features include: historical average runs, run variance, low-score probability, match stage, venue, season, date-derived features (month, weekday)

**Bowling Regression Model** — Target: Economy rate

Features include: historical economy rate, wicket rate, dot-ball rate, match stage, venue, season, date-derived features

All features are computed using only historical matches prior to prediction to avoid data leakage.

---

## Repository Structure## Repository Structure

```
CAP5771-nvasanthakumar/
│
├── data_wrangling.ipynb                        # Data cleaning and feature engineering pipeline
├── data_modeling_data_visualization.ipynb      # ML models and interactive dashboard
├── batting_clean.csv                           # Cleaned batting dataset (16,515 rows)
├── bowling_clean.csv                           # Cleaned bowling dataset (12,978 rows)
├── ipl_cricket_only.db                         # SQLite database (Cricsheet, 2008–2024)
├── schema.png                                  # Entity-relationship diagram
├── data_dictionary.pdf                         # Variable definitions and descriptions
├── requirements.txt                            # Python dependencies
├── app.py                                      # Streamlit deployment dashboard
├── bat_features.pkl                            # Saved batting feature list
├── bowl_features.pkl                           # Saved bowling feature list
├── README.md                                   # This file
│
└── diary/
    ├── Module6_DataWranglingP1_Cleaning_Diary.txt
    ├── DataWranglingP2_FeatureEngineering_Diary.txt
    ├── DataModelingP1_Fundamentals_Diary.txt
    ├── DataModelingP2_Evaluation_Diary.txt
    └── DataVisualization_Diary.txt
```

---

## Data Source

All data is obtained from **Cricsheet** — https://cricsheet.org

Processing steps:
1. Parsed YAML files using Python
2. Aggregated to player–match level
3. Stored in a structured SQLite relational database

---

## Database Design

A SQLite database (`ipl_cricket_only.db`) provides structured storage and reproducibility.

Core tables: `matches`, `players`, `player_match_batting`, `player_match_bowling`

The variable `stage_bucket` (league / playoffs / final) serves as the objective pressure proxy. A schema diagram (`schema.png`) is included in the repository.

---

## Data Wrangling Pipeline

**Part I — Cleaning**
- Data type correction (dates, numeric fields)
- Duplicate handling (player–match uniqueness)
- Structural missing value handling
- Standardization of categorical fields
- Validation of impossible values

**Part II — Feature Engineering**
- Derived metrics (e.g., runs per ball)
- Pressure indicator variables
- Date feature extraction (year, month, weekday)
- Season year extraction for time-based splitting
- Stage bucket classification (league / playoffs / final)
- Categorical encoding and feature selection
- Correlation inspection

---

## How to Reproduce

### Step 1 — Data Wrangling

1. Open `data_wrangling.ipynb` in Google Colab
2. Upload to `/content/`: `ipl_cricket_only.db`, `matches.csv`, `players.csv`, `player_match_batting.csv`, `player_match_bowling.csv`
3. Click **Runtime → Run All**
4. Outputs: `batting_clean.csv`, `bowling_clean.csv`

### Step 2 — Data Modeling and Visualization

1. Open `data_modeling_data_visualization.ipynb` in Google Colab
2. Upload `batting_clean.csv` and `bowling_clean.csv` to `/content/`
3. Click **Runtime → Run All**
4. All model results, charts, and the dashboard generate automatically

No external APIs or credentials are required.

---

## Dashboard

The interactive dashboard has six tabs:

| Tab | Description |
|-----|-------------|
| 📊 Model Performance | R² comparison across all five models for batting and bowling |
| 🔍 Feature Importance | Top 10 features driving batting and bowling predictions |
| 🎯 Actual vs Predicted | Scatter plots showing prediction accuracy for both models |
| 📈 Stage Analysis | Average performance by match stage (league vs playoffs vs final) |
| 🏆 Player Rankings | Top batters and bowlers filtered by venue, season and match type |
| 🏏 Player Prediction | Predict any player at any venue, stage and season |

---

## Models

| Model | Task | Test R² |
|-------|------|---------|
| Linear Regression | Batting (baseline) | 0.8931 |
| Ridge Regression | Batting | 0.9187 |
| Lasso Regression | Batting | 0.9160 |
| Elastic Net | Batting | 0.9070 |
| Random Forest | Batting (best) | **0.9506** |
| Linear Regression | Bowling (baseline) | 0.7547 |
| Ridge Regression | Bowling (best) | **0.8611** |
| Lasso Regression | Bowling | 0.7874 |
| Elastic Net | Bowling | 0.7779 |
| Random Forest | Bowling | 0.7817 |

**Time-based split:** Train: 2008–2021 | Validation: 2022 | Test: 2023–2024

---

## Key Results

**Batting:** Random Forest achieves test R² of 0.9561, confirming that nonlinear relationships between player identity, venue, and match stage drive batting performance.

**Bowling:** Ridge Regression generalizes best with test R² of 0.8611, suggesting bowling economy benefits from regularization to handle noise and variability.

**Prediction examples:**
- Shubman Gill: predicted 35.65 runs in playoffs vs 30.14 in league matches
- JJ Bumrah: predicted economy of 7.06 in finals vs 7.56 in league matches

---

## Streamlit Deployment

**Main file:** `app.py` | **Python version:** 3.12

```bash
streamlit run app.py
```

Required files in the same directory as `app.py`:
- `batting_clean.csv`, `bowling_clean.csv`
- `bat_features.pkl`, `bowl_features.pkl`
- All `.pkl` model files
- `requirements.txt`, `runtime.txt`

Notes: Uses relative paths — no path changes needed. No database or API keys required at runtime. Models are pre-trained (inference only).

---

## Technical Stack

Python 3.10+ · pandas · NumPy · matplotlib · seaborn · SQLite · scikit-learn · ipywidgets · Streamlit

---

## Summary

This project demonstrates end-to-end data science practice: structured database design, transparent wrangling and feature engineering, pressure-aware performance analysis, five-model regression comparison with proper evaluation, cross-validation and residual analysis, a context-aware prediction system with venue and stage adjustments, and an interactive dashboard for result communication. All outputs are fully reproducible from repository contents.
