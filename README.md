# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

### CAP5771 — Introduction to Data Science  
University of Florida

---

## 🚀 Streamlit Deployment

This project includes a **Streamlit dashboard (`app.py`)** for direct deployment.

### Deployment Settings
- **Main file:** `app.py`  
- **Python version (Advanced Settings):** `Python 3.12`  
- **Start command:**
```bash
streamlit run app.### CAP5771 — Introduction to Data Science
University of Florid# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

### CAP5771 — Introduction to Data Science  
University of Florida

---

## 🚀 Streamlit Deployment

This project includes a **Streamlit dashboard (`app.py`)** for direct deployment.

### Deployment Settings
- **Main file:** `app.py`  
- **Python version (Advanced Settings):** `Python 3.12`  
- **Start command:**
```bash
streamlit run app.py
---

Required Files (Same Directory as app.py)
	•	batting_clean.csv
	•	bowling_clean.csv
	•	All .pkl model files
	•	bat_features.pkl
	•	bowl_features.pkl
	•	requirements.txt
	•	runtime.txt

Notes:
	•	Uses relative paths (BASE_DIR) → no path changes needed
	•	No database required at runtime
	•	No API keys needed
	•	Models are pre-trained → inference only

## Project Overview

This project analyzes how competitive pressure influences player
performance in the Indian Premier League (IPL) and builds regression
models to predict future performance under similar match conditions.

Competitive pressure is operationalized using tournament stage:

- **League** (baseline)
- **Playoffs**
- **Final**

The dataset spans IPL seasons from **2008–2024** and is derived
entirely from structured ball-by-ball match records provided by
Cricsheet.

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

This project addresses two connected objectives:

---

### 1️⃣ Descriptive Objective — Performance Under Pressure

Do IPL players exhibit higher performance risk and variability in
pressure matches compared to league matches?

The analysis evaluates stage-based shifts in:

#### Batting
- Average runs per innings
- Variance of runs
- Probability of low score (runs < 10)
- Stage-based Pressure Impact Index

#### Bowling
- Economy rate
- Variance of economy
- Probability of high-cost spell
- Wickets per match
- Stage-based Pressure Impact Index

Pressure matches are defined as **Playoffs + Finals**, with League
matches serving as the baseline.

---

### 2️⃣ Predictive Objective — Player-Level Regression

In addition to descriptive analysis, this project builds supervised
regression models to predict future player performance.

Two separate modeling pipelines are constructed:

#### Batting Regression Model
**Target:** Runs scored

**Features include:**
- Historical average runs
- Run variance
- Low-score probability
- Match stage
- Venue
- Season
- Date-derived features (month, weekday)

#### Bowling Regression Model
**Target:** Economy rate

**Features include:**
- Historical economy rate
- Wicket rate
- Dot-ball rate
- Match stage
- Venue
- Season
- Date-derived features

All features are computed using only historical matches prior to
prediction to avoid data leakage.

---

## Repository Structure
```
CAP5771-nvasanthakumar/
│
├── data_wrangling.ipynb
│     Data cleaning and feature engineering pipeline.
│     Loads raw match data from the SQLite database,
│     cleans data types and formats, handles missing
│     values and duplicates, engineers features, and
│     exports batting_clean.csv and bowling_clean.csv
│     ready for modeling.
│
├── data_modeling_data_visualization.ipynb
│     Machine learning models and interactive dashboard.
│     Loads batting_clean.csv and bowling_clean.csv,
│     trains five regression models for batting and
│     bowling, evaluates using MAE/RMSE/R², performs
│     cross-validation and error analysis, and presents
│     all results through a 6-tab interactive dashboard
│     built with ipywidgets.
│
├── batting_clean.csv
│     Cleaned and feature-engineered batting dataset.
│     One row per player per match. 16,515 rows.
│     Output of data_wrangling.ipynb.
│
├── bowling_clean.csv
│     Cleaned and feature-engineered bowling dataset.
│     One row per bowler per match. 12,978 rows.
│     Output of data_wrangling.ipynb.
│
├── ipl_cricket_only.db
│     SQLite relational database containing raw IPL
│     match data from Cricsheet (2008-2024).
│     Source database for data_wrangling.ipynb.
│
├── schema.png
│     Entity-relationship diagram showing the database
│     schema and table relationships.
│
├── data_dictionary.pdf
│     Definitions and descriptions of all variables
│     used across the project datasets.
│
├── requirements.txt
│     All Python dependencies with versions required
│     to reproduce the project outputs.
│
├── README.md
│     This file. Project overview, structure, and
│     reproduction instructions.
│
└── diary/
      Weekly diary entries documenting decisions,
      findings and reflections for each module.
      │
      ├── Module6_DataWranglingP1_Cleaning_Diary.txt
      ├── DataWranglingP2_FeatureEngineering_Diary.txt
      ├── DataModelingP1_Fundamentals_Diary.txt
      ├── DataModelingP2_Evaluation_Diary.txt
      └── DataVisualization_Diary.txt
```

---

## Data Source

All data is obtained from:

**Cricsheet**
https://cricsheet.org

The dataset includes IPL ball-by-ball match records from 2008–2024.

Processing steps:

1. Parsed YAML files using Python
2. Aggregated to player–match level
3. Stored in a structured SQLite relational database

---

## Database Design

A SQLite database (`ipl_cricket_only.db`) ensures structured
storage and reproducibility.

Core tables:

- `matches`
- `players`
- `player_match_batting`
- `player_match_bowling`

The variable `stage_bucket` (league/playoffs/final) serves as
the objective pressure proxy.

A schema diagram (`schema.png`) is included in the repository.

---

## Data Wrangling Pipeline

The project includes a reproducible two-stage wrangling process:

### Part I — Cleaning
- Data type correction (dates, numeric fields)
- Duplicate handling (player–match uniqueness)
- Structural missing value handling
- Standardization of categorical fields
- Validation of impossible values

### Part II — Feature Engineering
- Derived metrics (e.g., runs per ball)
- Pressure indicator variables
- Date feature extraction (match year, month, weekday)
- Season year extraction for time-based splitting
- Stage bucket classification (league / playoffs / final)
- Categorical encoding
- Feature selection for regression
- Correlation inspection

This produces two final model-ready datasets:
- `batting_clean.csv` — batting regression dataset
- `bowling_clean.csv` — bowling regression dataset

---

## How to Reproduce Milestone 2 Outputs

### Step 1 — Data Wrangling

1. Open `data_wrangling.ipynb` in Google Colab
2. Upload the following files to `/content/`:
   - `ipl_cricket_only.db`
   - `matches.csv`
   - `players.csv`
   - `player_match_batting.csv`
   - `player_match_bowling.csv`
3. Click **Runtime → Run All**
4. The notebook will produce:
   - `batting_clean.csv`
   - `bowling_clean.csv`

### Step 2 — Data Modeling and Visualization

1. Open `data_modeling_data_visualization.ipynb` in Google Colab
2. Upload these two files to `/content/`:
   - `batting_clean.csv`
   - `bowling_clean.csv`
3. Click **Runtime → Run All**
4. All model results, charts and the dashboard will generate
   automatically

No external APIs or credentials are required.

---

## How to Run the Dashboard

1. Open `data_modeling_data_visualization.ipynb` in Google Colab
2. Upload `batting_clean.csv` and `bowling_clean.csv` to `/content/`
3. Click **Runtime → Run All**
4. Scroll to the bottom of the notebook
5. The interactive dashboard will appear automatically

The dashboard has six tabs:

| Tab | What it shows |
|-----|---------------|
| 📊 Model Performance | R² comparison across all five models for batting and bowling |
| 🔍 Feature Importance | Top 10 features driving batting and bowling predictions |
| 🎯 Actual vs Predicted | Scatter plots showing prediction accuracy for both models |
| 📈 Stage Analysis | Average performance by match stage (league vs playoffs vs final) |
| 🏆 Player Rankings | Top batters and bowlers filtered by venue, season and match type |
| 🏏 Player Prediction | Predict any player at any venue, stage and season |

Use the toggle buttons at the top to switch between tabs.
In the Player Rankings tab use the dropdowns to filter by venue,
season and match type. In the Player Prediction tab type any
player name and select a venue, stage and season to generate
a prediction with a full breakdown of base prediction plus
venue adjustment plus stage adjustment.

---

## Models Used

| Model | Task | Test R² |
|-------|------|---------|
| Linear Regression | Batting (baseline) | 0.8931 |
| Ridge Regression | Batting | 0.9187 |
| Lasso Regression | Batting | 0.9160 |
| Elastic Net | Batting | 0.9070 |
| Random Forest | Batting (best) | 0.9506 |
| Linear Regression | Bowling (baseline) | 0.7547 |
| Ridge Regression | Bowling (best) | 0.8611 |
| Lasso Regression | Bowling | 0.7874 |
| Elastic Net | Bowling | 0.7779 |
| Random Forest | Bowling | 0.7817 |

**Train / Validation / Test Split (time-based):**
- Train      : seasons 2008–2021
- Validation : season 2022
- Test       : seasons 2023–2024

---

## Key Results

**Batting:** Random Forest achieves the best test R² of 0.9561,
confirming that nonlinear relationships between player identity,
venue and match stage drive batting performance.

**Bowling:** Ridge Regression generalises best with test R² of
0.8611, suggesting bowling economy benefits from regularisation
to handle noise and variability in bowling data.

**Prediction System:** The final prediction functions use Random
Forest for batting and Ridge Regression for bowling, with venue
and stage context adjustments. Predictions respond meaningfully
to different contexts — for example Shubman Gill is predicted
to score 35.65 runs in playoffs vs 30.14 in league matches,
reflecting his genuine big-match reputation. JJ Bumrah is
predicted to bowl most economically in finals (7.06 economy)
vs league matches (7.56 economy), consistent with his
real-world reputation as a pressure bowler.

---

## Technical Stack

- Python 3.10+
- pandas
- NumPy
- matplotlib
- seaborn
- SQLite
- scikit-learn
- ipywidgets

---

## Reproducibility

The repository includes:

- Full wrangling notebook (`data_wrangling.ipynb`)
- Full modeling and dashboard notebook
  (`data_modeling_data_visualization.ipynb`)
- SQLite database (`ipl_cricket_only.db`)
- Cleaned datasets (`batting_clean.csv`, `bowling_clean.csv`)
- Schema diagram (`schema.png`)
- Data dictionary (`data_dictionary.pdf`)
- `requirements.txt`
- Weekly diary entries in `diary/`

All analysis is reproducible directly from repository contents
by uploading the CSV files to Colab and pressing Run All.

---

## Summary

This project demonstrates:

- Clear and measurable problem formulation
- Structured relational database design
- Transparent data cleaning and feature engineering pipeline
- Pressure-aware performance analysis across match stages
- Five-model regression comparison with proper evaluation
- Cross-validation and residual error analysis
- Context-aware prediction system with venue and stage adjustments
- Interactive six-tab dashboard for result communication
- Honest documentation of data leakage and model limitations

It provides a reproducible foundation for analyzing and predicting
player performance under competitive pressure in professional cricket.
