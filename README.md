# IPL Player Performance Under Pressure: Risk Analysis & Regression Modeling

### CAP5771 — Introduction to Data Science
University of Florida

---

## Project Overview

This project analyzes how competitive pressure influences player performance in the Indian Premier League (IPL) and builds regression models to predict future performance under similar match conditions.

Competitive pressure is operationalized using tournament stage:

- **League** (baseline)
- **Playoffs**
- **Final**

The dataset spans IPL seasons from **2008–2024** and is derived entirely from structured ball-by-ball match records provided by Cricsheet.

This project integrates:

- Pressure-based performance analysis
- Risk and variability measurement
- Structured database design
- Data wrangling and feature engineering
- Player-level regression modeling

All analysis is fully reproducible using cricket-derived data only.

---

## Research Objectives

This project addresses two connected objectives:

---

### 1️⃣ Descriptive Objective — Performance Under Pressure

Do IPL players exhibit higher performance risk and variability in pressure matches compared to league matches?

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

Pressure matches are defined as **Playoffs + Finals**, with League matches serving as the baseline.

---

### 2️⃣ Predictive Objective — Player-Level Regression

In addition to descriptive analysis, this project builds supervised regression models to predict future player performance.

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

---

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

All features are computed using only historical matches prior to prediction to avoid data leakage.

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

A SQLite database (`ipl_cricket_only.db`) ensures structured storage and reproducibility.

Core tables:

- `matches`
- `players`
- `player_match_batting`
- `player_match_bowling`

The variable `stage_bucket` (league/playoffs/final) serves as the objective pressure proxy.

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
- Date feature extraction (month, weekday)
- Categorical encoding
- Feature selection for regression
- Correlation inspection

This produces two final model-ready datasets:
- Batting regression dataset
- Bowling regression dataset

---

## Technical Stack

- Python 3.10+
- pandas
- NumPy
- matplotlib
- seaborn
- SQLite
- scikit-learn

---

## How to Run (Google Colab)

1. Open the notebook in Google Colab.
2. Upload the following files to `/content/`:

- `ipl_cricket_only.db`
- `matches.csv`
- `players.csv`
- `player_match_batting.csv`
- `player_match_bowling.csv`

3. Click: **Runtime → Run All**

The notebook will:
- Load the database
- Perform exploratory analysis
- Compute pressure metrics
- Build feature-engineered datasets
- Prepare regression-ready feature tables

No external APIs or credentials are required.

---

## Reproducibility

The repository includes:

- Full analysis notebook
- SQLite database (or regeneration instructions)
- Schema diagram (`schema.png`)
- Data dictionary (`data_dictionary.pdf`)
- `requirements.txt`

All analysis is reproducible directly from repository contents.

---

## Summary

This project demonstrates:

- Clear and measurable problem formulation
- Structured relational database design
- Transparent data cleaning and feature engineering
- Pressure-aware performance analysis
- Role-specific regression modeling framework

It provides a reproducible foundation for analyzing and predicting player performance under competitive pressure in professional cricket.