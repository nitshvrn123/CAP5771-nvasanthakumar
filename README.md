# IPL Pressure & Performance Risk Analysis

### CAP5771 — Introduction to Data Science  
University of Florida  

---

## Project Overview

This project studies how competitive pressure affects player performance in the Indian Premier League (IPL).

Rather than focusing only on average performance, I analyze **performance risk and variability** under high-stakes match conditions.

Competitive pressure is operationalized using tournament stage:

- **League matches** (baseline)
- **Playoffs**
- **Finals**

The primary analytical comparison is:

> League vs Pressure Matches (Playoffs + Finals)

The dataset covers IPL seasons from **2008 through 2024**, using structured ball-by-ball match data.

This project is fully reproducible and uses only cricket-derived data (no psychological inference).

---

## Research Question

Do IPL players exhibit higher performance risk and variability in pressure matches compared to league matches?

The analysis is conducted separately for:

### Batting
- Average runs per innings  
- Probability of low score (runs < 10)  
- Variance of runs  
- Stage-based Pressure Impact Index  

### Bowling
- Average economy rate  
- Probability of high-cost spell  
- Variance of economy  
- Wickets per match  
- Stage-based Pressure Impact Index  

---

## Data Source

All data is derived from:

**Cricsheet**  
https://cricsheet.org

The dataset includes structured ball-by-ball IPL records from 2008–2024.

Data processing steps:

1. Parsed YAML files using Python  
2. Aggregated to player–match level  
3. Stored in a structured SQLite relational database  

---

## Database Design

A SQLite database (`ipl_cricket_only.db`) is used to ensure reproducibility and structured querying.

Core tables:

- `matches` — Match-level metadata and stage classification  
- `players` — Standardized player identifiers  
- `player_match_batting` — Batting performance per match  
- `player_match_bowling` — Bowling performance per match  

The variable `stage_bucket` (league/playoffs/final) serves as the project’s objective pressure proxy.

A schema diagram (`schema.png`) is included in the repository.

---

## Data Exploration

Exploratory analysis was conducted using:

- `.head()`
- `.info()`
- `.describe()`
- Duplicate checks
- Missing value analysis
- Stage distribution inspection
- Simple visualizations

Key observations:

- League matches dominate the dataset  
- Finals represent a small percentage of total observations  
- Missing bowling strike rates are structural (no wicket cases)  
- Extreme performances are retained as meaningful signals  

No corrections were performed during exploration; issues were documented.

---

## Pressure Impact Framework

To quantify stage-based shifts in performance, I constructed a **Pressure Impact Index**.

For batting:

Impact combines:
- Increase in low-score probability  
- Decrease in average runs relative to league baseline  

For bowling:

Impact combines:
- Increase in economy  
- Increase in high-cost spell probability  
- Decrease in wicket-taking effectiveness  

Players are categorized as:

- Stable  
- More risky under pressure  
- High pressure drop  

---

## Machine Learning Extension (Planned)

This project prepares the dataset for predictive modeling.

### Batting Model
Goal:
Predict expected runs scored in a future match under similar conditions.

Features:
- Historical average runs  
- Variance of runs  
- Low-score rate  
- Match stage  
- Season  
- Venue  

Target:
- Runs scored  

---

### Bowling Model
Goal:
Predict expected economy rate under similar match conditions.

Features:
- Historical economy  
- Wicket rate  
- Dot-ball rate  
- Match stage  
- Season  
- Venue  

Target:
- Economy rate  

All features will be computed using only historical matches prior to prediction to avoid data leakage.

---

## Technical Stack

- Python 3.10+
- pandas
- NumPy
- matplotlib
- seaborn
- SQLite
- scikit-learn (for future ML models)

---

## How to Run This Project (Google Colab)

This project was developed and executed in Google Colab.

### Steps:

1. Open the notebook in Google Colab.

2. Upload the Files : matches.csv
players.csv
player_match_batting.csv
player_match_bowling.csv
mainly : ipl_cricket_only.db

## Running in Google Colab

running this notebook in Google Colab:

1. Open the notebook in Colab.
2. Click the folder icon on the left sidebar.
3. Click “Upload”.
4. Upload the following files into the root directory:

- ipl_cricket_only.db
- matches.csv
- players.csv
- player_match_batting.csv
- player_match_bowling.csv

Make sure the files appear directly under `/content/`.

The notebook automatically looks for:

/content/ipl_cricket_only.db

3. Click: Runtime → Run All

The notebook will:

- Load the database  
- Perform exploratory analysis  
- Compute batting and bowling pressure metrics  
- Generate visualizations  
- Construct Pressure Impact Indices  

No external APIs or credentials are required.

---

## Future Work

Future milestones will extend this work by:

- Building regression models for run prediction  
- Building regression models for economy prediction  
- Evaluating model performance across match stages  
- Investigating interaction effects between venue and pressure  
- Introducing uncertainty estimation for performance predictions  

---

## Reproducibility

The repository includes:

- Full analysis notebook  
- SQLite database (or regeneration instructions)  
- Schema diagram (`schema.png`)  
- Data dictionary (`data_dictionary.pdf`)  
- requirements.txt  

All analysis is reproducible using only the repository contents.

---

## Summary

This project demonstrates:

- Clear problem formulation  
- Structured data acquisition  
- Relational database design  
- Thoughtful exploratory analysis  
- A modeling-ready analytical framework  

It provides a reproducible foundation for studying performance risk under competitive pressure in professional cricket.
