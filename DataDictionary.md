# Data Dictionary
## Quantifying Performance Risk in Professional Cricket Across Match Stages

### Data Source
All data is derived from structured ball-by-ball match data provided by Cricsheet (https://cricsheet.org),
covering Indian Premier League matches from 2008–2024.

---

## Table: matches
| Column | Type | Description |
|------|------|-------------|
| id | INTEGER | Unique match identifier |
| date | DATE | Match date |
| season | INTEGER | IPL season |
| venue | TEXT | Match venue |
| team1 | TEXT | Team batting first |
| team2 | TEXT | Team batting second |
| match_type | TEXT | Match format |
| stage_bucket | TEXT | league / playoffs / final |
| winner | TEXT | Winning team |
| result | TEXT | Match result type |

---

## Table: players
| Column | Type | Description |
|------|------|-------------|
| player_name | TEXT | Standardized player name |

---

## Table: player_match_batting
| Column | Type | Description |
|------|------|-------------|
| match_id | INTEGER | Foreign key → matches.id |
| player | TEXT | Batter name |
| runs | INTEGER | Runs scored |
| balls | INTEGER | Balls faced |
| strike_rate | REAL | (runs / balls) × 100 |
| low_score | INTEGER | 1 if runs < 10 |
| stage_bucket | TEXT | Match pressure stage |

---

## Table: player_match_bowling
| Column | Type | Description |
|------|------|-------------|
| match_id | INTEGER | Foreign key → matches.id |
| bowler | TEXT | Bowler name |
| overs | REAL | Overs bowled |
| runs_conceded | INTEGER | Runs conceded |
| wickets | INTEGER | Wickets taken |
| economy | REAL | Runs per over |
| dot_ball_rate | REAL | Dot balls / total balls |
| strike_rate_balls_per_wicket | REAL | Balls per wicket |
| stage_bucket | TEXT | Match pressure stage |

---

## Derived Metrics (Not Stored)
- pressure_impact_index (batting & bowling)
- performance variance
- low-score rate
- high-cost spell indicator

Derived dynamically during analysis using SQL and pandas.

---

## Notes
This data dictionary supports reproducible, cricket-only analysis of performance variability
across match stages. No external or subjective data sources are used.
