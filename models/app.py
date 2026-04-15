import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IPL Cricket Predictor",
    page_icon="🏏",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    batting = pd.read_csv(BASE_DIR / "batting_clean.csv")
    bowling = pd.read_csv(BASE_DIR / "bowling_clean.csv")

    if "date" in batting.columns:
        batting["date"] = pd.to_datetime(batting["date"], errors="coerce")
        batting["match_year"] = batting["date"].dt.year
        batting["match_month"] = batting["date"].dt.month
        batting["match_weekday"] = batting["date"].dt.weekday
        batting = batting.drop(columns=["date"], errors="ignore")

    if "date" in bowling.columns:
        bowling["date"] = pd.to_datetime(bowling["date"], errors="coerce")
        bowling["match_year"] = bowling["date"].dt.year
        bowling["match_month"] = bowling["date"].dt.month
        bowling["match_weekday"] = bowling["date"].dt.weekday
        bowling = bowling.drop(columns=["date"], errors="ignore")

    if "season_year" not in batting.columns:
        batting["season_year"] = batting["season"].astype(str).str[:4].astype(int)

    if "season_year" not in bowling.columns:
        bowling["season_year"] = bowling["season"].astype(str).str[:4].astype(int)

    return batting, bowling


# ══════════════════════════════════════════════════════════════════════════
# LOAD MODELS + FEATURES
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "bat_model.pkl", "rb") as f:
        bat_linear = pickle.load(f)
    with open(BASE_DIR / "ridge_bat.pkl", "rb") as f:
        bat_ridge = pickle.load(f)
    with open(BASE_DIR / "lasso_bat.pkl", "rb") as f:
        bat_lasso = pickle.load(f)
    with open(BASE_DIR / "elastic_bat.pkl", "rb") as f:
        bat_elastic = pickle.load(f)
    with open(BASE_DIR / "rf_bat.pkl", "rb") as f:
        bat_rf = pickle.load(f)

    with open(BASE_DIR / "bowl_model.pkl", "rb") as f:
        bowl_linear = pickle.load(f)
    with open(BASE_DIR / "ridge_bowl.pkl", "rb") as f:
        bowl_ridge = pickle.load(f)
    with open(BASE_DIR / "lasso_bowl.pkl", "rb") as f:
        bowl_lasso = pickle.load(f)
    with open(BASE_DIR / "elastic_bowl.pkl", "rb") as f:
        bowl_elastic = pickle.load(f)
    with open(BASE_DIR / "rf_bowl.pkl", "rb") as f:
        bowl_rf = pickle.load(f)

    with open(BASE_DIR / "bat_features.pkl", "rb") as f:
        bat_features = pickle.load(f)
    with open(BASE_DIR / "bowl_features.pkl", "rb") as f:
        bowl_features = pickle.load(f)

    return (
        bat_linear, bat_ridge, bat_lasso, bat_elastic, bat_rf,
        bowl_linear, bowl_ridge, bowl_lasso, bowl_elastic, bowl_rf,
        bat_features, bowl_features
    )


batting_clean, bowling_clean = load_data()

(
    bat_linear, bat_ridge, bat_lasso, bat_elastic, bat_rf,
    bowl_linear, bowl_ridge, bowl_lasso, bowl_elastic, bowl_rf,
    bat_features, bowl_features
) = load_artifacts()

batting_players = sorted(batting_clean["player"].dropna().unique().tolist())
bowling_players = sorted(bowling_clean["bowler"].dropna().unique().tolist())
venues = sorted(batting_clean["venue"].dropna().unique().tolist())
match_types = sorted(batting_clean["match_type"].dropna().unique().tolist())

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def align_features(df, feature_list):
    X = df.copy()
    for col in feature_list:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_list]
    return X


def make_test_sets():
    test_bat = batting_clean[batting_clean["season_year"] >= 2023].copy()
    test_bowl = bowling_clean[bowling_clean["season_year"] >= 2023].copy()

    y_test_bat = test_bat["runs"]
    y_test_bowl = test_bowl["economy"]

    X_test_bat = align_features(test_bat.drop(columns=["runs"], errors="ignore"), bat_features)
    X_test_bowl = align_features(test_bowl.drop(columns=["economy"], errors="ignore"), bowl_features)

    return X_test_bat, y_test_bat, X_test_bowl, y_test_bowl


def make_empty_row(feature_list):
    return pd.DataFrame([{col: np.nan for col in feature_list}])


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def predict_existing_batting(player_name, stage, season, venue, match_type):
    data = batting_clean[
        (batting_clean["player"] == player_name) &
        (batting_clean["season_year"] < int(season))
    ].copy()

    if data.empty:
        return None, f"No history found for {player_name} before {season}."

    row = make_empty_row(bat_features)

    for col in bat_features:
        if col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                row.loc[0, col] = data[col].mean()
            else:
                mode_val = data[col].mode()
                row.loc[0, col] = mode_val.iloc[0] if not mode_val.empty else np.nan

    row.loc[0, "player"] = player_name
    row.loc[0, "stage_bucket"] = stage
    row.loc[0, "season"] = str(season)
    row.loc[0, "match_year"] = int(season)
    row.loc[0, "season_year"] = int(season)
    row.loc[0, "match_id"] = 0
    row.loc[0, "venue"] = venue
    row.loc[0, "match_type"] = match_type

    overall_avg = data["runs"].mean()
    venue_data = data[data["venue"] == venue]
    stage_data = data[data["stage_bucket"] == stage]

    venue_diff = (venue_data["runs"].mean() - overall_avg) if len(venue_data) >= 3 else 0
    venue_weight = min(len(venue_data) / 20, 0.4) if len(venue_data) >= 3 else 0
    stage_diff = (stage_data["runs"].mean() - overall_avg) if len(stage_data) >= 3 else 0
    stage_weight = min(len(stage_data) / 20, 0.3) if len(stage_data) >= 3 else 0

    base = bat_rf.predict(row)[0]
    venue_adjustment = venue_diff * venue_weight
    stage_adjustment = stage_diff * stage_weight
    final = max(0, base + venue_adjustment + stage_adjustment)

    return {
        "base": round(base, 2),
        "venue_adjustment": round(venue_adjustment, 2),
        "stage_adjustment": round(stage_adjustment, 2),
        "final": round(final, 2),
        "matches": len(data),
        "overall_avg": round(overall_avg, 2),
    }, None


def predict_existing_bowling(player_name, stage, season, venue, match_type):
    data = bowling_clean[
        (bowling_clean["bowler"] == player_name) &
        (bowling_clean["season_year"] < int(season))
    ].copy()

    if data.empty:
        return None, f"No history found for {player_name} before {season}."

    row = make_empty_row(bowl_features)

    for col in bowl_features:
        if col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                row.loc[0, col] = data[col].mean()
            else:
                mode_val = data[col].mode()
                row.loc[0, col] = mode_val.iloc[0] if not mode_val.empty else np.nan

    row.loc[0, "bowler"] = player_name
    row.loc[0, "stage_bucket"] = stage
    row.loc[0, "season"] = str(season)
    row.loc[0, "match_year"] = int(season)
    row.loc[0, "season_year"] = int(season)
    row.loc[0, "match_id"] = 0
    row.loc[0, "venue"] = venue
    row.loc[0, "match_type"] = match_type

    overall_avg = data["economy"].mean()
    venue_data = data[data["venue"] == venue]
    stage_data = data[data["stage_bucket"] == stage]

    venue_diff = (venue_data["economy"].mean() - overall_avg) if len(venue_data) >= 3 else 0
    venue_weight = min(len(venue_data) / 20, 0.4) if len(venue_data) >= 3 else 0
    stage_diff = (stage_data["economy"].mean() - overall_avg) if len(stage_data) >= 3 else 0
    stage_weight = min(len(stage_data) / 20, 0.3) if len(stage_data) >= 3 else 0

    base = bowl_ridge.predict(row)[0]
    venue_adjustment = venue_diff * venue_weight
    stage_adjustment = stage_diff * stage_weight
    final = max(0, base + venue_adjustment + stage_adjustment)

    return {
        "base": round(base, 2),
        "venue_adjustment": round(venue_adjustment, 2),
        "stage_adjustment": round(stage_adjustment, 2),
        "final": round(final, 2),
        "matches": len(data),
        "overall_avg": round(overall_avg, 2),
    }, None


def predict_new_batting(name, balls, strike_rate, low_score, stage, season, venue, match_type):
    date = pd.Timestamp(f"{season}-04-01")
    row = make_empty_row(bat_features)

    defaults = {
        "match_id": 0,
        "player": name if name else "New Player",
        "balls": balls,
        "strike_rate": strike_rate,
        "low_score": low_score,
        "season": str(season),
        "venue": venue,
        "match_type": match_type,
        "stage_bucket": stage,
        "match_year": int(season),
        "match_month": date.month,
        "match_weekday": date.weekday(),
        "season_year": int(season),
    }

    for col in bat_features:
        row.loc[0, col] = defaults.get(col, np.nan)

    return round(max(0, bat_rf.predict(row)[0]), 2)


def predict_new_bowling(name, balls, runs_conceded, wickets, dot_balls, wides, no_balls,
                        stage, season, venue, match_type):
    date = pd.Timestamp(f"{season}-04-01")
    overs = balls / 6.0
    dot_ball_rate = dot_balls / balls if balls > 0 else 0
    sr_bpw = balls / wickets if wickets > 0 else np.nan

    row = make_empty_row(bowl_features)

    defaults = {
        "match_id": 0,
        "bowler": name if name else "New Bowler",
        "balls": balls,
        "runs_conceded": runs_conceded,
        "wides": wides,
        "no_balls": no_balls,
        "wickets": wickets,
        "dot_balls": dot_balls,
        "overs": overs,
        "strike_rate_balls_per_wicket": sr_bpw,
        "dot_ball_rate": dot_ball_rate,
        "season": str(season),
        "venue": venue,
        "match_type": match_type,
        "stage_bucket": stage,
        "season_year": int(season),
        "match_year": int(season),
        "match_month": date.month,
        "match_weekday": date.weekday(),
    }

    for col in bowl_features:
        row.loc[0, col] = defaults.get(col, np.nan)

    return round(max(0, bowl_ridge.predict(row)[0]), 2)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#1E40AF,#1D4ED8);
            padding:24px;border-radius:12px;margin-bottom:24px">
    <h1 style="color:white;margin:0;font-size:26px">
        🏏 IPL Cricket Performance Prediction Dashboard
    </h1>
    <p style="color:#BFDBFE;margin:8px 0 0;font-size:13px">
        Machine learning models predicting batting runs and bowling economy rate.
        IPL match data 2008–2024 | CAP5771 University of Florida
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Performance",
    "🔍 Feature Importance",
    "🎯 Actual vs Predicted",
    "📈 Stage Analysis",
    "🏆 Player Rankings",
    "🏏 Player Prediction"
])

X_test_bat, y_test_bat, X_test_bowl, y_test_bowl = make_test_sets()

# ── Tab 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Model Performance Overview")
    st.caption("All five saved models evaluated on test set (seasons 2023–2024)")

    models_bat = {
        "Linear": bat_linear, "Ridge": bat_ridge, "Lasso": bat_lasso,
        "Elastic Net": bat_elastic, "Random Forest": bat_rf,
    }
    models_bowl = {
        "Linear": bowl_linear, "Ridge": bowl_ridge, "Lasso": bowl_lasso,
        "Elastic Net": bowl_elastic, "Random Forest": bowl_rf,
    }

    bat_scores = {k: round(r2_score(y_test_bat, v.predict(X_test_bat)), 4) for k, v in models_bat.items()}
    bowl_scores = {k: round(r2_score(y_test_bowl, v.predict(X_test_bowl)), 4) for k, v in models_bowl.items()}

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#3B82F6" if v == max(bat_scores.values()) else "#BFDBFE" for v in bat_scores.values()]
        bars = ax.bar(bat_scores.keys(), bat_scores.values(), color=colors, edgecolor="white", width=0.5)
        ax.set_title("Batting — Model Comparison (Test R²)", fontsize=12)
        ax.set_ylabel("R² Score")
        ax.set_ylim(max(0, min(bat_scores.values()) - 0.08), 1.0)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, bat_scores.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#10B981" if v == max(bowl_scores.values()) else "#A7F3D0" for v in bowl_scores.values()]
        bars = ax.bar(bowl_scores.keys(), bowl_scores.values(), color=colors, edgecolor="white", width=0.5)
        ax.set_title("Bowling — Model Comparison (Test R²)", fontsize=12)
        ax.set_ylabel("R² Score")
        ax.set_ylim(max(0, min(bowl_scores.values()) - 0.08), 1.0)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, bowl_scores.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    best_bat = max(bat_scores, key=bat_scores.get)
    best_bowl = max(bowl_scores, key=bowl_scores.get)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Batting Model", best_bat, f"R² = {bat_scores[best_bat]}")
    c2.metric("Best Bowling Model", best_bowl, f"R² = {bowl_scores[best_bowl]}")
    c3.metric("Batting Improvement", f"+{bat_scores[best_bat] - bat_scores['Linear']:.4f}", "vs Linear baseline")
    c4.metric("Bowling Improvement", f"+{bowl_scores[best_bowl] - bowl_scores['Linear']:.4f}", "vs Linear baseline")


# ── Tab 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Feature Importance Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Batting — Random Forest Feature Importance")
        importances = bat_rf.named_steps["model"].feature_importances_
        feat_names = bat_rf.named_steps["preprocess"].get_feature_names_out()
        fi_bat = pd.DataFrame({"feature": feat_names, "importance": importances})\
            .sort_values("importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#1D4ED8" if i == 0 else "#3B82F6" if i < 3 else "#BFDBFE" for i in range(len(fi_bat))]
        ax.barh(fi_bat["feature"], fi_bat["importance"], color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()
        ax.set_title("Top 10 Features — Batting", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Darker blue = stronger predictor")

    with col2:
        st.caption("Bowling — Ridge Regression Coefficients")
        coefs = bowl_ridge.named_steps["model"].coef_
        feat_names = bowl_ridge.named_steps["preprocess"].get_feature_names_out()
        fi_bowl = pd.DataFrame({
            "feature": feat_names, "coefficient": coefs, "abs_importance": np.abs(coefs)
        }).sort_values("abs_importance", ascending=False).head(10)

        colors = ["#10B981" if v >= 0 else "#EF4444" for v in fi_bowl["coefficient"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(fi_bowl["feature"], fi_bowl["coefficient"], color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Coefficient Value")
        ax.invert_yaxis()
        ax.set_title("Top 10 Features — Bowling", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🟢 Green = increases economy | 🔴 Red = decreases economy")


# ── Tab 3 ──────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Prediction Accuracy")
    bat_pred = bat_rf.predict(X_test_bat)
    bowl_pred = bowl_ridge.predict(X_test_bowl)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test_bat, bat_pred, alpha=0.4, s=20, color="#3B82F6", edgecolors="none")
        lims = [0, max(y_test_bat.max(), bat_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Runs")
        ax.set_ylabel("Predicted Runs")
        ax.set_title("Actual vs Predicted — Batting\n(Random Forest)", fontsize=11)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test_bowl, bowl_pred, alpha=0.4, s=20, color="#10B981", edgecolors="none")
        lims2 = [0, max(y_test_bowl.max(), bowl_pred.max())]
        ax.plot(lims2, lims2, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Economy")
        ax.set_ylabel("Predicted Economy")
        ax.set_title("Actual vs Predicted — Bowling\n(Ridge Regression)", fontsize=11)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    bat_mae = mean_absolute_error(y_test_bat, bat_pred)
    bowl_mae = mean_absolute_error(y_test_bowl, bowl_pred)
    bat_r2 = r2_score(y_test_bat, bat_pred)
    bowl_r2 = r2_score(y_test_bowl, bowl_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batting MAE", f"{bat_mae:.2f} runs")
    c2.metric("Batting R²", f"{bat_r2:.4f}")
    c3.metric("Bowling MAE", f"{bowl_mae:.2f} eco")
    c4.metric("Bowling R²", f"{bowl_r2:.4f}")


# ── Tab 4 ──────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Performance by Match Stage")

    stage_bat = batting_clean.groupby("stage_bucket")["runs"].mean().sort_values(ascending=False)
    stage_bowl = bowling_clean.groupby("stage_bucket")["economy"].mean().sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#1D4ED8", "#3B82F6", "#BFDBFE"][:len(stage_bat)]
        bars = ax.bar(stage_bat.index, stage_bat.values, color=colors, edgecolor="white", width=0.4)
        ax.set_title("Average Runs by Match Stage\n(Highest to Lowest)", fontsize=11)
        ax.set_ylabel("Average Runs")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, stage_bat.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#065F46", "#10B981", "#A7F3D0"][:len(stage_bowl)]
        bars = ax.bar(stage_bowl.index, stage_bowl.values, color=colors, edgecolor="white", width=0.4)
        ax.set_title("Average Economy by Match Stage\n(Highest to Lowest)", fontsize=11)
        ax.set_ylabel("Average Economy Rate")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, stage_bowl.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Tab 5 ──────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Player Rankings")
    st.caption("Filter by venue, season and match type then click Apply Filters.")

    col1, col2, col3 = st.columns(3)
    with col1:
        venue_sel = st.selectbox("Venue", ["All Venues"] + venues, key="rank_venue")
    with col2:
        season_sel = st.selectbox(
            "Season",
            ["All Seasons"] + [str(y) for y in sorted(batting_clean["season_year"].unique())],
            key="rank_season"
        )
    with col3:
        type_sel = st.selectbox("Match Type", ["All Types"] + match_types, key="rank_type")

    apply = st.button("Apply Filters", type="primary", key="rank_apply")
    st.divider()

    if "rank_results" not in st.session_state:
        st.session_state["rank_results"] = None

    if apply:
        def apply_filters_rank(df, player_col, value_col, ascending, venue_filter, season_filter, type_filter):
            data = df.copy()
            if venue_filter != "All Venues":
                data = data[data["venue"] == venue_filter]
            if season_filter != "All Seasons":
                data = data[data["season_year"] == int(season_filter)]
            if type_filter != "All Types":
                data = data[data["match_type"] == type_filter]

            result = (
                data.groupby(player_col)[value_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "avg", "count": "matches"})
                .query("matches >= 1")
                .sort_values("avg", ascending=ascending)
                .head(10)
                .reset_index()
            )
            if len(result) > 0:
                return result

            if type_filter != "All Types":
                data = df.copy()
                if venue_filter != "All Venues":
                    data = data[data["venue"] == venue_filter]
                if season_filter != "All Seasons":
                    data = data[data["season_year"] == int(season_filter)]
                result = (
                    data.groupby(player_col)[value_col]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "avg", "count": "matches"})
                    .query("matches >= 1")
                    .sort_values("avg", ascending=ascending)
                    .head(10)
                    .reset_index()
                )
                if len(result) > 0:
                    return result

            if season_filter != "All Seasons":
                data = df.copy()
                if venue_filter != "All Venues":
                    data = data[data["venue"] == venue_filter]
                result = (
                    data.groupby(player_col)[value_col]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "avg", "count": "matches"})
                    .query("matches >= 1")
                    .sort_values("avg", ascending=ascending)
                    .head(10)
                    .reset_index()
                )
                if len(result) > 0:
                    return result

            data = df.copy()
            return (
                data.groupby(player_col)[value_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "avg", "count": "matches"})
                .query("matches >= 1")
                .sort_values("avg", ascending=ascending)
                .head(10)
                .reset_index()
            )

        top_bat = apply_filters_rank(batting_clean, "player", "runs", False, venue_sel, season_sel, type_sel)
        top_bowl = apply_filters_rank(bowling_clean, "bowler", "economy", True, venue_sel, season_sel, type_sel)

        st.session_state["rank_results"] = {
            "top_bat": top_bat, "top_bowl": top_bowl,
            "venue_sel": venue_sel, "season_sel": season_sel,
        }

    if st.session_state["rank_results"] is not None:
        r = st.session_state["rank_results"]
        top_bat = r["top_bat"]
        top_bowl = r["top_bowl"]
        venue_sel = r["venue_sel"]
        season_sel = r["season_sel"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏏 Top Batters — Highest to Lowest**")
            if len(top_bat) > 0:
                fig, ax = plt.subplots(figsize=(7, 6))
                colors_bat = [
                    "#1D4ED8" if i == 0 else "#3B82F6" if i == 1 else "#60A5FA" if i == 2 else "#BFDBFE"
                    for i in range(len(top_bat))
                ]
                bars1 = ax.barh(top_bat["player"], top_bat["avg"], color=colors_bat, edgecolor="white", height=0.6)
                ax.set_title(f"Top Batters — Highest to Lowest\nSeason: {season_sel}  |  Venue: {venue_sel}",
                             fontsize=11, pad=15)
                ax.set_xlabel("Average Runs", fontsize=11)
                ax.tick_params(axis="y", labelsize=10)
                ax.invert_yaxis()
                for bar, val, m in zip(bars1, top_bat["avg"], top_bat["matches"]):
                    bar_width = bar.get_width()
                    if bar_width > 5:
                        ax.text(bar_width * 0.97, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f} ({m}m)", va="center", ha="right",
                                fontsize=8, fontweight="bold", color="white")
                    else:
                        ax.text(bar_width + 0.3, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f} ({m}m)", va="center", ha="left",
                                fontsize=8, fontweight="bold", color="#1F2937")
                for i in range(len(top_bat)):
                    ax.text(0.3, i, f"#{i+1}", va="center", fontsize=9, color="white", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(
                    f"🏏 #1 Batter: {top_bat.iloc[0]['player']} — "
                    f"{top_bat.iloc[0]['avg']:.1f} avg runs ({top_bat.iloc[0]['matches']} matches)"
                )
            else:
                st.info("No batting data available for this filter combination.")

        with col2:
            st.markdown("**🎯 Top Bowlers — Lowest Economy**")
            if len(top_bowl) > 0:
                fig, ax = plt.subplots(figsize=(7, 6))
                colors_bowl = [
                    "#065F46" if i == 0 else "#10B981" if i == 1 else "#34D399" if i == 2 else "#A7F3D0"
                    for i in range(len(top_bowl))
                ]
                bars2 = ax.barh(top_bowl["bowler"], top_bowl["avg"], color=colors_bowl, edgecolor="white", height=0.6)
                ax.set_title(f"Top Bowlers — Lowest Economy\nSeason: {season_sel}  |  Venue: {venue_sel}",
                             fontsize=11, pad=15)
                ax.set_xlabel("Average Economy Rate", fontsize=11)
                ax.tick_params(axis="y", labelsize=9)
                ax.invert_yaxis()
                for bar, val, m in zip(bars2, top_bowl["avg"], top_bowl["matches"]):
                    bar_width = bar.get_width()
                    if bar_width > 1:
                        ax.text(bar_width * 0.97, bar.get_y() + bar.get_height()/2,
                                f"{val:.2f} ({m}m)", va="center", ha="right",
                                fontsize=8, fontweight="bold", color="white")
                    else:
                        ax.text(bar_width + 0.05, bar.get_y() + bar.get_height()/2,
                                f"{val:.2f} ({m}m)", va="center", ha="left",
                                fontsize=8, fontweight="bold", color="#1F2937")
                for i in range(len(top_bowl)):
                    ax.text(0.05, i, f"#{i+1}", va="center", fontsize=9, color="white", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(
                    f"🎯 #1 Bowler: {top_bowl.iloc[0]['bowler']} — "
                    f"{top_bowl.iloc[0]['avg']:.2f} avg economy ({top_bowl.iloc[0]['matches']} matches)"
                )
            else:
                st.info("No bowling data available for this filter combination.")
    else:
        st.info("Select your filters above and click Apply Filters to see rankings.")


# ── Tab 6 ──────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Player Performance Predictor")
    st.caption("Predict for existing players using history or enter stats for a new player.")

    col1, col2 = st.columns(2)
    with col1:
        player_type = st.radio("Player Type", ["Existing Player", "New Player"], horizontal=True)
    with col2:
        mode = st.radio("Mode", ["Batting", "Bowling"], horizontal=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stage = st.selectbox("Stage", ["league", "playoffs", "final"])
    with c2:
        season = st.selectbox("Season", [str(y) for y in range(2018, 2025)], index=6)
    with c3:
        venue = st.selectbox("Venue", venues)
    with c4:
        match_type = st.selectbox("Match Type", match_types)

    st.divider()

    if player_type == "Existing Player":
        if mode == "Batting":
            player_name = st.selectbox("Select Batter", batting_players)
        else:
            player_name = st.selectbox("Select Bowler", bowling_players)
    else:
        if mode == "Batting":
            st.markdown("**New Batter Details**")
            c1, c2, c3 = st.columns(3)
            with c1:
                new_name = st.text_input("Name", placeholder="Enter batter name")
            with c2:
                new_balls = st.slider("Balls", 1, 120, 20)
            with c3:
                new_sr = st.number_input("Strike Rate", 0.0, 600.0, 120.0)
            new_low = st.selectbox(
                "Low Score",
                [("No — scored 10+", 0), ("Yes — scored < 10", 1)],
                format_func=lambda x: x[0]
            )
        else:
            st.markdown("**New Bowler Details**")
            c1, c2, c3 = st.columns(3)
            with c1:
                new_name = st.text_input("Name", placeholder="Enter bowler name")
            with c2:
                new_balls_b = st.slider("Balls", 1, 60, 24)
            with c3:
                new_runs_c = st.number_input("Runs Conceded", 0, 200, 30)
            c4, c5, c6 = st.columns(3)
            with c4:
                new_wkts = st.number_input("Wickets", 0, 10, 1)
            with c5:
                new_dots = st.number_input("Dot Balls", 0, 60, 10)
            with c6:
                new_wides = st.number_input("Wides", 0, 30, 2)
            new_nb = st.number_input("No Balls", 0, 20, 0)

    st.divider()

    if st.button("🏏 Get Prediction", type="primary", use_container_width=True):
        if player_type == "Existing Player":
            if mode == "Batting":
                result, error = predict_existing_batting(player_name, stage, season, venue, match_type)
                if error:
                    st.error(error)
                else:
                    st.success(f"**Predicted Runs: {result['final']} runs**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Base Prediction", f"{result['base']} runs")
                    c2.metric("Venue Adjustment", f"{result['venue_adjustment']:+.2f} runs")
                    c3.metric("Stage Adjustment", f"{result['stage_adjustment']:+.2f} runs")
                    c4.metric("Career Average", f"{result['overall_avg']} runs")
                    st.caption(f"Based on {result['matches']} historical matches")
            else:
                result, error = predict_existing_bowling(player_name, stage, season, venue, match_type)
                if error:
                    st.error(error)
                else:
                    st.success(f"**Predicted Economy: {result['final']}**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Base Prediction", f"{result['base']}")
                    c2.metric("Venue Adjustment", f"{result['venue_adjustment']:+.2f}")
                    c3.metric("Stage Adjustment", f"{result['stage_adjustment']:+.2f}")
                    c4.metric("Career Average", f"{result['overall_avg']}")
                    st.caption(f"Based on {result['matches']} historical matches")
        else:
            if mode == "Batting":
                prediction = predict_new_batting(
                    new_name, new_balls, new_sr,
                    new_low[1] if isinstance(new_low, tuple) else 0,
                    stage, season, venue, match_type
                )
                st.success(f"**Predicted Runs: {prediction} runs**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Balls", new_balls)
                c2.metric("Strike Rate", new_sr)
                c3.metric("Venue", venue[:25])
                st.caption(f"Based on {len(batting_clean):,} historical IPL matches.")
            else:
                prediction = predict_new_bowling(
                    new_name, new_balls_b, new_runs_c,
                    new_wkts, new_dots, new_wides, new_nb,
                    stage, season, venue, match_type
                )
                st.success(f"**Predicted Economy: {prediction}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Balls", new_balls_b)
                c2.metric("Runs Conceded", new_runs_c)
                c3.metric("Wickets", new_wkts)
                st.caption(f"Based on {len(bowling_clean):,} historical IPL matches.")


# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "IPL Cricket Performance Prediction | CAP5771 Introduction to Data Science | "
    "University of Florida | Data: Cricsheet 2008–2024"
)
