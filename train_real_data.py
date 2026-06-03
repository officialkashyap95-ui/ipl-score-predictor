"""
IPL Score Predictor - Real Data Training Script
Compatible with: ball_by_ball_data.csv + ipl_matches_data.csv

Run: python3 train_real_data.py
Auto-generates ipl_model.pkl used by app.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DELIVERIES   = os.path.join(BASE_DIR, "data", "archive", "ball_by_ball_data.csv")
MATCHES      = os.path.join(BASE_DIR, "data", "archive", "ipl_matches_data.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "ipl_model.pkl")

# ── Team ID → Name mapping ────────────────────────────────────────────────────
TEAM_MAP = {
    1   : "Royal Challengers Bangalore",
    2   : "Sunrisers Hyderabad",
    3   : "Kolkata Knight Riders",
    4   : "Pune Warriors India",
    5   : "Mumbai Indians",
    6   : "Chennai Super Kings",
    129 : "Delhi Capitals",
    134 : "Rajasthan Royals",
    252 : "Kings XI Punjab",
    494 : "Deccan Chargers",
    614 : "Gujarat Titans",
    615 : "Lucknow Super Giants",
    1414: "Gujarat Lions",
    1419: "Rising Pune Supergiant",
}


# ── Step 1: Load & Clean Data ─────────────────────────────────────────────────
def load_data():
    print("📂 Loading data...")
    d = pd.read_csv(DELIVERIES)
    m = pd.read_csv(MATCHES)

    # Map numeric team IDs → team names
    d['batting_team']     = d['team_batting'].map(TEAM_MAP).fillna("Unknown")
    d['bowling_team']     = d['team_bowling'].map(TEAM_MAP).fillna("Unknown")
    m['toss_winner_name'] = m['toss_winner'].map(TEAM_MAP).fillna("Unknown")

    # Normalize: over_number is 0-indexed (0–19), convert to 1-indexed (1–20)
    d['over'] = d['over_number'] + 1
    d['ball'] = d['ball_number']

    # is_wicket → boolean
    d['is_wicket'] = d['is_wicket'].astype(str).str.lower() == 'true'

    # Keep 1st innings only, remove super overs
    d = d[(d['innings'] == 1) & (d['is_super_over'] == False)].copy()

    # Compute first_innings_score from ball-by-ball (not in matches file)
    fi_score = d.groupby('match_id')['total_runs'].sum().reset_index()
    fi_score.columns = ['match_id', 'first_innings_score']
    m = m.merge(fi_score, on='match_id', how='left')
    m = m.dropna(subset=['first_innings_score'])
    m['first_innings_score'] = m['first_innings_score'].astype(int)

    print(f"   Deliveries  : {len(d):,} rows  (1st innings only)")
    print(f"   Matches     : {len(m):,} rows")
    print(f"   Score range : {m['first_innings_score'].min()} – {m['first_innings_score'].max()}")
    print(f"   Seasons     : {sorted(m['season'].unique().tolist())}")
    print(f"   Teams       : {sorted(d['batting_team'].unique().tolist())}")
    return d, m


# ── Step 2: League Average Score at Each Over ─────────────────────────────────
def compute_avg_score_at_over(deliveries):
    print("\n📊 Computing league average score at each over...")
    avg_score_at_over = {}
    for over_num in range(1, 20):
        scores = []
        for match_id, grp in deliveries.groupby('match_id'):
            cum = grp[grp['over'] <= over_num]['total_runs'].sum()
            scores.append(cum)
        avg_score_at_over[over_num] = round(np.mean(scores), 2)
    print(f"   Over 6  avg : {avg_score_at_over[6]:.1f} runs")
    print(f"   Over 10 avg : {avg_score_at_over[10]:.1f} runs")
    print(f"   Over 15 avg : {avg_score_at_over[15]:.1f} runs")
    return avg_score_at_over


# ── Step 3: Feature Engineering ───────────────────────────────────────────────
def build_features(deliveries, matches, avg_score_at_over):
    print("\n⚙️  Engineering features from ball-by-ball data...")

    d = deliveries.sort_values(['match_id', 'over', 'ball'])
    rows = []

    for match_id, grp in d.groupby('match_id'):
        mi = matches[matches['match_id'] == match_id]
        if mi.empty:
            continue
        mi           = mi.iloc[0]
        final_score  = mi['first_innings_score']
        venue        = mi['venue']
        batting_team = grp.iloc[0]['batting_team']
        bowling_team = grp.iloc[0]['bowling_team']

        # toss_winner_name vs batting_team, toss_decision == 'bat'
        toss_batting = 1 if (
            mi['toss_winner_name'] == batting_team and
            mi['toss_decision'] == 'bat'
        ) else 0

        grp = grp.reset_index(drop=True)

        for over_num in range(1, 20):
            over_data = grp[grp['over'] == over_num]
            if over_data.empty:
                continue

            cum           = grp[grp['over'] <= over_num]
            score         = int(cum['total_runs'].sum())
            wickets       = int(cum['is_wicket'].sum())
            balls_rem     = (20 * 6) - len(cum)
            rr            = round(score / max(over_num, 0.1), 2)

            last2         = grp[(grp['over'] > over_num - 2) & (grp['over'] <= over_num)]
            last2_rr      = round(last2['total_runs'].sum() / max(len(last2) / 6, 0.1), 2)

            last5         = grp[(grp['over'] > over_num - 5) & (grp['over'] <= over_num)]
            last5_runs    = int(last5['total_runs'].sum())

            proj_score    = round(score + (balls_rem / 6) * rr, 1)

            # Soft wicket factor: 0 wkts=1.0 → 9 wkts=0.676
            wicket_factor = round(1.0 - (wickets / 10) * 0.36, 3)

            # Pace vs league average
            avg_at_over   = avg_score_at_over.get(over_num, 120)
            pace_vs_avg   = round(score / max(avg_at_over, 1), 3)

            rows.append({
                'batting_team'   : batting_team,
                'bowling_team'   : bowling_team,
                'venue'          : venue,
                'over'           : over_num,
                'balls_remaining': balls_rem,
                'wickets_fallen' : wickets,
                'wicket_factor'  : wicket_factor,
                'current_score'  : score,
                'run_rate'       : rr,
                'last2_run_rate' : last2_rr,
                'last5_runs'     : last5_runs,
                'proj_score'     : proj_score,
                'pace_vs_avg'    : pace_vs_avg,
                'toss_batting'   : toss_batting,
                'final_score'    : final_score,
            })

    df = pd.DataFrame(rows)
    print(f"   Training rows : {len(df):,}")
    print(f"   Score range   : {df['final_score'].min()} – {df['final_score'].max()}")
    print(f"   Nulls         : {df.isnull().sum().sum()}")
    return df


# ── Step 4: Encode Categoricals ───────────────────────────────────────────────
def encode(df):
    print("\n🔠 Encoding categorical features...")
    all_teams  = sorted(df['batting_team'].unique().tolist())
    all_venues = sorted(df['venue'].unique().tolist())

    le_team  = LabelEncoder().fit(all_teams)
    le_venue = LabelEncoder().fit(all_venues)

    df['batting_team_enc'] = le_team.transform(df['batting_team'])
    df['bowling_team_enc'] = le_team.transform(df['bowling_team'])
    df['venue_enc']        = le_venue.transform(df['venue'])

    print(f"   Teams  : {len(all_teams)} → {all_teams}")
    print(f"   Venues : {len(all_venues)}")
    return df, le_team, le_venue, all_teams, all_venues


# ── Step 5: Train & Compare Models ────────────────────────────────────────────
def train(df, le_team, le_venue, all_teams, all_venues, avg_score_at_over):
    features = [
        'batting_team_enc', 'bowling_team_enc', 'venue_enc',
        'over', 'balls_remaining',
        'wickets_fallen', 'wicket_factor',
        'current_score', 'run_rate', 'last2_run_rate',
        'last5_runs', 'proj_score', 'pace_vs_avg', 'toss_batting'
    ]

    X = df[features]
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(
                                n_estimators=300, random_state=42,
                                n_jobs=-1, min_samples_leaf=3),
        "Gradient Boosting" : GradientBoostingRegressor(
                                n_estimators=300, learning_rate=0.05,
                                max_depth=6, random_state=42),
    }

    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("─" * 55)

    best_model, best_r2, best_name = None, -999, ""
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        print(f"{name:<25} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, mdl

    print(f"\n🏆 Best Model : {best_name}  (R² = {best_r2:.4f})")

    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_,
                       index=features).sort_values(ascending=False)
        print("\n📌 Feature Importances:")
        for feat, imp in fi.items():
            bar = "█" * int(imp * 40)
            print(f"   {feat:<22} {bar} {imp:.3f}")

    preds_test = best_model.predict(X_test)
    print(f"\n✅ Calibration — avg predicted: {preds_test.mean():.1f}  avg actual: {y_test.mean():.1f}")

    artifacts = {
        "model"            : best_model,
        "le_team"          : le_team,
        "le_venue"         : le_venue,
        "features"         : features,
        "teams"            : all_teams,
        "venues"           : all_venues,
        "avg_score_at_over": avg_score_at_over,
    }
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n💾 Model saved → {MODEL_OUTPUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deliveries, matches       = load_data()
    avg_score_at_over         = compute_avg_score_at_over(deliveries)
    df                        = build_features(deliveries, matches, avg_score_at_over)
    df, le_team, le_venue, all_teams, all_venues = encode(df)
    train(df, le_team, le_venue, all_teams, all_venues, avg_score_at_over)
    print("\n✅ Done! Now run:  streamlit run app.py")