"""
Run this script ONCE to train the model on the real IPL dataset.
Files should be at: data/archive/ball_by_ball_data.csv and data/archive/ipl_matches_data.csv
Then run:
    python3 generate_and_train.py
"""

# ── Step 1: Import all required libraries ─────────────────────────────────────
import pandas as pd                          # for reading and processing CSV files
import numpy as np                           # for mathematical calculations
from sklearn.ensemble import RandomForestRegressor  # our main ML algorithm
from sklearn.model_selection import train_test_split # to split data into train and test
from sklearn.preprocessing import LabelEncoder       # to convert text to numbers
from sklearn.metrics import r2_score, mean_absolute_error  # to evaluate model accuracy
import pickle                                # to save the trained model to a file
import json                                  # to save teams and venues list

# ── Step 2: Load the raw IPL dataset from CSV files ──────────────────────────
# ball_by_ball_data.csv → ball by ball data (278,000+ rows)
# ipl_matches_data.csv  → match level information (1,169 rows)
deliveries = pd.read_csv('data/archive/ball_by_ball_data.csv')
matches    = pd.read_csv('data/archive/ipl_matches_data.csv')

print("🏏 Training IPL Score Predictor on real data...")
print(f"   Deliveries: {len(deliveries):,} rows")
print(f"   Matches   : {len(matches):,} rows")

# ── Step 3: Fix column names — dataset uses different names ──────────────────
# over_number is 0-indexed (0 to 19) → convert to 1-indexed (1 to 20)
# ball_number is the ball within the over
deliveries['over'] = deliveries['over_number'] + 1
deliveries['ball'] = deliveries['ball_number']

# Team IDs are numbers — map them to actual IPL team names
TEAM_MAP = {
    1   : "Royal Challengers Bengaluru",
    2   : "Sunrisers Hyderabad",
    3   : "Kolkata Knight Riders",
    4   : "Pune Warriors India",
    5   : "Mumbai Indians",
    6   : "Chennai Super Kings",
    129 : "Delhi Capitals",
    134 : "Rajasthan Royals",
    252 : "Punjab Kings",
    494 : "Deccan Chargers",
    614 : "Gujarat Titans",
    615 : "Lucknow Super Giants",
    1414: "Gujarat Lions",
    1419: "Rising Pune Supergiant",
}
deliveries['batting_team'] = deliveries['team_batting'].map(TEAM_MAP).fillna("Unknown")
deliveries['bowling_team'] = deliveries['team_bowling'].map(TEAM_MAP).fillna("Unknown")

# is_wicket: convert to proper boolean (True/False)
deliveries['is_wicket'] = deliveries['is_wicket'].astype(str).str.lower() == 'true'

# Remove super overs — they are extra overs, not part of regular innings
deliveries = deliveries[deliveries['is_super_over'] == False].copy()

# ── Step 4: Merge datasets to get venue info into deliveries ──────────────────
# We only need match_id, venue and season from matches
match_info = matches[['match_id', 'venue', 'season']].copy()
df = deliveries.merge(match_info, on='match_id', how='left')

# ── Step 5: Filter first innings only ────────────────────────────────────────
# We only want to predict first innings score
# Second innings is a chase — different problem entirely
df = df[df['innings'] == 1].copy()

# Sort by match, over and ball to ensure correct order for cumulative calculations
df = df.sort_values(['match_id', 'over', 'ball'])

# ── Step 6: Feature Engineering — Cumulative Runs ────────────────────────────
# cum_runs = running total of runs scored ball by ball within each match
# Example: Ball1→1run, Ball2→5runs, Ball3→5runs ... Ball120→178runs
# This tells us the score at any point in the innings
df['cum_runs'] = df.groupby('match_id')['total_runs'].cumsum()

# ── Step 7: Feature Engineering — Cumulative Wickets ─────────────────────────
# cum_wickets = running total of wickets fallen ball by ball
# is_wicket is already boolean True/False → convert to int 1/0 then cumsum
df['cum_wickets'] = df.groupby('match_id')['is_wicket'].apply(
    lambda x: x.astype(int).cumsum()
).reset_index(level=0, drop=True)

# ── Step 8: Feature Engineering — Ball Number ────────────────────────────────
# Total balls bowled = over × 6 + ball in current over
# Example: Over 10, Ball 3 → (10 × 6) + 3 = 63 balls bowled
df['ball_num'] = df['over'] * 6 + df['ball']

# ── Step 9: Feature Engineering — Last 30 Balls Runs ─────────────────────────
# Captures recent scoring momentum — last 5 overs of scoring
# Helps model understand if team is accelerating or slowing down
def last_n_runs(group, n=30):
    runs   = group['total_runs'].values
    result = np.zeros(len(runs))
    for i in range(len(runs)):
        result[i] = runs[max(0, i-n):i].sum()
    return pd.Series(result, index=group.index)

df['last_30_balls_runs'] = df.groupby('match_id', group_keys=False).apply(last_n_runs)

# ── Step 10: Feature Engineering — Current Run Rate ──────────────────────────
# run_rate = runs scored ÷ overs bowled so far
# replace(0, np.nan) prevents divide by zero on first ball
df['run_rate'] = (df['cum_runs'] / (df['ball_num']/6).replace(0, np.nan)).fillna(0)

# ── Step 11: Feature Engineering — Projected Linear Score ────────────────────
# Simple projection assuming current run rate continues for all 20 overs
# Example: run_rate = 9.5 → projected = 9.5 × 20 = 190
df['projected_linear'] = df['run_rate'] * 20

# ── Step 12: Feature Engineering — Projected CRR Score ───────────────────────
# More realistic: current score + (run rate × overs remaining)
# Example: Score=95, RR=9.5, Overs left=10 → 95 + (9.5 × 10) = 190
df['projected_crr'] = df['cum_runs'] + df['run_rate'] * (20 - df['over'])

# ── Step 13: Create Target Column — Final Score ───────────────────────────────
# final_score = maximum cumulative runs in each match = actual final score
final_scores = df.groupby('match_id')['cum_runs'].max().reset_index()
final_scores.columns = ['match_id', 'final_score']
df = df.merge(final_scores, on='match_id', how='left')

# ── Step 14: Filter valid overs and remove missing values ────────────────────
# Keep only overs 1 to 19 (over 20 is last — prediction not useful)
df = df[(df['over'] >= 1) & (df['over'] <= 19)].dropna(
    subset=['batting_team', 'bowling_team', 'venue', 'cum_runs', 'cum_wickets',
            'run_rate', 'last_30_balls_runs', 'projected_linear',
            'projected_crr', 'final_score'])

print(f"   Training rows : {len(df):,}")

# ── Step 15: Label Encoding — Convert text categories to numbers ──────────────
# ML models only understand numbers — not text like "Mumbai Indians"
# LabelEncoder: Mumbai Indians→5, Chennai Super Kings→0, Wankhede→14 etc.
le_bat   = LabelEncoder()
le_bowl  = LabelEncoder()
le_venue = LabelEncoder()

df['bat_enc']   = le_bat.fit_transform(df['batting_team'])
df['bowl_enc']  = le_bowl.fit_transform(df['bowling_team'])
df['venue_enc'] = le_venue.fit_transform(df['venue'])

# ── Step 16: Define Features (X) and Target (y) ──────────────────────────────
features = ['bat_enc',            # which team is batting
            'bowl_enc',           # which team is bowling
            'venue_enc',          # which ground is being played at
            'over',               # current over number
            'cum_runs',           # runs scored so far
            'cum_wickets',        # wickets fallen so far
            'run_rate',           # current run rate
            'last_30_balls_runs', # runs in last 5 overs (momentum)
            'projected_linear',   # simple score projection
            'projected_crr']      # CRR based score projection

X = df[features]
y = df['final_score']

# ── Step 17: Split data into Training and Testing sets ───────────────────────
# 80% for training, 20% for testing
# random_state=42 ensures same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n   Training samples : {len(X_train):,}")
print(f"   Test samples     : {len(X_test):,}")

# ── Step 18: Train the Random Forest Model ────────────────────────────────────
# n_estimators=200   → builds 200 decision trees
# max_depth=20       → each tree can ask max 20 questions
# min_samples_leaf=3 → each leaf needs at least 3 data points
# n_jobs=-1          → uses all CPU cores for faster training
print("\n⏳ Training Random Forest... (this takes 2-3 minutes)")
model = RandomForestRegressor(n_estimators=200, max_depth=20,
                               min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Step 19: Evaluate Model on Test Data ─────────────────────────────────────
preds = model.predict(X_test)
rmse  = np.sqrt(np.mean((y_test - preds)**2))
print(f"\n✅ R² Score : {r2_score(y_test, preds):.4f}")
print(f"✅ MAE      : {mean_absolute_error(y_test, preds):.2f} runs")
print(f"✅ RMSE     : {rmse:.2f} runs")

# ── Step 20: Save trained model and encoders to files ────────────────────────
# model.pkl    → saves the trained Random Forest model
# encoders.pkl → saves the label encoders (needed to encode new input in app)
# meta.json    → saves list of all teams and venues (used in app dropdowns)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump({'bat': le_bat,
                 'bowl': le_bowl,
                 'venue': le_venue}, f)

with open('meta.json', 'w') as f:
    json.dump({
        'teams' : sorted(df['batting_team'].unique().tolist()),
        'venues': sorted(df['venue'].unique().tolist())
    }, f)

print("\n✅ Saved: model.pkl, encoders.pkl, meta.json")
print("🚀 Now run: streamlit run app.py")