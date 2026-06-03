import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f0c29); }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #16213e 100%);
    border-right: 2px solid #f59e0b;
  }
  .metric-card {
    background: linear-gradient(135deg, #1e3a5f, #0f2027);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px 0;
  }
  .metric-card h2 { color: #f59e0b; font-size: 2.2rem; margin: 0; }
  .metric-card p  { color: #cbd5e1; font-size: 0.9rem; margin: 4px 0 0; }
  .header-banner {
    background: linear-gradient(90deg, #f59e0b, #ef4444, #f59e0b);
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 20px;
  }
  .header-banner h1 { color: #0f0c29; font-size: 2rem; margin: 0; font-weight: 900; }
  .header-banner p  { color: #1a1a3e; margin: 4px 0 0; font-size: 1rem; }
  .section-title {
    color: #f59e0b;
    font-size: 1.1rem;
    font-weight: 700;
    border-bottom: 1px solid #f59e0b44;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }
  .stButton > button {
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    color: #0f0c29 !important;
    font-weight: 900;
    font-size: 1.1rem;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    width: 100%;
    transition: transform 0.15s;
  }
  .stButton > button:hover { transform: scale(1.03); }
  label { color: #cbd5e1 !important; }
  .info-box {
    background: #1e3a5f44;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 12px 16px;
    color: #cbd5e1;
    font-size: 0.88rem;
    margin-top: 8px;
  }
  .range-box {
    background: #1a2a1a;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 12px 16px;
    color: #b0d0b0;
    font-size: 0.88rem;
    margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── File paths ────────────────────────────────────────────────────────────────
MATCHES_PATH = "data/archive/ipl_matches_data.csv"

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    with open("meta.json") as f:
        meta = json.load(f)
    return model, enc, meta

@st.cache_data
def load_matches():
    return pd.read_csv(MATCHES_PATH)

model, enc, meta = load_assets()
teams  = meta["teams"]
venues = meta["venues"]

# ── Dynamic Range Function ────────────────────────────────────────────────────
def get_dynamic_range(over, wickets, run_rate):
    """
    Calculate prediction range dynamically based on match situation.

    Key findings from real IPL data analysis:
    - More overs remaining = wider range (more uncertainty)
    - More wickets fallen  = much wider range (team in trouble = unpredictable)
    - Slow run rate        = wider range (could explode or collapse)

    Returns: (margin_low, margin_high)
    """

    # Base margin from over number — fewer overs left = more certain
    if over <= 6:
        base = 27        # powerplay — lots can change
    elif over <= 10:
        base = 24        # middle overs — still uncertain
    elif over <= 15:
        base = 18        # getting clearer
    else:
        base = 12        # death overs — most certain

    # Wicket factor — biggest driver of uncertainty in T20
    # Data: 0-2 wkts = ±26 error, 3-5 wkts = ±42, 6+ wkts = ±66
    if wickets <= 2:
        wicket_add = 0       # settled innings — predictable
    elif wickets <= 5:
        wicket_add = 12      # some pressure — wider range
    else:
        wicket_add = 28      # deep trouble — very unpredictable

    # Run rate factor — slow teams have much more variance
    # Data: slow RR(<7) = ±46 error, avg RR = ±22, fast RR = ±30
    if run_rate < 7:
        rr_add = 15          # slow start — could explode or collapse
    elif run_rate > 10:
        rr_add = 8           # flying — hard to predict peak
    else:
        rr_add = 0           # normal pace — no extra uncertainty

    margin = base + wicket_add + rr_add

    # Asymmetric range: more room on upside (teams accelerate in death overs)
    # than downside (hard to score even less than current pace)
    margin_low  = int(margin * 0.8)
    margin_high = int(margin * 1.2)

    return margin_low, margin_high


# ── Range label ───────────────────────────────────────────────────────────────
def get_range_label(over, wickets, run_rate):
    """Returns a human-readable explanation of why range is wide or narrow."""
    if wickets >= 6:
        return "⚠️ Wide range — many wickets down, outcome very unpredictable"
    elif wickets >= 3 and run_rate < 7:
        return "⚠️ Wide range — under pressure with slow scoring"
    elif over >= 16 and wickets <= 2:
        return "✅ Narrow range — late overs, set batting, high confidence"
    elif run_rate < 7:
        return "📊 Wider range — slow start increases uncertainty"
    elif over <= 6:
        return "📊 Wider range — early overs, match still open"
    else:
        return "📊 Normal range — typical mid-innings uncertainty"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🏏 IPL Score Predictor</h1>
  <p>Real IPL Data 2008–2025 · Random Forest · ML-Powered</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Match Setup")
    batting_team = st.selectbox(
        "🏏 Batting Team", teams,
        index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
    bowling_team_options = [t for t in teams if t != batting_team]
    bowling_team = st.selectbox(
        "⚡ Bowling Team", bowling_team_options,
        index=bowling_team_options.index("Chennai Super Kings")
        if "Chennai Super Kings" in bowling_team_options else 0)
    venue = st.selectbox("🏟️ Venue", venues)
    st.markdown("---")
    st.markdown("## 📊 Current Match Situation")
    current_over      = st.slider("🎯 Current Over",            1,  19,  10)
    current_score     = st.slider("🏆 Current Score (runs)",    10, 220,  85)
    wickets_fallen    = st.slider("❌ Wickets Fallen",           0,   9,   2)
    last_5_overs_runs = st.slider("🔥 Runs in Last 5 Overs",    0,  80,  35)
    st.markdown("---")
    predict_btn = st.button("🚀 PREDICT SCORE")

# ── Derived metrics ───────────────────────────────────────────────────────────
run_rate   = round(current_score / current_over, 2) if current_over > 0 else 0
overs_left = 20 - current_over

# ── Prediction Function ───────────────────────────────────────────────────────
def predict_score(bat, bowl, ven, over, score, wkts, rr, last30):
    try:
        bat_enc  = enc['bat'].transform([bat])[0]
        bowl_enc = enc['bowl'].transform([bowl])[0]
        ven_enc  = enc['venue'].transform([ven])[0]
    except ValueError:
        return None

    proj_linear = rr * 20
    proj_crr    = score + rr * (20 - over)
    features    = np.array([[bat_enc, bowl_enc, ven_enc, over, score,
                              wkts, rr, last30, proj_linear, proj_crr]])
    ml_pred = model.predict(features)[0]

    wickets_remaining = 10 - wkts
    wickets_in_hand_factor = {
        10:1.00, 9:0.99, 8:0.97, 7:0.95, 6:0.92,
        5:0.87,  4:0.80, 3:0.70, 2:0.55, 1:0.38, 0:0.00
    }
    wif = wickets_in_hand_factor[min(wickets_remaining, 10)]

    if over <= 6:
        phase_multiplier = 1.05
    elif over <= 15:
        phase_multiplier = 1.00
    else:
        phase_multiplier = 1.08

    expected_rr    = rr * wif * phase_multiplier
    crr_projection = score + (expected_rr * (20 - over))

    crr_weight = max(0.20, min(0.80, 1.0 - (over / 20)))
    ml_weight  = 1.0 - crr_weight
    blended    = (ml_weight * ml_pred) + (crr_weight * crr_projection)

    return max(int(round(blended)), score + 5)


# ── Top metric cards ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h2>{current_score}</h2><p>Current Score</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h2>{wickets_fallen}</h2><p>Wickets Down</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h2>{run_rate}</h2><p>Current Run Rate</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h2>{overs_left}</h2><p>Overs Remaining</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Prediction result ─────────────────────────────────────────────────────────
if predict_btn:
    predicted = predict_score(batting_team, bowling_team, venue,
                               current_over, current_score, wickets_fallen,
                               run_rate, last_5_overs_runs)
    if predicted is None:
        st.error("⚠️ Team or venue not found in training data. Try a different selection.")
    else:
        runs_needed = max(0, predicted - current_score)
        req_rr      = round(runs_needed / overs_left, 2) if overs_left > 0 else 0

        # ── DYNAMIC RANGE based on match situation ────────────────────────────
        margin_low, margin_high = get_dynamic_range(current_over, wickets_fallen, run_rate)
        lower       = predicted - margin_low
        upper       = predicted + margin_high
        range_label = get_range_label(current_over, wickets_fallen, run_rate)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e3a5f,#0f2027);border:2px solid #f59e0b;
                    border-radius:16px;padding:30px;text-align:center;margin:16px 0;">
          <p style="color:#94a3b8;font-size:1rem;margin:0;">🤖 Predicted Final Score</p>
          <h1 style="color:#f59e0b;font-size:4rem;margin:8px 0;font-weight:900;">{predicted}</h1>
          <p style="color:#cbd5e1;font-size:1rem;">Likely range: <b style='color:#f59e0b'>{lower} – {upper}</b></p>
          <p style="color:#94a3b8;font-size:0.9rem;margin-top:8px;">{batting_team} vs {bowling_team} · {venue}</p>
        </div>
        """, unsafe_allow_html=True)

        # Range explanation
        st.markdown(f'<div class="range-box">{range_label}</div>', unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="metric-card"><h2>{runs_needed}</h2><p>Runs Still to Come</p></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="metric-card"><h2>{req_rr}</h2><p>Required Run Rate (rest)</p></div>', unsafe_allow_html=True)
        with s3:
            pace = "🔥 Aggressive" if run_rate > 9 else ("⚡ Good" if run_rate > 7.5 else "🐢 Slow")
            st.markdown(f'<div class="metric-card"><h2>{pace}</h2><p>Batting Pace</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="section-title">📈 Score Projection</div>', unsafe_allow_html=True)
            overs_range = list(range(1, 21))
            projected   = []
            for ov in overs_range:
                if ov <= current_over:
                    projected.append(round(current_score * ov / current_over))
                else:
                    remaining_rr = runs_needed / overs_left if overs_left > 0 else 0
                    projected.append(current_score + round(remaining_rr * (ov - current_over)))
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=overs_range[:current_over], y=projected[:current_over],
                mode='lines+markers', name='Actual Progress',
                line=dict(color='#f59e0b', width=3), marker=dict(size=5)))
            fig1.add_trace(go.Scatter(
                x=overs_range[current_over-1:], y=projected[current_over-1:],
                mode='lines+markers', name='Projected',
                line=dict(color='#ef4444', width=3, dash='dash'), marker=dict(size=5)))
            # Show range band
            fig1.add_hrect(y0=lower, y1=upper, fillcolor='#f59e0b',
                           opacity=0.08, line_width=0,
                           annotation_text=f"Range {lower}–{upper}",
                           annotation_font_color='#f59e0b')
            fig1.add_hline(y=predicted, line_dash="dot", line_color="#22c55e",
                           annotation_text=f"Predicted: {predicted}",
                           annotation_font_color="#22c55e")
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,50,0.6)',
                font=dict(color='#cbd5e1'), legend=dict(font=dict(color='#cbd5e1')),
                xaxis=dict(title='Over', gridcolor='#1e3a5f', color='#94a3b8'),
                yaxis=dict(title='Runs', gridcolor='#1e3a5f', color='#94a3b8'),
                margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="section-title">🎯 Score Range Distribution</div>', unsafe_allow_html=True)
            # Use dynamic margin for distribution — wider when uncertain
            spread = (margin_low + margin_high) / 2
            np.random.seed(42)
            sim_scores = np.clip(
                np.random.normal(predicted, spread * 0.6, 500).astype(int),
                lower, upper)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=sim_scores, nbinsx=25,
                marker=dict(color='#f59e0b', opacity=0.8,
                            line=dict(color='#ef4444', width=1))))
            fig2.add_vline(x=predicted, line_dash="dash", line_color="#22c55e",
                           annotation_text=f"Most likely: {predicted}",
                           annotation_font_color="#22c55e")
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,50,0.6)',
                font=dict(color='#cbd5e1'),
                xaxis=dict(title='Final Score', gridcolor='#1e3a5f', color='#94a3b8'),
                yaxis=dict(title='Frequency', gridcolor='#1e3a5f', color='#94a3b8'),
                margin=dict(l=10, r=10, t=10, b=10), height=320, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
          <b>💡 Model Info:</b> Trained on 134,190 deliveries from 1,158 IPL matches (2008–2025).
          Algorithm: <b>Random Forest Regressor</b> · R² Score: <b>80.3%</b> · MAE: <b>±13.5 runs</b>
          &nbsp;|&nbsp; <b>Range method:</b> Dynamic — based on over, wickets & run rate
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:5rem;">🏏</div>
      <h2 style="color:#f59e0b;margin:12px 0;">Ready to Predict!</h2>
      <p style="color:#94a3b8;font-size:1.1rem;">
        Set the match situation in the sidebar and hit <b style='color:#f59e0b'>PREDICT SCORE</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown('<div class="metric-card"><h2>1,158</h2><p>IPL Matches</p></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="metric-card"><h2>134K+</h2><p>Deliveries</p></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="metric-card"><h2>2008–2025</h2><p>Seasons Covered</p></div>', unsafe_allow_html=True)
    with d4:
        st.markdown('<div class="metric-card"><h2>80.3%</h2><p>Model R² Score</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🏆 Team Win Rates (All Time)</div>', unsafe_allow_html=True)

    matches_df = load_matches()
    winner_col = None
    for col in matches_df.columns:
        if 'winner' in col.lower() or 'winning_team' in col.lower() or 'match_winner' in col.lower():
            winner_col = col
            break

    team_stats = {}
    if winner_col:
        for team in pd.concat([matches_df['team1'], matches_df['team2']]).unique():
            played = len(matches_df[(matches_df['team1']==team)|(matches_df['team2']==team)])
            won    = len(matches_df[matches_df[winner_col]==team])
            if played > 20:
                team_stats[team] = round(won/played*100, 1)

    team_df = pd.DataFrame(list(team_stats.items()),
                            columns=['Team','Win%']).sort_values('Win%', ascending=True)

    # Map team IDs to names for display
    TEAM_MAP = {
        1:'Royal Challengers Bangalore', 2:'Sunrisers Hyderabad',
        3:'Kolkata Knight Riders',       4:'Pune Warriors India',
        5:'Mumbai Indians',              6:'Chennai Super Kings',
        129:'Delhi Capitals',          134:'Rajasthan Royals',
        252:'Punjab Kings',            494:'Deccan Chargers',
        614:'Gujarat Titans',          615:'Lucknow Super Giants',
        1414:'Gujarat Lions',         1419:'Rising Pune Supergiant',
    }
    team_df['Team'] = team_df['Team'].apply(
        lambda x: TEAM_MAP.get(x, x) if isinstance(x, int) else x)

    fig3 = go.Figure(go.Bar(
        x=team_df['Win%'], y=team_df['Team'], orientation='h',
        marker=dict(color=team_df['Win%'], colorscale='YlOrRd', showscale=False),
        text=team_df['Win%'].astype(str)+'%',
        textposition='outside', textfont=dict(color='#cbd5e1')))
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,32,50,0.6)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(title='Win %', gridcolor='#1e3a5f', color='#94a3b8'),
        yaxis=dict(color='#94a3b8'),
        margin=dict(l=10, r=60, t=10, b=10), height=420)
    st.plotly_chart(fig3, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px;color:#475569;font-size:0.8rem;
            border-top:1px solid #1e3a5f;margin-top:30px;">
  🏏 IPL Score Predictor · Built with Streamlit & Scikit-learn · Real IPL Data 2008–2025
</div>
""", unsafe_allow_html=True)
