"""
IPL 2026 Prediction Dashboard — Cricket Green Theme (LOCAL MODE)
"""
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys_path_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path_root))
from model.train import SoftEnsemble  # noqa: F401 — required for pickle

# ─── PATHS ────────────────────────────────────────────────────────────────────

DB_PATH        = sys_path_root / "ipl.db"
STANDINGS_PATH = sys_path_root / "standings.json"

# ─── TEAM CONFIG ──────────────────────────────────────────────────────────────

TEAM_COLORS = {
    "Mumbai Indians":              "#004BA0",
    "Chennai Super Kings":         "#F5A800",
    "Royal Challengers Bengaluru": "#C8102E",
    "Kolkata Knight Riders":       "#6A0DAD",
    "Delhi Capitals":              "#0057A8",
    "Rajasthan Royals":            "#E91E8C",
    "Sunrisers Hyderabad":         "#F7611E",
    "Punjab Kings":                "#D71920",
    "Gujarat Titans":              "#4A4A4A",
    "Lucknow Super Giants":        "#00B4D8",
}
TEAM_SHORT = {
    "Mumbai Indians":              "MI",
    "Chennai Super Kings":         "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders":       "KKR",
    "Delhi Capitals":              "DC",
    "Rajasthan Royals":            "RR",
    "Sunrisers Hyderabad":         "SRH",
    "Punjab Kings":                "PBKS",
    "Gujarat Titans":              "GT",
    "Lucknow Super Giants":        "LSG",
}

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IPL 2026 Predictor (Local)",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

.stApp { background: #f0f4f0; color: #1a2e1a; }
h1,h2,h3 { font-family: 'Teko', sans-serif; letter-spacing: 0.04em; color: #1a2e1a; }

.hero-wrap {
    background: linear-gradient(135deg, #1a3320 0%, #2d5a2d 50%, #1a3320 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(45deg, transparent, transparent 40px,
        rgba(255,255,255,0.015) 40px, rgba(255,255,255,0.015) 80px);
}
.hero-title {
    font-family: 'Teko', sans-serif; font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 700; color: #e8f5e8; letter-spacing: 0.06em; line-height: 1; margin: 0;
}
.hero-accent { color: #7ecb7e; }
.hero-sub {
    color: #6aaa6a; font-size: 0.8rem; letter-spacing: 0.3em; text-transform: uppercase;
    margin-bottom: 0.5rem; font-family: 'Source Sans 3', sans-serif; font-weight: 600;
}
.hero-desc { color: #9dbf9d; font-size: 0.95rem; margin-top: 0.8rem; font-weight: 300; }

.kpi-card {
    background: #fff; border: 1px solid #d4e8d4; border-top: 4px solid #2d7a2d;
    border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
}
.kpi-val { font-family: 'Teko', sans-serif; font-size: 2.2rem; color: #1a5c1a; line-height: 1; font-weight: 600; }
.kpi-label { font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: #7a9a7a; margin-top: 0.3rem; font-weight: 600; }

.sec-head {
    font-family: 'Teko', sans-serif; font-size: 1.5rem; font-weight: 600; color: #1a3320;
    border-left: 4px solid #2d7a2d; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0; letter-spacing: 0.04em;
}

.card { background: #fff; border: 1px solid #d4e8d4; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.75rem; }
.card-sm { background: #fff; border: 1px solid #d4e8d4; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
.result-card {
    background: #fff; border: 1px solid #d4e8d4; border-left: 4px solid #2d7a2d;
    border-radius: 0 10px 10px 0; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
    display: flex; justify-content: space-between; align-items: center;
}

.pred-wrap { background: #fff; border: 1px solid #d4e8d4; border-radius: 14px; padding: 2rem; margin-top: 1rem; text-align: center; }
.pred-prob { font-family: 'Teko', sans-serif; font-size: 4rem; font-weight: 700; line-height: 1; }
.pred-team { font-size: 1rem; font-weight: 600; color: #1a2e1a; margin-top: 0.3rem; }
.pred-code { font-size: 0.8rem; color: #7a9a7a; font-weight: 600; letter-spacing: 0.1em; }

.model-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.6rem 0; border-bottom: 1px solid #e8f0e8; text-align: left; }
.model-name { font-size: 0.8rem; font-weight: 600; color: #4a6a4a; width: 160px; flex-shrink: 0; letter-spacing: 0.05em; text-transform: uppercase; }
.model-bar-wrap { flex: 1; background: #f0f4f0; border-radius: 4px; height: 10px; overflow: hidden; }
.model-bar { height: 100%; border-radius: 4px; }
.model-pct { font-family: 'Teko', sans-serif; font-size: 1.1rem; font-weight: 600; color: #1a5c1a; width: 52px; text-align: right; flex-shrink: 0; }

.stButton > button {
    background: #1a5c1a !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.05em !important; padding: 0.6rem 2rem !important;
}
.stButton > button:hover { background: #2d7a2d !important; }

.stSelectbox label, .stTextInput label {
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #4a6a4a !important;
}

button[data-baseweb="tab"] { color: #1a3320 !important; font-family: "Source Sans 3", sans-serif !important; font-weight: 600 !important; font-size: 0.85rem !important; }
button[data-baseweb="tab"]:hover { color: #2d7a2d !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1a5c1a !important; border-bottom-color: #2d7a2d !important; }

div[data-baseweb="select"] * { color: #1a2e1a !important; }
div[data-baseweb="select"] > div { background: #fff !important; border-color: #b0d0b0 !important; }

.stMarkdown p { color: #1a2e1a; }
.run-meta { font-size: 0.72rem; color: #9ab09a; letter-spacing: 0.08em; }
footer { visibility: hidden; }

header[data-testid="stHeader"] {
    background-color: #f0f4f0 !important; border-bottom: 1px solid #d4e8d4 !important;
}
header[data-testid="stHeader"] * { color: #1a2e1a !important; }
header[data-testid="stHeader"] button { color: #1a2e1a !important; background: transparent !important; }
header[data-testid="stHeader"] svg { fill: #1a2e1a !important; color: #1a2e1a !important; }
</style>
""", unsafe_allow_html=True)


# ─── LOCAL DATA LOADERS ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_sim_results():
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute("""
            SELECT run_id, run_at, n_simulations,
                   matches_played, matches_remaining, results_json
            FROM simulation_results
            ORDER BY run_at DESC LIMIT 1
        """).fetchone()
        conn.close()
        if not row:
            return {}
        return {
            "run_id":            row[0],
            "run_at":            row[1],
            "n_simulations":     row[2],
            "matches_played":    row[3],
            "matches_remaining": row[4],
            "results":           json.loads(row[5]),
        }
    except Exception as e:
        st.error(f"DB error: {e}")
        return {}


@st.cache_data(ttl=60)
def load_points_table():
    try:
        if STANDINGS_PATH.exists():
            data = json.loads(STANDINGS_PATH.read_text())
            rows = data.get("standings", [])
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"team": "Team", "M": "P"})
                df = df.sort_values(["Pts", "NRR"], ascending=False).reset_index(drop=True)
                df.index += 1
                return df

        conn    = sqlite3.connect(DB_PATH)
        matches = pd.read_sql("""
            SELECT team1, team2, winner, result, result_margin, date
            FROM matches
            WHERE season='2026'
              AND (winner IS NOT NULL OR result IN ('no result','abandoned'))
            ORDER BY date
        """, conn)
        conn.close()

        stats = defaultdict(lambda: {"P":0,"W":0,"L":0,"NR":0,"Pts":0})
        for _, r in matches.iterrows():
            t1, t2 = r["team1"], r["team2"]
            w      = r["winner"]
            result = (r.get("result") or "").lower()
            if result in ("no result", "abandoned") or (pd.isna(w) and "tie" not in result):
                for t in [t1, t2]:
                    stats[t]["P"] += 1; stats[t]["NR"] += 1; stats[t]["Pts"] += 1
                continue
            loser = t2 if w == t1 else t1
            for t in [t1, t2]: stats[t]["P"] += 1
            stats[w]["W"] += 1; stats[w]["Pts"] += 2; stats[loser]["L"] += 1

        rows = []
        for team, s in stats.items():
            rows.append({"Team": team, "P": s["P"], "W": s["W"], "L": s["L"],
                         "NR": s["NR"], "Pts": s["Pts"], "NRR": 0.0})
        df = pd.DataFrame(rows).sort_values(["Pts", "NRR"], ascending=False).reset_index(drop=True)
        df.index += 1
        return df
    except Exception as e:
        st.error(f"Standings error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_recent():
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql("""
            SELECT date, team1, team2, winner, result, result_margin, venue
            FROM matches
            WHERE season='2026' AND winner IS NOT NULL
            ORDER BY date DESC LIMIT 8
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Recent matches error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_venue_map():
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            "SELECT team1, team2, venue FROM matches WHERE venue IS NOT NULL", conn
        )
        conn.close()
        vm = defaultdict(set)
        for _, row in df.iterrows():
            vm[frozenset([row["team1"], row["team2"]])].add(row["venue"])
        return {k: sorted(v) for k, v in vm.items()}
    except Exception as e:
        st.error(f"Venue error: {e}")
        return {}


@st.cache_data(ttl=60)
def load_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT run_at, matches_played, results_json
            FROM simulation_results
            ORDER BY run_at ASC
        """).fetchall()
        conn.close()

        seen = {}
        for row in rows:
            seen[row[1]] = row

        records = []
        for row in seen.values():
            for team, prob in json.loads(row[2]).items():
                records.append({
                    "run_at":  row[0][:10],
                    "matches": int(row[1]),
                    "team":    team,
                    "prob":    round(prob * 100, 1),
                })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"History error: {e}")
        return pd.DataFrame()


def run_prediction(team1, team2, venue, toss_winner, toss_decision):
    from model.predict import predict_match
    conn = sqlite3.connect(DB_PATH)
    try:
        return predict_match(
            team1=team1, team2=team2, season="2026",
            venue=venue or None,
            toss_winner=None if toss_winner == "Unknown" else toss_winner,
            toss_decision=toss_decision,
            conn=conn,
        )
    finally:
        conn.close()


# ─── HERO ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
  <div class="hero-sub">Machine Learning · Monte Carlo · Real-time</div>
  <div class="hero-title">IPL 2026 <span class="hero-accent">Predictor</span></div>
  <div class="hero-desc">Ensemble model (XGBoost + Random Forest + Logistic Regression) updated after every match.</div>
</div>
""", unsafe_allow_html=True)

# ─── REFRESH BUTTON ───────────────────────────────────────────────────────────

colA, colB = st.columns([8, 1])
with colB:
    if st.button("🔄"):
        st.cache_data.clear()
        st.rerun()

# ─── LOAD + GUARD ─────────────────────────────────────────────────────────────

latest = load_sim_results()
if not latest:
    st.warning("No simulation results found. Run simulate.py first.")
    st.stop()

probs             = latest["results"]
run_time          = latest["run_at"][:16].replace("T", " ")
leader            = list(probs.keys())[0]
leader_p          = list(probs.values())[0]
matches_played    = latest.get("matches_played", 0)
matches_remaining = latest.get("matches_remaining", 0)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────

for col, (val, label) in zip(st.columns(4), [
    (matches_played,                 "Matches Played"),
    (matches_remaining,              "Remaining"),
    (TEAM_SHORT.get(leader, leader), "Favourite"),
    (f"{leader_p:.0%}",              "Win Probability"),
]):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f'<p class="run-meta">Last updated: {run_time} UTC &nbsp;·&nbsp; {int(latest["n_simulations"]):,} simulations per run</p>',
    unsafe_allow_html=True,
)

# ─── TABS (4 tabs — toss removed) ────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆  Win Probabilities",
    "📋  Points Table",
    "📈  Probability Trends",
    "🔮  Match Predictor",
])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-head">Tournament Win Probabilities</div>', unsafe_allow_html=True)

    teams  = list(probs.keys())
    pvals  = [v * 100 for v in probs.values()]
    colors = [TEAM_COLORS.get(t, "#2d7a2d") for t in teams]
    shorts = [TEAM_SHORT.get(t, t) for t in teams]

    fig = go.Figure(go.Bar(
        x=pvals, y=shorts, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pvals], textposition="outside",
        textfont=dict(color="#1a2e1a", size=12, family="Source Sans 3"),
        hovertemplate="<b>%{customdata}</b><br>%{x:.1f}%<extra></extra>",
        customdata=teams,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a2e1a", family="Source Sans 3"),
        xaxis=dict(showgrid=True, gridcolor="#d4e8d4", title="Win Probability (%)",
                   title_font_color="#7a9a7a", tickfont_color="#7a9a7a",
                   range=[0, max(pvals)*1.25], zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(size=13, color="#1a2e1a", family="Source Sans 3")),
        margin=dict(l=20, r=70, t=10, b=40), height=400, bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    badges = []
    for t in teams[:4]:
        c = TEAM_COLORS.get(t, "#2d7a2d")
        badges.append(
            f'<span style="display:inline-block;background:{c}22;border:1.5px solid {c};'
            f'color:{c};font-weight:700;font-size:0.8rem;padding:0.2rem 0.7rem;'
            f'border-radius:5px;margin:0.2rem;letter-spacing:0.05em">{TEAM_SHORT.get(t,t)}</span>'
        )
    st.markdown(
        f'<div class="card" style="border-top:3px solid #2d7a2d">'
        f'<div style="font-size:0.7rem;letter-spacing:0.2em;text-transform:uppercase;'
        f'color:#7a9a7a;margin-bottom:0.5rem;font-weight:600">Current Top-4 Playoff Contenders</div>'
        f'{" ".join(badges)}</div>',
        unsafe_allow_html=True,
    )


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    pts_df = load_points_table()
    recent = load_recent()

    st.markdown('<div class="sec-head">2026 Points Table</div>', unsafe_allow_html=True)
    if pts_df.empty:
        st.info("No 2026 matches completed yet.")
    else:
        cols_to_show = [c for c in ["Team","P","W","L","NR","Pts","NRR","Form"] if c in pts_df.columns]
        st.dataframe(pts_df[cols_to_show], use_container_width=True, height=400)
        st.caption("Top 4 rows = current playoff zone")

    st.markdown('<div class="sec-head">Recent Results</div>', unsafe_allow_html=True)
    if recent.empty:
        st.info("No results yet.")
    else:
        for _, r in recent.iterrows():
            wc     = TEAM_COLORS.get(r["winner"], "#2d7a2d")
            loser  = r["team2"] if r["winner"] == r["team1"] else r["team1"]
            margin = f"by {int(r['result_margin'])} {r['result']}" if pd.notna(r.get("result_margin")) else ""
            venue  = r.get("venue", "") or ""
            st.markdown(
                f'<div class="result-card"><div>'
                f'<span style="color:{wc};font-weight:700;font-size:1rem">{r["winner"]}</span>'
                f'<span style="color:#9ab09a;margin:0 0.5rem;font-size:0.9rem"> beat </span>'
                f'<span style="color:#4a6a4a;font-size:0.95rem">{loser}</span>'
                f'<span style="color:#1a5c1a;font-size:0.85rem;margin-left:0.5rem;font-weight:600"> {margin}</span>'
                f'</div><div style="text-align:right">'
                f'<div style="font-size:0.78rem;color:#7a9a7a;font-weight:600">{r["date"]}</div>'
                f'<div style="font-size:0.72rem;color:#9ab09a">{venue}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-head">Win Probability Trends</div>', unsafe_allow_html=True)
    hist = load_history()

    if hist.empty or hist["run_at"].nunique() < 2:
        st.info("Need at least 2 simulation runs to show trends.")
    else:
        selected = st.multiselect(
            "Filter teams", sorted(hist["team"].unique()),
            default=sorted(hist["team"].unique())[:6],
        )
        fig2 = px.line(
            hist[hist["team"].isin(selected)], x="matches", y="prob", color="team",
            color_discrete_map=TEAM_COLORS, markers=True,
            labels={"prob": "Win Probability (%)", "matches": "Matches Played"},
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1a2e1a", family="Source Sans 3"),
            xaxis=dict(showgrid=True, gridcolor="#d4e8d4", title_font_color="#7a9a7a", tickfont_color="#7a9a7a"),
            yaxis=dict(showgrid=True, gridcolor="#d4e8d4", title_font_color="#7a9a7a", tickfont_color="#7a9a7a"),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#d4e8d4", borderwidth=1, font=dict(color="#1a2e1a")),
            margin=dict(l=20, r=20, t=20, b=40), height=450,
        )
        fig2.update_traces(line=dict(width=2.5), marker=dict(size=8))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Each point = one simulation run after a match completed.")


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-head">Match Win Predictor</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a6a4a;margin-bottom:1.2rem">Select two teams to get head-to-head win probability from the ensemble model.</p>', unsafe_allow_html=True)

    teams_list = sorted(TEAM_COLORS.keys())
    venue_map  = load_venue_map()

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams_list, index=0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams_list if t != team1], index=0)

    pair_key    = frozenset([team1, team2])
    pair_venues = venue_map.get(pair_key, [])
    all_venues  = sorted(set(v for vlist in venue_map.values() for v in vlist))

    if pair_venues:
        venue_options = ["Select venue..."] + pair_venues + ["--- All venues ---"] + [v for v in all_venues if v not in pair_venues]
        venue_label   = f"Venue ({len(pair_venues)} historical matchups)"
    else:
        venue_options = ["Select venue..."] + all_venues
        venue_label   = "Venue"
    venue_sel = st.selectbox(venue_label, venue_options)
    venue     = None if venue_sel in ("Select venue...", "--- All venues ---") else venue_sel

    if st.button("Predict Match", use_container_width=True):
        if team1 == team2:
            st.error("Please select two different teams.")
        else:
            with st.spinner("Running ensemble prediction..."):
                try:
                    result = run_prediction(team1, team2, venue, "Unknown", "bat")
                    p1     = result["p_team1_wins"]
                    p2     = result["p_team2_wins"]
                    mp     = result["model_probs"]
                    c1c    = TEAM_COLORS.get(team1, "#1a5c1a")
                    c2c    = TEAM_COLORS.get(team2, "#00B4D8")
                    fav    = team1 if p1 > p2 else team2
                    fav_c  = TEAM_COLORS.get(fav, "#1a5c1a")

                    model_rows = ""
                    for mname, mprob in mp.items():
                        bar_color = "#1a5c1a" if mname == "Ensemble" else "#4a9a4a"
                        model_rows += (
                            f'<div class="model-row">'
                            f'<span class="model-name">{mname}</span>'
                            f'<div class="model-bar-wrap"><div class="model-bar" style="width:{mprob*100:.1f}%;background:{bar_color}"></div></div>'
                            f'<span class="model-pct">{mprob:.0%}</span>'
                            f'</div>'
                        )

                    st.markdown(
                        f'<div class="pred-wrap">'
                        f'<div style="display:flex;justify-content:space-around;align-items:center;gap:2rem;margin-bottom:1.5rem">'
                        f'<div><div class="pred-prob" style="color:{c1c}">{p1:.0%}</div>'
                        f'<div class="pred-team">{team1}</div>'
                        f'<div class="pred-code">{TEAM_SHORT.get(team1,"")}</div></div>'
                        f'<div style="font-family:Teko,sans-serif;font-size:1.8rem;color:#9ab09a;font-weight:600">VS</div>'
                        f'<div><div class="pred-prob" style="color:{c2c}">{p2:.0%}</div>'
                        f'<div class="pred-team">{team2}</div>'
                        f'<div class="pred-code">{TEAM_SHORT.get(team2,"")}</div></div>'
                        f'</div>'
                        f'<div style="background:#f0f4f0;border-radius:6px;height:10px;overflow:hidden;margin-bottom:1.5rem">'
                        f'<div style="height:100%;width:{p1*100:.1f}%;background:linear-gradient(90deg,{c1c},{c2c});border-radius:6px"></div>'
                        f'</div>'
                        f'<div style="text-align:left">'
                        f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:#7a9a7a;margin-bottom:0.8rem">Individual Model Breakdown</div>'
                        f'{model_rows}</div></div>',
                        unsafe_allow_html=True,
                    )

                    confidence = abs(p1 - p2)
                    conf_label = "Strong" if confidence > 0.2 else "Moderate" if confidence > 0.1 else "Narrow"
                    venue_line = f'<div style="font-size:0.78rem;color:#7a9a7a;margin-top:0.3rem">Venue: {venue}</div>' if venue else ""
                    st.markdown(
                        f'<div class="card-sm" style="border-left:4px solid {fav_c};border-radius:0 8px 8px 0;margin-top:0.6rem">'
                        f'<span style="color:{fav_c};font-weight:700">{fav}</span>'
                        f'<span style="color:#4a6a4a"> are favoured to win — <strong style="color:#1a5c1a">{conf_label} confidence</strong>'
                        f' ({confidence:.0%} margin between teams)</span>'
                        f'{venue_line}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:2.5rem 0 1rem;color:#9ab09a;'
    'font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase">'
    'IPL 2026 Predictor · Ensemble ML · Streamlit · Built for learning'
    '</div>',
    unsafe_allow_html=True,
)