import time
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AVC&C Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label  { color: #8b8fa8 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important; font-size: 2rem; font-weight: 700;
    }

    /* Section headers */
    h3 { color: #c9d1f0 !important; }

    /* Divider */
    hr { border-color: #2a2d3e; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #13151f; }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.78rem;
        color: #555878;
        padding: 24px 0 8px;
        border-top: 1px solid #2a2d3e;
        margin-top: 32px;
        line-height: 1.8;
    }
    .footer b { color: #8b8fa8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "vehicle_log.csv")
COLORS = {"car": "#1dcfaa", "bus": "#f0a500", "truck": "#5b9cf6"}
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1f0",
    xaxis=dict(gridcolor="#2a2d3e", linecolor="#2a2d3e"),
    yaxis=dict(gridcolor="#2a2d3e", linecolor="#2a2d3e"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=0, t=32, b=0),
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/University_of_Bradford_logo.svg/320px-University_of_Bradford_logo.svg.png",
        width=160,
    )
    st.markdown("## AVC&C System")
    st.caption("University of Bradford · Group G25")
    st.divider()

    weather_filter = st.selectbox(
        "Weather Filter",
        ["All", "Normal", "Rain", "Fog", "Snow", "Sand"],
        index=0,
    )
    auto_refresh = st.checkbox("Auto Refresh (5 s)", value=True)
    st.divider()
    st.caption("v1.0 · Bradford Council 2026")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🚦 Automated Vehicle Classification & Counting")
st.caption("Real-time traffic monitoring dashboard · University of Bradford 2026")
st.divider()

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("empty")
        return df
    except Exception:
        return pd.DataFrame(
            columns=["timestamp", "track_id", "class", "confidence", "x", "y", "weather"]
        )


df_raw = load_data(CSV_PATH)

if df_raw.empty:
    st.warning("No data yet — start `vehicle_counter.py` to begin logging detections.")

# Apply weather filter
df = df_raw.copy()
if weather_filter != "All" and not df.empty:
    df = df[df["weather"] == weather_filter]

# ── Metric cards ─────────────────────────────────────────────────────────────
cars   = int((df["class"] == "car").sum())   if not df.empty else 0
buses  = int((df["class"] == "bus").sum())   if not df.empty else 0
trucks = int((df["class"] == "truck").sum()) if not df.empty else 0
total  = cars + buses + trucks

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚗 Cars",    cars)
c2.metric("🚌 Buses",   buses)
c3.metric("🚛 Trucks",  trucks)
c4.metric("📊 Total",   total)

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Vehicle Class Breakdown")
    if not df.empty:
        counts = df["class"].value_counts().reset_index()
        counts.columns = ["class", "count"]
        fig_bar = px.bar(
            counts,
            x="class",
            y="count",
            color="class",
            color_discrete_map=COLORS,
            text="count",
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0)
        fig_bar.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Waiting for data…")

with col_right:
    st.subheader("Detections Over Time (by Minute)")
    if not df.empty:
        df_time = df.copy()
        df_time["minute"] = (
            pd.to_datetime(df_time["timestamp"], format="%H:%M:%S", errors="coerce")
            .dt.floor("1min")
            .dt.strftime("%H:%M")
        )
        timeline = (
            df_time.groupby(["minute", "class"])
            .size()
            .reset_index(name="count")
        )
        fig_line = px.line(
            timeline,
            x="minute",
            y="count",
            color="class",
            color_discrete_map=COLORS,
            markers=True,
        )
        fig_line.update_traces(line_width=2)
        fig_line.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Waiting for data…")

st.divider()

# ── Recent detections table ──────────────────────────────────────────────────
st.subheader("Recent Detections (last 20)")
if not df.empty:
    recent = df.tail(20).iloc[::-1].reset_index(drop=True)
    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp":  st.column_config.TextColumn("Timestamp"),
            "track_id":   st.column_config.NumberColumn("Track ID"),
            "class":      st.column_config.TextColumn("Class"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "x":          st.column_config.NumberColumn("X"),
            "y":          st.column_config.NumberColumn("Y"),
            "weather":    st.column_config.TextColumn("Weather"),
        },
    )
else:
    st.info("No detections to display yet.")

st.divider()

# ── Model performance ────────────────────────────────────────────────────────
st.subheader("Model Performance")

p1, p2, p3 = st.columns(3)

def perf_gauge(title: str, value: float, suffix: str = "%") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": "#ffffff", "size": 36}},
            title={"text": title, "font": {"color": "#8b8fa8", "size": 13}},
            gauge={
                "axis": {"range": [0, 100] if suffix == "%" else [0, 60],
                         "tickcolor": "#2a2d3e"},
                "bar": {"color": "#1dcfaa"},
                "bgcolor": "#1a1d27",
                "bordercolor": "#2a2d3e",
                "steps": [{"range": [0, 70], "color": "#151821"},
                           {"range": [70, 90], "color": "#1a1d27"}],
                "threshold": {
                    "line": {"color": "#f0a500", "width": 2},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1f0",
        height=220,
        margin=dict(l=20, r=20, t=40, b=0),
    )
    return fig

with p1:
    st.plotly_chart(perf_gauge("Detection Accuracy", 91.4), use_container_width=True)
with p2:
    st.plotly_chart(perf_gauge("Classification Accuracy", 88.6), use_container_width=True)
with p3:
    st.plotly_chart(perf_gauge("Processing Speed", 25, suffix=" FPS"), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <b>Ayush Acharya</b> — AI/ML Lead &nbsp;|&nbsp;
        <b>Pratham Patel</b> — Data &amp; Testing &nbsp;|&nbsp;
        <b>Aman Gill</b> — Dashboard &amp; Docs<br>
        Supervised by <b>Dr Kulwinder Panesar</b> · University of Bradford · Group G25 · 2026
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
