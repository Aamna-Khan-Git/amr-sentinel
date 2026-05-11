import sys
import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from orchestrator import ask

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="AMR Sentinel API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class AskRequest(BaseModel):
        question: str = Field(..., min_length=5)
        k_literature: int = Field(5, ge=1, le=20)

    @app.get("/health")
    def health_check():
        return {"status": "ok", "service": "AMR Sentinel"}

    @app.post("/ask")
    def ask_endpoint(request: AskRequest):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")
        result = ask(question=request.question, k_literature=request.k_literature)
        if result.get("error") and not result.get("answer"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result


def _make_trend_chart(df):
    import plotly.graph_objects as go
    fig = go.Figure()
    if "country" in df.columns:
        for country in df["country"].unique():
            sub = df[df["country"] == country]
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["avg_pct_resistant"],
                mode="lines+markers", name=country,
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% resistant<extra>" + country + "</extra>"
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["avg_pct_resistant"],
            mode="lines+markers+text",
            text=df["avg_pct_resistant"].round(1).astype(str) + "%",
            textposition="top center",
            line=dict(color="#e63946", width=3),
            marker=dict(size=10),
            hovertemplate="<b>Year %{x}</b><br>%{y:.1f}% resistant<extra></extra>"
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="CRITICAL (50%)")
    fig.add_hline(y=25, line_dash="dot", line_color="orange", annotation_text="HIGH (25%)")
    fig.update_layout(
        title="Resistance Trend Over Time",
        xaxis_title="Year",
        yaxis_title="% Resistant",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def _make_compare_chart(df):
    import plotly.express as px
    import plotly.graph_objects as go
    df_sorted = df.sort_values("avg_pct_resistant", ascending=True)
    colors = []
    for v in df_sorted["avg_pct_resistant"]:
        if v >= 50:
            colors.append("#d62828")
        elif v >= 25:
            colors.append("#f77f00")
        elif v >= 10:
            colors.append("#fcbf49")
        else:
            colors.append("#2a9d8f")

    fig = go.Figure(go.Bar(
        x=df_sorted["avg_pct_resistant"],
        y=df_sorted["country"],
        orientation="h",
        marker_color=colors,
        text=df_sorted["avg_pct_resistant"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% resistant<extra></extra>"
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="CRITICAL")
    fig.add_vline(x=25, line_dash="dot", line_color="orange", annotation_text="HIGH")
    fig.update_layout(
        title="Resistance by Country",
        xaxis_title="% Resistant",
        xaxis=dict(range=[0, 110]),
        template="plotly_white",
        height=max(400, len(df_sorted) * 28),
        margin=dict(l=120)
    )
    return fig


def _make_top_resistant_chart(df):
    import plotly.graph_objects as go
    df["label"] = df["organism"] + " / " + df["antibiotic"]
    df_sorted = df.sort_values("avg_pct_resistant", ascending=True)
    colors = []
    for v in df_sorted["avg_pct_resistant"]:
        if v >= 50:
            colors.append("#d62828")
        elif v >= 25:
            colors.append("#f77f00")
        elif v >= 10:
            colors.append("#fcbf49")
        else:
            colors.append("#2a9d8f")

    fig = go.Figure(go.Bar(
        x=df_sorted["avg_pct_resistant"],
        y=df_sorted["label"],
        orientation="h",
        marker_color=colors,
        text=df_sorted["avg_pct_resistant"].round(1).astype(str) + "%",
        textposition="outside",
        customdata=df_sorted[["total_isolates", "n_countries"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x:.1f}% resistant<br>"
            "Isolates: %{customdata[0]:,.0f}<br>"
            "Countries: %{customdata[1]}<extra></extra>"
        )
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="CRITICAL (50%)")
    fig.add_vline(x=25, line_dash="dot", line_color="orange", annotation_text="HIGH (25%)")
    fig.update_layout(
        title="Top Resistant Organism / Antibiotic Combinations",
        xaxis_title="Average % Resistant",
        xaxis=dict(range=[0, 115]),
        template="plotly_white",
        height=max(400, len(df_sorted) * 32),
        margin=dict(l=200)
    )
    return fig


def _make_bubble_chart(df):
    import plotly.express as px
    if "total_isolates" not in df.columns or "n_countries" not in df.columns:
        return None
    df["label"] = df["organism"] + " / " + df["antibiotic"]
    fig = px.scatter(
        df,
        x="avg_pct_resistant",
        y="n_countries",
        size="total_isolates",
        color="avg_pct_resistant",
        color_continuous_scale=["#2a9d8f", "#fcbf49", "#f77f00", "#d62828"],
        hover_name="label",
        hover_data={"total_isolates": ":,.0f", "avg_pct_resistant": ":.1f", "n_countries": True},
        labels={
            "avg_pct_resistant": "% Resistant",
            "n_countries": "Number of Countries",
            "total_isolates": "Total Isolates"
        },
        title="Resistance Burden: % Resistant vs Geographic Spread",
        size_max=60,
    )
    fig.add_vline(x=50, line_dash="dash", line_color="red")
    fig.add_vline(x=25, line_dash="dot", line_color="orange")
    fig.update_layout(template="plotly_white", height=450, coloraxis_showscale=True)
    return fig


def _resistance_level(pct):
    if pct >= 50:
        return "🔴 CRITICAL"
    elif pct >= 25:
        return "🟠 HIGH"
    elif pct >= 10:
        return "🟡 MODERATE"
    else:
        return "🟢 LOW"


def run_streamlit_ui():
    try:
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
    except ImportError:
        print("Run: pip install streamlit pandas plotly")
        sys.exit(1)

    st.set_page_config(page_title="AMR Sentinel", page_icon="🦠", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white; border-radius: 10px; padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
    }
    .stButton button {
        background-color: #e63946; color: white;
        border-radius: 8px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://www.ecdc.europa.eu/sites/default/files/images/ECDC-logo.png", width=160)
        st.title("⚙️ Settings")
        api_key = st.text_input(
            "Anthropic API Key", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        k_lit = st.slider("PubMed abstracts (k)", min_value=1, max_value=15, value=5)
        st.markdown("---")
        st.markdown("**Resistance Thresholds**")
        st.markdown("🔴 CRITICAL: ≥50%")
        st.markdown("🟠 HIGH: 25–50%")
        st.markdown("🟡 MODERATE: 10–25%")
        st.markdown("🟢 LOW: <10%")
        st.markdown("---")
        st.caption("Data: EARS-Net / EFSA European AMR Surveillance")

    st.title("🦠 AMR Sentinel")
    st.caption("European Antimicrobial Resistance Surveillance & Action Dashboard")

    with st.expander("💡 Example questions", expanded=False):
        cols = st.columns(2)
        examples = [
            "How has ciprofloxacin resistance in E. coli changed over time?",
            "Compare tetracycline resistance in Salmonella across countries.",
            "Which organism/antibiotic pairs have the highest resistance?",
            "How has ampicillin resistance in E. coli trended over time?",
            "Compare CIP resistance in C. jejuni across European countries.",
            "What are the top resistant combinations in Enterococcus?",
        ]
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, key=ex, use_container_width=True):
                    st.session_state["prefill"] = ex

    prefill = st.session_state.pop("prefill", "")
    question = st.text_area(
        "Ask your AMR question",
        value=prefill,
        height=80,
        placeholder="e.g. Compare ciprofloxacin resistance in E. coli across European countries",
    )

    if st.button("🔍 Analyse", type="primary", use_container_width=False):
        if not question.strip():
            st.warning("Please type a question first.")
            st.stop()
        if not api_key:
            st.error("Please provide your Anthropic API key in the sidebar.")
            st.stop()

        os.environ["ANTHROPIC_API_KEY"] = api_key

        with st.spinner("Querying surveillance database and literature..."):
            result = ask(question=question, k_literature=k_lit)

        data   = result.get("data", {})
        lit    = result.get("literature", {})
        intent = result.get("intent")
        meta   = result.get("narrative_metadata", {})

        # ── KPI metrics row ───────────────────────────────────────────────
        if data.get("rows"):
            df = pd.DataFrame(data["rows"])
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)

            if "avg_pct_resistant" in df.columns:
                avg = df["avg_pct_resistant"].mean()
                mx  = df["avg_pct_resistant"].max()
                m1.metric("Average Resistance", f"{avg:.1f}%")
                m2.metric("Peak Resistance", f"{mx:.1f}%", delta=_resistance_level(mx))

            if "total_isolates" in df.columns:
                m3.metric("Total Isolates", f"{int(df['total_isolates'].sum()):,}")

            if "n_countries" in df.columns:
                m4.metric("Countries Covered", int(df["n_countries"].max()))
            elif "country" in df.columns:
                m4.metric("Countries Covered", df["country"].nunique())

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────
        if data.get("rows"):
            df = pd.DataFrame(data["rows"])

            if intent == "trend" and "year" in df.columns and "avg_pct_resistant" in df.columns:
                st.plotly_chart(_make_trend_chart(df), use_container_width=True)

            elif intent == "compare" and "country" in df.columns and "avg_pct_resistant" in df.columns:
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.plotly_chart(_make_compare_chart(df), use_container_width=True)
                with col2:
                    st.markdown("### Resistance Levels")
                    for _, row in df.sort_values("avg_pct_resistant", ascending=False).head(10).iterrows():
                        level = _resistance_level(row["avg_pct_resistant"])
                        st.markdown(f"{level} **{row['country']}** — {row['avg_pct_resistant']:.1f}%")

            elif intent == "top_resistant" and "organism" in df.columns:
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.plotly_chart(_make_top_resistant_chart(df), use_container_width=True)
                with col2:
                    bubble = _make_bubble_chart(df)
                    if bubble:
                        st.plotly_chart(bubble, use_container_width=True)

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Analysis & Recommendations")
        if result.get("error") and not result.get("answer"):
            st.error("Error: " + str(result["error"]))
        else:
            st.markdown(result["answer"])

        # ── Literature ────────────────────────────────────────────────────
        if lit.get("hits"):
            with st.expander("📚 PubMed abstracts (" + str(lit["hit_count"]) + " retrieved)"):
                for hit in lit["hits"]:
                    st.markdown(
                        "**[" + str(hit["rank"]) + "] " + hit["title"] + "** (" + str(hit["year"]) + ")"
                    )
                    st.caption("PMID: " + str(hit["pmid"]))
                    st.markdown("> " + hit["snippet"])
                    st.markdown("---")

        # ── Raw data ──────────────────────────────────────────────────────
        if data.get("rows"):
            with st.expander("🗃️ Raw surveillance data (" + str(data["row_count"]) + " rows)"):
                st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True)

        # ── Metadata ──────────────────────────────────────────────────────
        with st.expander("🔧 Metadata"):
            st.json({
                "intent": intent,
                "parsed_params": result.get("parsed_params"),
                "model": meta.get("model"),
                "input_tokens": meta.get("input_tokens"),
                "output_tokens": meta.get("output_tokens"),
                "lit_store_size": lit.get("store_size"),
            })


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--ui" in args:
        run_streamlit_ui()
    else:
        if not _FASTAPI_AVAILABLE:
            print("Run: pip install fastapi uvicorn")
            sys.exit(1)
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
