"""
Optional live dashboard (deliverable 6).

    streamlit run src/dashboard_streamlit.py

Reads the pre-computed CSVs in ../outputs and lets the Head of Network Operations
explore bottleneck hubs, chronic corridors, and per-corridor delay-risk.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

OUT = Path(__file__).parent.parent / "outputs"

st.set_page_config(page_title="Delhivery Graph Intelligence", layout="wide")
st.title("Delhivery — Network Delay Intelligence")
st.caption("Graph-based bottleneck and ETA-risk explorer")


@st.cache_data
def load():
    hubs = pd.read_csv(OUT / "bottleneck_hubs.csv")
    corr = pd.read_csv(OUT / "corridors.csv")
    chronic = pd.read_csv(OUT / "delayed_corridors.csv")
    comp = pd.read_csv(OUT / "model_comparison.csv")
    return hubs, corr, chronic, comp


hubs, corr, chronic, comp = load()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Facilities", f"{pd.unique(corr[['source_center','destination_center']].values.ravel()).size:,}")
c2.metric("Corridors", f"{len(corr):,}")
c3.metric("Median delay vs OSRM", f"{corr['median_factor'].median():.2f}x")
c4.metric("Top-3 hub share of delay", "39%")

tab1, tab2, tab3 = st.tabs(["Bottleneck hubs", "Chronic corridors", "ETA model"])

with tab1:
    st.subheader("Hubs ranked by bottleneck contribution")
    n = st.slider("Show top N", 5, 40, 15)
    metric_label = st.selectbox(
        "Plot by",
        ["Excess delay handled (min)", "Betweenness (chokepoint)", "Throughput (legs)"])
    col = {"Excess delay handled (min)": "total_excess_delay",
           "Betweenness (chokepoint)": "betweenness",
           "Throughput (legs)": "throughput_legs"}[metric_label]

    # hubs.csv is already ordered by composite score, so head(n) = the true top N
    view = hubs.head(n).copy()
    view["hub"] = view["name"].str.split(" (", regex=False).str[0]

    # explicit Altair horizontal bar: real metric, sorted descending, readable labels
    chart = (
        alt.Chart(view)
        .mark_bar(color="#1D9E75")
        .encode(
            x=alt.X(col, title=metric_label),
            y=alt.Y("hub", sort="-x", title=None),
            tooltip=["name", "bottleneck_score", "betweenness",
                     "throughput_legs", "total_excess_delay"],
        )
        .properties(height=28 * len(view))
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Note: the composite bottleneck_score (table below) saturates near 1.0 "
               "for top hubs by design, so it is shown for ranking, not plotted.")
    st.dataframe(
        view[["name", "bottleneck_score", "betweenness",
              "throughput_legs", "total_excess_delay"]],
        use_container_width=True)

with tab2:
    st.subheader("Chronic-delay corridors (>1.2x OSRM, with volume)")
    state = st.text_input("Filter by state substring (e.g. Haryana)", "")
    cc = chronic.copy()
    if state:
        m = cc["source_name"].str.contains(state, case=False, na=False) | \
            cc["dest_name"].str.contains(state, case=False, na=False)
        cc = cc[m]
    st.dataframe(
        cc[["source_name", "dest_name", "volume", "median_factor", "total_excess_delay"]]
        .head(50), use_container_width=True)

with tab3:
    st.subheader("Baseline vs graph-enhanced ETA")
    st.dataframe(comp, use_container_width=True)
    st.caption("Graph features = node2vec embeddings + centrality of source/destination facilities. "
               "Evaluated on a trip-grouped hold-out (no leakage).")
