"""
Delhivery Graph Intelligence — consulting-grade analytics dashboard.
Reads pre-computed outputs from ../outputs; no heavy graph/ML at runtime.

    streamlit run src/dashboard_streamlit.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"
DATA = ROOT / "data"

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhivery · Graph Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── design tokens ──────────────────────────────────────────────────────────
ACCENT  = "#F59E0B"   # amber — primary accent
ACCENT2 = "#60A5FA"   # slate-blue — secondary
NEG     = "#F87171"   # coral — alert / median line
BG_PAGE = "#0F1117"
BG_CARD = "#161829"
BG_CARD2 = "#1E2235"
TEXT_HI  = "#F1F5F9"
TEXT_MU  = "#94A3B8"
BORDER   = "#2D3748"

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* typography */
  html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}

  /* hero banner */
  .hero {{
    background: linear-gradient(135deg, {BG_PAGE} 0%, {BG_CARD} 100%);
    border-left: 4px solid {ACCENT};
    padding: 1.6rem 2rem 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
  }}
  .hero-title {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {TEXT_HI};
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
  }}
  .hero-thesis {{
    font-size: 0.95rem;
    color: {TEXT_MU};
    margin: 0 0 0.5rem 0;
    line-height: 1.65;
    max-width: 860px;
  }}
  .hero-window {{
    font-size: 0.72rem;
    color: {ACCENT};
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0;
  }}

  /* KPI cards */
  .kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.25rem 0.9rem 1.25rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .kpi-value {{
    font-size: 1.7rem;
    font-weight: 800;
    color: {ACCENT};
    line-height: 1.1;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
  }}
  .kpi-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {TEXT_MU};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.15rem;
  }}
  .kpi-sub {{
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.1rem;
  }}

  /* section headers */
  .sec-title {{
    font-size: 1.1rem;
    font-weight: 800;
    color: {TEXT_HI};
    border-bottom: 2px solid {ACCENT};
    padding-bottom: 0.35rem;
    margin: 0 0 1rem 0;
    display: inline-block;
    letter-spacing: -0.01em;
  }}

  /* insight callout */
  .insight {{
    background: {BG_CARD2};
    border-left: 3px solid {ACCENT2};
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.855rem;
    color: #CBD5E1;
    margin: 0.8rem 0;
    line-height: 1.6;
  }}
  .insight strong {{ color: {TEXT_HI}; }}

  /* formula caption */
  .fcap {{
    font-size: 0.75rem;
    color: {TEXT_MU};
    font-style: italic;
    margin: 0.2rem 0 0.9rem 0;
    line-height: 1.4;
  }}

  /* formula label */
  .flabel {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1rem 0 0.2rem 0;
  }}

  /* divider */
  .div {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 2rem 0;
  }}

  /* suppress streamlit chrome */
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
  div[data-testid="stVerticalBlock"] > div {{ gap: 0; }}
</style>
""", unsafe_allow_html=True)


# ── data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_hubs():
    try:
        return pd.read_csv(OUT / "bottleneck_hubs.csv")
    except Exception:
        return None


@st.cache_data
def load_corridors():
    try:
        return pd.read_csv(OUT / "corridors.csv")
    except Exception:
        return None


@st.cache_data
def load_chronic():
    try:
        return pd.read_csv(OUT / "delayed_corridors.csv")
    except Exception:
        return None


@st.cache_data
def load_model():
    try:
        return pd.read_csv(OUT / "model_comparison.csv")
    except Exception:
        return None


@st.cache_data
def load_ftl_rec():
    try:
        return pd.read_csv(OUT / "ftl_carting_recommendations.csv")
    except Exception:
        return None


@st.cache_data
def load_ftl_obs():
    try:
        return pd.read_csv(OUT / "ftl_vs_carting_observed.csv")
    except Exception:
        return None


@st.cache_data
def load_legs():
    try:
        cols = ["factor", "is_delayed_20", "route_type", "tod_bucket",
                "source_state", "dest_state"]
        return pd.read_csv(DATA / "legs.csv", usecols=cols)
    except Exception:
        return None


hubs    = load_hubs()
corr    = load_corridors()
chronic = load_chronic()
comp    = load_model()
ftl_rec = load_ftl_rec()
ftl_obs = load_ftl_obs()
legs    = load_legs()


# ── altair theme helper ────────────────────────────────────────────────────
def _cfg(chart):
    """Apply a consistent dark-slate theme to any Altair chart."""
    return (
        chart
        .configure(background=BG_PAGE)
        .configure_view(strokeWidth=0, fill=BG_CARD)
        .configure_axis(
            labelColor=TEXT_MU,
            titleColor=TEXT_MU,
            gridColor=BORDER,
            domainColor=BORDER,
            tickColor=BORDER,
            labelFontSize=11,
            titleFontSize=11,
            titlePadding=8,
        )
        .configure_title(color=TEXT_HI, fontSize=13, fontWeight=600, anchor="start")
        .configure_legend(
            labelColor=TEXT_MU,
            titleColor=TEXT_MU,
            labelFontSize=10,
            titleFontSize=10,
            symbolSize=80,
        )
    )


# ── hero ───────────────────────────────────────────────────────────────────
n_fac  = (len(pd.unique(corr[["source_center", "destination_center"]].values.ravel()))
          if corr is not None else "—")
n_corr = len(corr) if corr is not None else "—"
n_legs = len(legs) if legs is not None else "—"

st.markdown(f"""
<div class="hero">
  <div class="hero-title">Delhivery &middot; Graph Intelligence</div>
  <div class="hero-thesis">
    OSRM under-predicts delivery time on <strong style="color:{ACCENT}">94.7 % of legs</strong>;
    delay is concentrated in a handful of hubs — graph-enhanced ETA reduces MAE by 11.6 %.
  </div>
  <div class="hero-window">
    Analysis window: Sep – Oct 2018 &nbsp;·&nbsp;
    {n_legs:,} legs &nbsp;·&nbsp; {n_fac:,} facilities &nbsp;·&nbsp; {n_corr:,} corridors
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ──────────────────────────────────────────────────────────────
med_fac = f"{legs['factor'].median():.2f}×" if legs is not None else "—"
pct_br  = (f"{legs['is_delayed_20'].mean() * 100:.1f} %"
           if legs is not None else "—")
top_hub = (hubs.iloc[0]["name"].split(" (")[0].replace("_", " ")
           if hubs is not None and len(hubs) else "—")
mae_imp = (f"{comp.iloc[1]['MAE_improvement_%']:.1f} %"
           if comp is not None else "—")

kpis = [
    (f"{n_fac:,}",   "Facilities",           "unique OD network nodes"),
    (f"{n_corr:,}",  "Corridors",            "unique OD pairs"),
    (med_fac,        "Median delay factor",   "actual ÷ OSRM time"),
    (pct_br,         "Legs breaching >1.2×",  "chronic over-run rate"),
    (top_hub,        "Top bottleneck hub",    "highest composite score"),
    (mae_imp,        "MAE improvement",       "graph vs. trip-feature baseline"),
]

k_cols = st.columns(len(kpis))
for col, (val, label, sub) in zip(k_cols, kpis):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 1  THE OSRM GAP ─────────────────────────────────────────────────────
st.markdown('<div class="sec-title">1 · The OSRM Gap</div>',
            unsafe_allow_html=True)

col_hist, col_math = st.columns([3, 2], gap="large")

with col_hist:
    if legs is not None:
        STEP = 0.2
        fv   = legs["factor"].dropna()
        fv   = fv[(fv >= 0.5) & (fv <= 8.0)]
        bins = np.arange(0.5, 8.0 + STEP, STEP)
        counts, edges = np.histogram(fv, bins=bins)
        hist_df = pd.DataFrame({
            "x":     edges[:-1].round(2),
            "x2":    edges[1:].round(2),
            "legs":  counts,
        })
        med_f = float(fv.median())

        bars = (
            alt.Chart(hist_df)
            .mark_bar(color=ACCENT, opacity=0.88, binSpacing=1)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0.5, 8]),
                         title="Delay factor  (actual ÷ OSRM predicted time)"),
                x2="x2:Q",
                y=alt.Y("legs:Q", title="Number of legs"),
                tooltip=[
                    alt.Tooltip("x:Q",    title="Factor ≥",  format=".2f"),
                    alt.Tooltip("x2:Q",   title="Factor <",  format=".2f"),
                    alt.Tooltip("legs:Q", title="Legs",       format=","),
                ],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"v": [med_f]}))
            .mark_rule(color=NEG, strokeWidth=2, strokeDash=[6, 3])
            .encode(x="v:Q")
        )
        label_df = pd.DataFrame({
            "v": [med_f],
            "lbl": [f"Median = {med_f:.1f}×"],
        })
        label = (
            alt.Chart(label_df)
            .mark_text(align="left", dx=7, color=NEG, fontSize=11, fontWeight=600)
            .encode(x="v:Q", y=alt.value(18), text="lbl:N")
        )
        chart1 = _cfg(
            (bars + rule + label)
            .properties(height=300, title="Distribution of per-leg delay factor (clipped at 8×)")
        )
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("legs.csv not found.")

with col_math:
    st.markdown('<div class="flabel">Delay factor definition</div>',
                unsafe_allow_html=True)
    st.latex(r"f = \dfrac{t_{\text{act}}}{t_{\text{osrm}}}")
    st.markdown(
        '<div class="fcap">f = 1.0 means on-time; every unit above 1 is '
        "a multiple of OSRM's predicted duration actually spent in transit.</div>",
        unsafe_allow_html=True,
    )

    if legs is not None:
        pct_v = legs["is_delayed_20"].mean() * 100
        above2 = (legs["factor"] > 2.0).mean() * 100
        st.markdown(f"""
        <div class="insight">
          <strong>{pct_v:.1f} % of legs</strong> have f &gt; 1.2 —
          OSRM systematically underestimates road time across virtually
          the entire network. The median factor is {med_f:.1f}×, meaning
          a typical delivery takes roughly <em>twice</em> the predicted
          duration. {above2:.0f} % of legs exceed even 2× the OSRM time,
          pointing to unmodelled dwell time, traffic, and multi-stop
          route structure — not statistical noise.
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 2  BOTTLENECK HUBS ──────────────────────────────────────────────────
st.markdown('<div class="sec-title">2 · Bottleneck Hubs</div>',
            unsafe_allow_html=True)

if hubs is not None:
    n_show = st.slider("Show top N hubs", 5, min(40, len(hubs)), 15, key="hub_n")
    view = hubs.head(n_show).copy()
    view["hub"]   = view["name"].str.split(" (", regex=False).str[0]
    view["state"] = view["name"].str.extract(r"\(([^)]+)\)", expand=False).fillna("")

    cb1, cb2 = st.columns(2, gap="medium")

    with cb1:
        bar_h = (
            alt.Chart(view)
            .mark_bar(color=ACCENT, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("total_excess_delay:Q",
                         title="Total excess delay absorbed (minutes)",
                         axis=alt.Axis(format=",.0f")),
                y=alt.Y("hub:N", sort="-x", title=None,
                         axis=alt.Axis(labelLimit=200)),
                tooltip=[
                    alt.Tooltip("name:N",               title="Hub"),
                    alt.Tooltip("total_excess_delay:Q",  title="Excess delay (min)",  format=",.0f"),
                    alt.Tooltip("bottleneck_score:Q",    title="Composite score",      format=".4f"),
                    alt.Tooltip("betweenness:Q",         title="Betweenness",          format=".4f"),
                    alt.Tooltip("throughput_legs:Q",     title="Legs",                 format=","),
                ],
            )
            .properties(height=max(260, 22 * n_show),
                        title="Excess delay absorbed per hub (sorted by composite score)")
        )
        st.altair_chart(_cfg(bar_h), use_container_width=True)

    with cb2:
        scat = (
            alt.Chart(view)
            .mark_circle(stroke=BG_PAGE, strokeWidth=1)
            .encode(
                x=alt.X("betweenness:Q",
                         title="Betweenness centrality  (fraction of shortest paths)"),
                y=alt.Y("total_excess_delay:Q",
                         title="Total excess delay (min)",
                         axis=alt.Axis(format=",.0f")),
                size=alt.Size("throughput_legs:Q",
                               title="Throughput (legs)",
                               scale=alt.Scale(range=[60, 800])),
                color=alt.Color("bottleneck_score:Q",
                                 title="Composite score",
                                 scale=alt.Scale(scheme="goldorange")),
                tooltip=[
                    alt.Tooltip("hub:N",                title="Hub"),
                    alt.Tooltip("betweenness:Q",        title="Betweenness",         format=".4f"),
                    alt.Tooltip("total_excess_delay:Q", title="Excess delay (min)",  format=",.0f"),
                    alt.Tooltip("throughput_legs:Q",    title="Throughput (legs)",   format=","),
                    alt.Tooltip("bottleneck_score:Q",   title="Composite score",     format=".4f"),
                ],
            )
            .properties(height=max(260, 22 * n_show),
                        title="Betweenness vs. excess delay (bubble size = throughput)")
        )
        st.altair_chart(_cfg(scat), use_container_width=True)

    # formulas
    fm1, fm2, fm3 = st.columns(3, gap="large")
    with fm1:
        st.markdown('<div class="flabel">Betweenness centrality</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"C_B(v) = \sum_{s \neq v \neq t} \dfrac{\sigma_{st}(v)}{\sigma_{st}}"
        )
        st.markdown(
            '<div class="fcap">Fraction of all shortest paths between every pair '
            "(s, t) that route through hub v — a structural chokepoint measure.</div>",
            unsafe_allow_html=True,
        )
    with fm2:
        st.markdown('<div class="flabel">Composite bottleneck score</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"S(v) = 0.5\,p_{\text{delay}}(v) + 0.3\,p_{\text{betw}}(v)"
            r"+ 0.2\,p_{\text{thru}}(v)"
        )
        st.markdown(
            '<div class="fcap">Weighted sum of percentile ranks (each p ∈ [0, 1]): '
            "delay is the primary signal, betweenness the structural signal, "
            "throughput the volume signal.</div>",
            unsafe_allow_html=True,
        )
    with fm3:
        st.markdown('<div class="flabel">Excess delay per hub</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"D(v) = \sum_{e \ni v}\left(t^{\text{act}}_e - t^{\text{osrm}}_e\right)"
        )
        st.markdown(
            '<div class="fcap">Total minutes of delay accumulated across every leg '
            "touching hub v — combines frequency and severity of delay.</div>",
            unsafe_allow_html=True,
        )

    top_row = hubs.iloc[0]
    top_name  = top_row["name"].split(" (")[0]
    top_state = top_row["name"]
    top_delay = top_row["total_excess_delay"]
    top_betw  = top_row["betweenness"]
    st.markdown(f"""
    <div class="insight">
      <strong>What this means for operations:</strong>
      {top_name} ({top_state.split("(")[1].rstrip(")")}) absorbs
      <strong>{top_delay:,.0f} minutes</strong> of excess delay — the highest of any
      node in the network. Its betweenness of {top_betw:.3f} means it sits on
      {top_betw * 100:.1f} % of all shortest inter-hub paths. An operational
      intervention here — priority docking, express slot booking, additional
      sortation capacity — propagates benefit across the widest possible set of
      downstream corridors.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Full hub ranking table"):
        tbl = view[["hub", "state", "bottleneck_score", "betweenness",
                     "throughput_legs", "total_excess_delay",
                     "delay_pct", "betw_pct", "thru_pct"]].copy()
        tbl.columns = ["Hub", "State", "Score", "Betweenness", "Legs",
                       "Excess delay (min)", "Delay pctile",
                       "Betw pctile", "Thru pctile"]
        st.dataframe(
            tbl.style.format({
                "Score":            "{:.4f}",
                "Betweenness":      "{:.4f}",
                "Legs":             "{:,.0f}",
                "Excess delay (min)": "{:,.0f}",
                "Delay pctile":     "{:.3f}",
                "Betw pctile":      "{:.3f}",
                "Thru pctile":      "{:.3f}",
            }),
            use_container_width=True,
        )
else:
    st.info("bottleneck_hubs.csv not found.")

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 3  CHRONIC CORRIDORS ─────────────────────────────────────────────────
st.markdown('<div class="sec-title">3 · Chronic Corridors</div>',
            unsafe_allow_html=True)

if chronic is not None:
    cc_ctrl, cc_info = st.columns([1, 2], gap="large")
    with cc_ctrl:
        state_filt = st.text_input(
            "Filter by state substring",
            placeholder="e.g. Haryana, Karnataka",
            label_visibility="visible",
        )
        top_k = st.slider("Show top K corridors", 5, 50, 20, key="corr_k")

    cc = chronic.copy()
    cc["src_short"]  = cc["source_name"].str.split(" (", regex=False).str[0]
    cc["dst_short"]  = cc["dest_name"].str.split(" (", regex=False).str[0]
    cc["label"] = (cc["src_short"] + " → " + cc["dst_short"]).str[:48]

    if state_filt.strip():
        mask = (
            cc["source_name"].str.contains(state_filt, case=False, na=False)
            | cc["dest_name"].str.contains(state_filt, case=False, na=False)
        )
        cc = cc[mask]

    cc = cc.head(top_k).reset_index(drop=True)

    if len(cc) == 0:
        st.warning("No corridors match that filter.")
    else:
        ch1, ch2 = st.columns(2, gap="medium")

        with ch1:
            bar_c = (
                alt.Chart(cc)
                .mark_bar(color=ACCENT,
                           cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    x=alt.X("total_excess_delay:Q",
                             title="Total excess delay (min)",
                             axis=alt.Axis(format=",.0f")),
                    y=alt.Y("label:N", sort="-x", title=None,
                             axis=alt.Axis(labelLimit=240)),
                    tooltip=[
                        alt.Tooltip("label:N",              title="Corridor"),
                        alt.Tooltip("total_excess_delay:Q", title="Excess delay (min)", format=",.0f"),
                        alt.Tooltip("median_factor:Q",      title="Median factor",      format=".2f"),
                        alt.Tooltip("volume:Q",             title="Volume (legs)",       format=".0f"),
                    ],
                )
                .properties(height=max(200, 18 * len(cc)),
                             title="Excess delay by corridor")
            )
            st.altair_chart(_cfg(bar_c), use_container_width=True)

        with ch2:
            br_c = (
                alt.Chart(cc)
                .mark_bar(color=ACCENT2,
                           cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    x=alt.X("breach_rate:Q",
                             title="Breach rate (fraction of legs with f > 1.2)",
                             scale=alt.Scale(domain=[0, 1]),
                             axis=alt.Axis(format=".0%")),
                    y=alt.Y(
                        "label:N",
                        sort=alt.EncodingSortField(
                            "total_excess_delay", order="descending"
                        ),
                        title=None,
                        axis=alt.Axis(labelLimit=240),
                    ),
                    tooltip=[
                        alt.Tooltip("label:N",         title="Corridor"),
                        alt.Tooltip("breach_rate:Q",   title="Breach rate",  format=".1%"),
                        alt.Tooltip("volume:Q",        title="Volume (legs)", format=".0f"),
                        alt.Tooltip("median_factor:Q", title="Median factor", format=".2f"),
                    ],
                )
                .properties(height=max(200, 18 * len(cc)),
                             title="Breach rate per corridor (fraction of legs exceeding 1.2×)")
            )
            st.altair_chart(_cfg(br_c), use_container_width=True)

        st.markdown("""
        <div class="insight">
          <strong>What this means for operations:</strong>
          High breach rate + high excess delay is the priority signal — both dimensions
          must be elevated to justify SLA renegotiation or route re-engineering.
          Corridors with breach rate = 1.0 are systemically broken on every single leg;
          those with moderate breach rate but high total delay carry large volume and
          warrant capacity or scheduling fixes.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Corridor detail table"):
            disp_cc = cc[["label", "volume", "median_factor",
                           "total_excess_delay", "breach_rate",
                           "median_distance", "ftl_share"]].copy()
            disp_cc.columns = ["Corridor", "Volume", "Median factor",
                                "Excess delay (min)", "Breach rate",
                                "Median distance (km)", "FTL share"]
            st.dataframe(
                disp_cc.style.format({
                    "Median factor":       "{:.2f}×",
                    "Excess delay (min)":  "{:,.0f}",
                    "Breach rate":         "{:.1%}",
                    "Median distance (km)":"{:.0f}",
                    "FTL share":           "{:.1%}",
                    "Volume":              "{:,.0f}",
                }),
                use_container_width=True,
            )
else:
    st.info("delayed_corridors.csv not found.")

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 4  GRAPH-ENHANCED ETA ────────────────────────────────────────────────
st.markdown('<div class="sec-title">4 · Graph-Enhanced ETA Model</div>',
            unsafe_allow_html=True)

if comp is not None:
    base_row = comp[comp["model"].str.contains("Baseline", case=False)].iloc[0]
    enh_row  = comp[~comp["model"].str.contains("Baseline", case=False)].iloc[0]

    ec1, ec2 = st.columns([3, 2], gap="large")

    with ec1:
        # MAE comparison
        mae_df = pd.DataFrame({
            "Model": ["Baseline\n(trip features)", "Graph-enhanced\n(+node2vec+centrality)"],
            "MAE":   [base_row["MAE_min"], enh_row["MAE_min"]],
            "Color": ["Baseline", "Graph-enhanced"],
        })
        mae_lo = min(mae_df["MAE"]) * 0.88
        mae_hi = max(mae_df["MAE"]) * 1.04
        mae_chart = (
            alt.Chart(mae_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=60)
            .encode(
                x=alt.X("Model:N", title=None,
                         axis=alt.Axis(labelAngle=0, labelLimit=220)),
                y=alt.Y("MAE:Q", title="MAE (minutes)",
                         scale=alt.Scale(domain=[mae_lo, mae_hi])),
                color=alt.Color(
                    "Color:N",
                    scale=alt.Scale(
                        domain=["Baseline", "Graph-enhanced"],
                        range=[ACCENT2, ACCENT],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("MAE:Q",   title="MAE (min)", format=".2f"),
                ],
            )
            .properties(height=230, title="Mean Absolute Error — lower is better")
        )
        st.altair_chart(_cfg(mae_chart), use_container_width=True)

        # within-15% comparison
        w15_df = pd.DataFrame({
            "Model":   ["Baseline\n(trip features)", "Graph-enhanced\n(+node2vec+centrality)"],
            "W15":     [base_row["within_15pct"], enh_row["within_15pct"]],
            "Color":   ["Baseline", "Graph-enhanced"],
        })
        w15_lo = min(w15_df["W15"]) * 0.94
        w15_hi = max(w15_df["W15"]) * 1.04
        w15_chart = (
            alt.Chart(w15_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=60)
            .encode(
                x=alt.X("Model:N", title=None,
                         axis=alt.Axis(labelAngle=0, labelLimit=220)),
                y=alt.Y("W15:Q", title="% of legs within ±15 % of actual",
                         scale=alt.Scale(domain=[w15_lo, w15_hi])),
                color=alt.Color(
                    "Color:N",
                    scale=alt.Scale(
                        domain=["Baseline", "Graph-enhanced"],
                        range=[ACCENT2, ACCENT],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("W15:Q",   title="Within-15 %", format=".2f"),
                ],
            )
            .properties(height=230, title="Within-15 % accuracy — higher is better")
        )
        st.altair_chart(_cfg(w15_chart), use_container_width=True)

    with ec2:
        st.markdown('<div class="flabel">Mean Absolute Error</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"\text{MAE} = \frac{1}{N}\sum_{i}\left|\hat{y}_i - y_i\right|"
        )
        st.markdown(
            '<div class="fcap">Average absolute gap (minutes) between predicted '
            "and actual transit time across held-out legs.</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="flabel">Within-15 % accuracy</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"\frac{1}{N}\sum_{i}"
            r"\mathbf{1}\!\left[\frac{|\hat{y}_i-y_i|}{y_i}\le 0.15\right]"
        )
        st.markdown(
            '<div class="fcap">Fraction of predictions within ±15 % of ground '
            "truth — the SLA-relevant accuracy metric for delivery promise windows.</div>",
            unsafe_allow_html=True,
        )

        mae_g = enh_row["MAE_improvement_%"]
        w15_g = enh_row["within15_gain_pts"]
        st.markdown(f"""
        <div class="insight">
          <strong>Honest framing:</strong>
          Graph features improve MAE by <strong>{mae_g:.1f} %</strong>
          ({base_row['MAE_min']:.1f} → {enh_row['MAE_min']:.1f} min) and
          within-15 % accuracy by <strong>+{w15_g:.1f} pp</strong>
          ({base_row['within_15pct']:.1f} → {enh_row['within_15pct']:.1f} %).
          This is a real but modest improvement — the baseline already achieves
          R² = {base_row['R2']:.3f}. The primary value of graph enrichment is
          interpretability: which structural hubs drive prediction uncertainty,
          not raw accuracy alone.
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("model_comparison.csv not found.")

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 5  FTL vs CARTING ───────────────────────────────────────────────────
st.markdown('<div class="sec-title">5 · FTL vs. Carting Analysis</div>',
            unsafe_allow_html=True)

BAND_ORDER = ["short <50km", "mid 50-200", "long 200-800", "ultra >800"]

if ftl_obs is not None:
    obs = ftl_obs.copy()
    obs["dist_band"] = pd.Categorical(
        obs["dist_band"], categories=BAND_ORDER, ordered=True
    )
    obs = obs.sort_values(["dist_band", "route_type"]).reset_index(drop=True)

    fc1, fc2 = st.columns([3, 2], gap="large")

    with fc1:
        speed_ch = (
            alt.Chart(obs)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("dist_band:N", title="Distance band",
                         sort=BAND_ORDER,
                         axis=alt.Axis(labelAngle=-15)),
                y=alt.Y("median_speed_kmph:Q",
                         title="Median observed speed (km/h)"),
                color=alt.Color(
                    "route_type:N",
                    title="Route type",
                    scale=alt.Scale(
                        domain=["FTL", "Carting"],
                        range=[ACCENT, ACCENT2],
                    ),
                ),
                xOffset="route_type:N",
                tooltip=[
                    alt.Tooltip("dist_band:N",        title="Distance band"),
                    alt.Tooltip("route_type:N",       title="Route type"),
                    alt.Tooltip("median_speed_kmph:Q",title="Median speed (km/h)", format=".1f"),
                    alt.Tooltip("median_factor:Q",    title="Median factor",       format=".2f"),
                    alt.Tooltip("legs:Q",             title="Legs",                format=","),
                ],
            )
            .properties(
                height=280,
                title="Observed median speed by distance band and route type",
            )
        )
        st.altair_chart(_cfg(speed_ch), use_container_width=True)

        factor_ch = (
            alt.Chart(obs)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("dist_band:N", title="Distance band",
                         sort=BAND_ORDER,
                         axis=alt.Axis(labelAngle=-15)),
                y=alt.Y("median_factor:Q",
                         title="Median delay factor (actual ÷ OSRM)"),
                color=alt.Color(
                    "route_type:N",
                    title="Route type",
                    scale=alt.Scale(
                        domain=["FTL", "Carting"],
                        range=[ACCENT, ACCENT2],
                    ),
                ),
                xOffset="route_type:N",
                tooltip=[
                    alt.Tooltip("dist_band:N",     title="Distance band"),
                    alt.Tooltip("route_type:N",    title="Route type"),
                    alt.Tooltip("median_factor:Q", title="Median factor",  format=".2f"),
                    alt.Tooltip("legs:Q",          title="Legs",           format=","),
                ],
            )
            .properties(
                height=250,
                title="Delay factor by distance band (FTL vs. Carting)",
            )
        )
        st.altair_chart(_cfg(factor_ch), use_container_width=True)

    with fc2:
        st.markdown('<div class="flabel">Breakeven SLA logic</div>',
                    unsafe_allow_html=True)
        st.latex(
            r"\text{breakeven}_{\text{SLA}} = "
            r"\frac{C_{\text{FTL}} - C_{\text{carting}}}{\Delta t_{\text{saved}}}"
        )
        st.markdown(
            '<div class="fcap">Minimum time saving (minutes) at which FTL becomes '
            "cost-neutral vs. carting — computed per distance band × time-of-day "
            "bucket using tunable cost assumptions.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("""
        <div class="insight">
          <strong>Key finding:</strong> FTL achieves higher observed speed at
          every distance band, but the cost premium is only cleared at
          <strong>200–800 km</strong> where the time saving reliably
          exceeds the breakeven SLA (≈ 6–7 min).
          Short and mid-range legs favour carting because the breakeven
          threshold (30–67 min depending on time-of-day) is not consistently
          cleared. At ultra-long distances (&gt;800 km), carting is also
          recommended — the speed advantage narrows and relay / rail
          alternatives warrant investigation.
        </div>
        """, unsafe_allow_html=True)

if ftl_rec is not None:
    st.markdown("**Recommendation matrix — distance band × time-of-day**")
    rec = ftl_rec.copy()
    rec["dist_band"] = pd.Categorical(
        rec["dist_band"], categories=BAND_ORDER, ordered=True
    )
    rec = rec.sort_values(["dist_band", "tod_bucket"]).reset_index(drop=True)
    disp_r = rec[["dist_band", "tod_bucket", "legs",
                   "avg_time_saved_ftl_min", "median_breakeven_sla",
                   "recommend"]].copy()
    disp_r.columns = ["Distance band", "Time of day", "Legs",
                       "Avg time saved (min)", "Breakeven SLA (min)", "Recommend"]

    def _color_rec(col):
        return [
            "background-color: #1A3020; color: #4ADE80; font-weight:700"
            if v == "FTL"
            else "background-color: #1A1D2E; color: #94A3B8"
            for v in col
        ]

    st.dataframe(
        disp_r.style
        .apply(_color_rec, subset=["Recommend"])
        .format({
            "Legs":               "{:,}",
            "Avg time saved (min)": "{:.1f}",
            "Breakeven SLA (min)":  "{:.1f}",
        }),
        use_container_width=True,
        height=340,
    )
else:
    st.info("ftl_carting_recommendations.csv not found.")

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 6  NETWORK MAP ───────────────────────────────────────────────────────
net_png = OUT / "figures" / "fig_network.png"
if net_png.exists():
    st.markdown('<div class="sec-title">6 · Network Map</div>',
                unsafe_allow_html=True)
    st.image(
        str(net_png),
        use_container_width=True,
        caption=(
            "Pre-computed network graph — node size ∝ betweenness centrality, "
            "edge weight = median delay factor."
        ),
    )
    st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── § 7  METHODOLOGY & LIMITATIONS ────────────────────────────────────────
st.markdown('<div class="sec-title">7 · Methodology &amp; Limitations</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="insight" style="border-left-color:{ACCENT};">
  <strong>Data grain.</strong>
  The unit of analysis is a per-OD leg with cumulative timing columns un-cumulated —
  each leg reflects only the transit time between its source and destination hub,
  not the full trip. Leg-level factor values are therefore comparable across routes
  of different lengths.
</div>

<div class="insight" style="border-left-color:{ACCENT};">
  <strong>Edge weights.</strong>
  Graph edge weights use the <em>median</em> delay factor (robust to the 77× maximum
  observed in the raw data). Arithmetic mean would grossly overstate expected delay
  on most corridors due to the heavy right tail.
</div>

<div class="insight" style="border-left-color:{ACCENT};">
  <strong>Temporal scope.</strong>
  The dataset covers Sep – Oct 2018. Risk scores are computed from historical batch
  data and are <em>not</em> a live feed. They reflect structural network patterns —
  which hubs are structurally prone to delay — not real-time or seasonal disruptions.
</div>

<div class="insight" style="border-left-color:{ACCENT};">
  <strong>FTL / Carting cost model.</strong>
  Breakeven SLA thresholds are computed from tunable cost-per-leg assumptions.
  The <em>direction</em> of each recommendation is robust to reasonable cost
  parameter variation; the exact breakeven number depends on the operator's
  contractual cost structure and should be recalibrated before operational rollout.
</div>

<div class="insight" style="border-left-color:{ACCENT};">
  <strong>ETA model evaluation.</strong>
  The model is evaluated on a trip-grouped hold-out (no leakage across legs from
  the same trip). The 11.6 % MAE improvement is real but modest — graph features
  add interpretable structural context, not a step-change in raw predictive accuracy.
  The baseline already achieves R² = """ + (f"{base_row['R2']:.3f}" if comp is not None else "0.975") + """.
</div>
""", unsafe_allow_html=True)
