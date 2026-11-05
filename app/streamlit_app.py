"""Interactive view of the virtual flow meter.

Run ``vfm prepare && vfm benchmark && vfm train`` first, then
``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vfm import pipeline
from vfm.config import PHASES, TARGETS, default_paths
from vfm.evaluation.metrics import regression_metrics
from vfm.features import add_features
from vfm.models import LABELS, ChokeMeter

PATHS = default_paths()
COLOURS = {
    "well_mean": "#9467bd",
    "last_test": "#7f7f7f",
    "physics": "#1f77b4",
    "gbm": "#ff7f0e",
    "hybrid": "#2ca02c",
}
st.set_page_config(page_title="Volve virtual flow meter", layout="wide")


@st.cache_data
def load_daily() -> pd.DataFrame:
    return add_features(pipeline.load_daily(PATHS))


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PATHS.data_dir / "processed" / "predictions_walk_forward.parquet")


@st.cache_resource
def load_meter() -> ChokeMeter:
    meter = pipeline.load_meter(PATHS, "physics")
    assert isinstance(meter, ChokeMeter)
    return meter


try:
    daily, predictions, meter = load_daily(), load_predictions(), load_meter()
except FileNotFoundError as missing:
    st.error(f"{missing}. Run `vfm prepare`, `vfm benchmark` and `vfm train` first.")
    st.stop()

st.title("Virtual flow metering — Volve field")
st.caption(
    "Daily oil, gas and water rates estimated from wellhead and downhole sensors. "
    "Walk-forward evaluation: every meter is recalibrated monthly and used blind for the "
    "next 30 days."
)

with st.sidebar:
    wells = sorted(predictions["well"].unique())
    well = st.selectbox("Well", wells, index=wells.index(pipeline.FIGURE_WELL))
    phase = st.radio("Phase", PHASES, horizontal=True)
    models = st.multiselect(
        "Meters",
        list(LABELS),
        default=["last_test", "physics", "gbm"],
        format_func=LABELS.__getitem__,
    )

subset = predictions[(predictions["well"] == well) & (predictions["phase"] == phase)]

# -- accuracy tiles --------------------------------------------------------------------
cols = st.columns(max(len(models), 1))
for col, model in zip(cols, models, strict=False):
    rows = subset[subset["model"] == model]
    m = regression_metrics(rows["actual"].to_numpy(), rows["predicted"].to_numpy())
    col.metric(
        LABELS[model],
        f"{m['wmape']:.1%} wMAPE",
        f"{m['bias_pct']:+.1f}% volume bias",
        delta_color="off",
    )

# -- time series -----------------------------------------------------------------------
fig = go.Figure()
reference = subset[subset["model"] == subset["model"].iloc[0]].sort_values("date")
fig.add_scatter(
    x=reference["date"],
    y=reference["actual"],
    name="Allocated rate",
    line={"color": "#d62728", "width": 2},
)
for model in models:
    rows = subset[subset["model"] == model].sort_values("date")
    fig.add_scatter(
        x=rows["date"],
        y=rows["predicted"],
        name=LABELS[model],
        line={"width": 1.2, "color": COLOURS.get(model)},
    )
for cutoff in subset["cutoff"].unique():
    fig.add_vline(x=cutoff, line={"color": "lightgray", "dash": "dot", "width": 1})
fig.update_layout(
    height=420,
    yaxis_title=f"{phase} rate [Sm3/d]",
    margin={"t": 10},
    legend={"orientation": "h", "y": -0.15},
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Dotted lines mark monthly recalibrations.")

# -- what-if ---------------------------------------------------------------------------
st.subheader("What if the choke is moved?")
well_rows = daily[daily["well"].astype(str) == well].sort_values("date")
latest = well_rows.tail(1)
left, right = st.columns([1, 2])
with left:
    choke = st.slider("Choke opening [%]", 5.0, 100.0, float(latest["choke_pct"].iloc[0]), 1.0)
    dp = st.slider(
        "Differential pressure across choke [bar]",
        1.0,
        40.0,
        float(np.nan_to_num(latest["dp_choke"].iloc[0], nan=5.0)),
        0.5,
    )
    scenario = latest.assign(choke_pct=choke, dp_choke=dp)
    now, then = meter.predict(latest), meter.predict(scenario)
    for target_phase in PHASES:
        target = TARGETS[target_phase]
        st.metric(
            f"{target_phase} [Sm3/d]",
            f"{then[target].iloc[0]:,.0f}",
            f"{then[target].iloc[0] - now[target].iloc[0]:+,.0f} vs last day",
        )
with right:
    grid = np.linspace(5.0, 100.0, 60)
    sweep = pd.concat([latest] * len(grid), ignore_index=True).assign(choke_pct=grid, dp_choke=dp)
    curve = meter.predict(sweep)[TARGETS[phase]]
    chart = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", name="Choke-ratio meter"))
    chart.add_vline(x=choke, line={"dash": "dash"})
    chart.update_layout(
        height=320,
        xaxis_title="Choke opening [%]",
        yaxis_title=f"{phase} rate [Sm3/d]",
        margin={"t": 10},
    )
    st.plotly_chart(chart, use_container_width=True)
exponents = meter.exponents.get(well, meter.pooled)
st.caption(
    f"Fitted choke equation for {well}: rate ∝ opening^{exponents.choke:.2f} · "
    f"Δp^{exponents.dp:.2f} (within-block estimate)."
)
