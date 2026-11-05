"""Volve virtual flow meter: accuracy per well and a choke what-if.

Run ``make data benchmark train`` once, then ``make app``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vfm import pipeline
from vfm.config import TARGETS, default_paths
from vfm.evaluation.metrics import regression_metrics
from vfm.features import add_features
from vfm.models import LABELS, ChokeMeter

PATHS = default_paths()
REPO = "https://github.com/iamkane22/vfm-volve"
SHOWN = ("last_test", "physics", "gbm", "hybrid")
COLOURS = {
    "last_test": "#6B7280",
    "physics": "#0F766E",
    "gbm": "#F59E0B",
    "hybrid": "#6366F1",
}
UNITS = {"oil": "Sm³/d", "gas": "Sm³/d", "water": "Sm³/d"}
SHORT = {"last_test": "Last test", "physics": "Physics", "gbm": "Boosting", "hybrid": "Hybrid"}

st.set_page_config(page_title="Volve virtual flow meter", page_icon="🛢️", layout="wide")
st.html(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      h1 {font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.02em;}
      .lede {color: #4B5563; margin: -0.6rem 0 1.2rem 0; font-size: 0.98rem;}
      .kpi {border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 16px; background: #fff;}
      .kpi.best {border: 1.5px solid #0F766E; background: #F0FDFA;}
      .kpi .name {font-size: 0.82rem; color: #4B5563; display: flex; gap: 8px; align-items: center;}
      .kpi .swatch {width: 10px; height: 10px; border-radius: 3px; display: inline-block;}
      .kpi .value {font-size: 1.75rem; font-weight: 650; color: #111827; margin-top: 4px;}
      .kpi .sub {font-size: 0.8rem; color: #6B7280;}
      .kpi .tag {margin-left: auto; font-size: 0.7rem; color: #0F766E; font-weight: 600;}
      .note {color: #6B7280; font-size: 0.82rem;}
    </style>
    """
)


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
except FileNotFoundError:
    st.error("Data not prepared yet. Run `make data benchmark train`, then reload.")
    st.stop()

st.title("Volve virtual flow meter")
st.html(
    '<p class="lede">Daily oil, gas and water rates estimated from wellhead and downhole '
    "sensors on the Equinor Volve field. Every meter is recalibrated monthly and judged on "
    "the following 30 days it has not seen.</p>"
)

wells = sorted(predictions["well"].unique())
left, middle, right = st.columns([1.2, 1, 1.6])
well = left.selectbox("Well", wells, index=wells.index(pipeline.FIGURE_WELL))
phase = middle.segmented_control("Phase", ["oil", "gas", "water"], default="oil") or "oil"
picked = right.pills(
    "Meters",
    SHOWN,
    default=["last_test", "physics", "gbm"],
    selection_mode="multi",
    format_func=SHORT.__getitem__,
)
models = [m for m in SHOWN if m in (picked or [])] or ["physics"]

accuracy, what_if, method = st.tabs(["Accuracy", "Choke what-if", "How it works"])

# -- accuracy -----------------------------------------------------------------------
with accuracy:
    subset = predictions[(predictions["well"] == well) & (predictions["phase"] == phase)]
    scores = {
        m: regression_metrics(
            subset.loc[subset["model"] == m, "actual"].to_numpy(),
            subset.loc[subset["model"] == m, "predicted"].to_numpy(),
        )
        for m in models
    }
    best = min(scores, key=lambda m: scores[m]["wmape"])
    for column, model in zip(st.columns(len(models)), models, strict=True):
        s = scores[model]
        css = "kpi best" if model == best and len(models) > 1 else "kpi"
        tag = '<span class="tag">LOWEST ERROR</span>' if css == "kpi best" else ""
        column.html(
            f'<div class="{css}"><div class="name"><span class="swatch" '
            f'style="background:{COLOURS[model]}"></span>{LABELS[model]}{tag}</div>'
            f'<div class="value">{s["wmape"]:.1%}</div>'
            f'<div class="sub">error (wMAPE) · bias {s["bias_pct"]:+.1f}% · R² {s["r2"]:.2f}</div>'
            "</div>"
        )

    fig = go.Figure()
    actual = subset[subset["model"] == models[0]].sort_values("date")
    fig.add_scatter(
        x=actual["date"],
        y=actual["actual"],
        name="Allocated rate",
        line={"color": "#111827", "width": 2},
    )
    for model in models:
        rows = subset[subset["model"] == model].sort_values("date")
        fig.add_scatter(
            x=rows["date"],
            y=rows["predicted"],
            name=LABELS[model],
            line={
                "color": COLOURS[model],
                "width": 1.4,
                "dash": "dash" if model == "last_test" else "solid",
            },
        )
    for cutoff in subset["cutoff"].unique():
        fig.add_vline(x=cutoff, line={"color": "#E5E7EB", "width": 1})
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 12}},
        yaxis_title=f"{phase} rate [{UNITS[phase]}]",
        font={"family": "Inter, -apple-system, Helvetica, Arial, sans-serif", "size": 13},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#F3F4F6")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.html(
        '<p class="note">Out-of-sample predictions only. Thin grey lines mark the monthly '
        "recalibrations; each meter only uses data from before the line.</p>"
    )

# -- what-if ------------------------------------------------------------------------
with what_if:
    latest = daily[daily["well"].astype(str) == well].sort_values("date").tail(1)
    base_choke = float(latest["choke_pct"].iloc[0])
    base_dp = float(np.nan_to_num(latest["dp_choke"].iloc[0], nan=5.0))
    exponents = meter.exponents.get(well, meter.pooled)

    controls, outcome = st.columns([1, 2], gap="large")
    with controls:
        st.markdown(f"**Last recorded day:** {latest['date'].iloc[0]:%d %b %Y}")
        choke = st.slider("Choke opening [%]", 5.0, 100.0, base_choke, 1.0)
        dp = st.slider("Pressure drop across the choke [bar]", 1.0, 40.0, base_dp, 0.5)
        st.html(
            f'<p class="note">Fitted for {well}: rate ∝ opening<sup>{exponents.choke:.2f}</sup> · '
            f"Δp<sup>{exponents.dp:.2f}</sup>. Exponents come from short-term responses only "
            "(within-block estimate), so reservoir depletion does not leak into them.</p>"
        )

    now = meter.predict(latest)
    scenario = meter.predict(latest.assign(choke_pct=choke, dp_choke=dp))
    with outcome:
        cards = st.columns(3)
        for card, name in zip(cards, ("oil", "gas", "water"), strict=True):
            target = TARGETS[name]
            value, before = float(scenario[target].iloc[0]), float(now[target].iloc[0])
            change = value / before - 1.0 if before > 0 else 0.0
            card.metric(f"{name.capitalize()} [{UNITS[name]}]", f"{value:,.0f}", f"{change:+.1%}")

        grid = np.linspace(5.0, 100.0, 80)
        sweep = pd.concat([latest] * len(grid), ignore_index=True).assign(
            choke_pct=grid, dp_choke=dp
        )
        curve = meter.predict(sweep)[TARGETS[phase]]
        chart = go.Figure()
        chart.add_scatter(
            x=grid, y=curve, mode="lines", line={"color": "#0F766E", "width": 2.5}, name=phase
        )
        chart.add_scatter(
            x=[base_choke],
            y=[float(now[TARGETS[phase]].iloc[0])],
            mode="markers",
            marker={"size": 9, "color": "#9CA3AF"},
            name="last day",
        )
        chart.add_scatter(
            x=[choke],
            y=[float(scenario[TARGETS[phase]].iloc[0])],
            mode="markers",
            marker={"size": 11, "color": "#0F766E"},
            name="scenario",
        )
        chart.update_layout(
            template="plotly_white",
            height=320,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title="Choke opening [%]",
            yaxis_title=f"{phase} rate [{UNITS[phase]}]",
            showlegend=False,
        )
        chart.update_xaxes(showgrid=False)
        chart.update_yaxes(gridcolor="#F3F4F6")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

# -- method -------------------------------------------------------------------------
with method:
    st.markdown(
        f"""
**Data.** Equinor's Volve daily production (2008-2016), 6 producing wells, 7,730 well-days
after cleaning. Each rule of the cleaning is logged with its row count.

**Meters.**
- *Last test carried forward*: the well keeps producing what it produced in its last test.
- *Choke-ratio physics*: the last test scaled by the change in choke capacity,
  `opening^a · Δp^b`, with exponents fitted per well.
- *Gradient boosting*: trees on all sensors, the well as a categorical feature.
- *Hybrid*: physics plus a learned correction. It did not beat physics; kept as a
  documented negative result.

**Evaluation.** Walk-forward: recalibrate monthly, predict the next 30 days blind,
24 months in total. Never a random split.

Code, tests and the full write-up: [{REPO.removeprefix("https://")}]({REPO})
"""
    )
