"""Persist benchmark results as CSV/Markdown and, optionally, figures."""

from __future__ import annotations

import logging
from pathlib import Path

from vfm.evaluation.benchmark import BenchmarkResult
from vfm.models import LABELS

logger = logging.getLogger(__name__)


def summary_markdown(result: BenchmarkResult) -> str:
    table = result.summary().reset_index()
    lines = ["| Phase | Model | wMAPE | R² | Volume bias |", "|---|---|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.phase} | {LABELS.get(str(row.model), row.model)} | {row.wmape:.1%} "
            f"| {row.r2:.2f} | {row.bias_pct:+.1f}% |"
        )
    return "\n".join(lines) + "\n"


def write_report(result: BenchmarkResult, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = result.protocol.replace("-", "_")
    result.metrics.to_csv(reports_dir / f"metrics_{stem}.csv", index=False, float_format="%.6g")
    (reports_dir / f"summary_{stem}.md").write_text(summary_markdown(result), encoding="utf-8")
    logger.info("Wrote %s", reports_dir / f"summary_{stem}.md")


def write_figures(
    result: BenchmarkResult,
    figures_dir: Path,
    well: str,
    phase: str = "oil",
    models: tuple[str, ...] = ("last_test", "physics", "gbm"),
) -> None:
    """Measured vs predicted rates for one well, plus wMAPE by model and phase."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures (pip install '.[report]')")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = result.protocol.replace("-", "_")
    preds = result.predictions
    subset = preds[(preds["well"] == well) & (preds["phase"] == phase)]
    reference = subset[subset["model"] == models[0]].set_index("date")["actual"].sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reference.index, reference.to_numpy(), color="black", lw=1.4, label="Allocated rate")
    palette = {
        "last_test": "tab:gray",
        "physics": "tab:blue",
        "gbm": "tab:orange",
        "hybrid": "tab:green",
    }
    for model in models:
        series = subset[subset["model"] == model].set_index("date")["predicted"].sort_index()
        ax.plot(
            series.index, series.to_numpy(), color=palette.get(model), lw=1.0, label=LABELS[model]
        )
    ax.set_title(f"Well {well}: daily {phase} rate, {result.protocol} (out of sample)")
    ax.set_ylabel(f"{phase} rate [Sm3/d]")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}_timeseries_{phase}.png", dpi=150)
    plt.close(fig)

    # The sensor-blind well mean is an order of magnitude worse; it stays in the table but
    # would flatten the chart.
    wmape = result.summary()["wmape"].unstack("model").reindex(["oil", "gas", "water"])
    shown = [m for m in LABELS if m in wmape.columns and m != "well_mean"]
    wmape = wmape[shown].rename(columns=LABELS)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    wmape.plot.bar(ax=ax, rot=0, width=0.85)
    ax.set_ylabel("wMAPE (lower is better)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}_wmape.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote figures to %s", figures_dir)
