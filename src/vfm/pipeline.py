"""End-to-end steps shared by the CLI and the dashboard."""

from __future__ import annotations

import logging
import warnings

import joblib
import pandas as pd

from vfm.config import Paths
from vfm.data import cleaning, source
from vfm.evaluation.benchmark import BenchmarkResult, Protocol, run_benchmark
from vfm.evaluation.report import write_figures, write_report
from vfm.features import add_features
from vfm.models import Meter, build_model

logger = logging.getLogger(__name__)

FIGURE_WELL = "15/9-F-14"


def prepare(paths: Paths, *, force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and cache the daily table as Parquet."""
    source.fetch(paths.raw_xlsx, force=force_download)
    with warnings.catch_warnings():
        # The workbook carries an Excel extension openpyxl does not model; it is harmless.
        warnings.filterwarnings("ignore", message="Unknown extension", module="openpyxl")
        raw = cleaning.read_workbook(paths.raw_xlsx)
    daily, report = cleaning.clean(raw)
    paths.daily_parquet.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(paths.daily_parquet, index=False)
    logger.info("Cleaning steps:\n%s", report.to_frame().to_string(index=False))
    logger.info(
        "Saved %d rows for %d wells to %s", len(daily), daily["well"].nunique(), paths.daily_parquet
    )
    return daily


def load_daily(paths: Paths) -> pd.DataFrame:
    if not paths.daily_parquet.exists():
        raise FileNotFoundError(f"{paths.daily_parquet} not found; run `vfm prepare` first")
    daily = pd.read_parquet(paths.daily_parquet)
    daily["well"] = daily["well"].astype("category")
    return daily


def benchmark(paths: Paths, protocol: Protocol = "walk-forward") -> BenchmarkResult:
    result = run_benchmark(load_daily(paths), protocol=protocol)
    write_report(result, paths.reports_dir)
    result.predictions.to_parquet(
        paths.data_dir / "processed" / f"predictions_{protocol.replace('-', '_')}.parquet",
        index=False,
    )
    write_figures(result, paths.figures_dir, well=FIGURE_WELL)
    return result


def train(paths: Paths, model_name: str = "physics") -> Meter:
    """Fit a meter on all available data and save it with joblib."""
    meter = build_model(model_name).fit(add_features(load_daily(paths)))
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    target = paths.models_dir / f"{model_name}.joblib"
    joblib.dump(meter, target)
    logger.info("Saved %s", target)
    return meter


def load_meter(paths: Paths, model_name: str = "physics") -> Meter:
    meter: Meter = joblib.load(paths.models_dir / f"{model_name}.joblib")
    return meter
