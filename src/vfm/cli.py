"""Command-line interface: ``vfm prepare | benchmark | train``."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated

import typer

from vfm import pipeline
from vfm.config import default_paths
from vfm.models import MODEL_NAMES

app = typer.Typer(add_completion=False, help="Virtual flow metering on the Volve field.")


class ProtocolChoice(StrEnum):
    walk_forward = "walk-forward"
    holdout = "holdout"


@app.callback()
def _configure(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


@app.command()
def prepare(
    force_download: Annotated[bool, typer.Option(help="Re-download the workbook.")] = False,
) -> None:
    """Download the Volve workbook, clean it and cache it as Parquet."""
    pipeline.prepare(default_paths(), force_download=force_download)


@app.command()
def benchmark(
    protocol: Annotated[
        ProtocolChoice, typer.Option(help="Evaluation protocol.")
    ] = ProtocolChoice.walk_forward,
) -> None:
    """Score every meter out of sample and write reports/ and figures."""
    result = pipeline.benchmark(default_paths(), protocol=protocol.value)
    typer.echo(result.summary().to_string(float_format=lambda v: f"{v:.3f}"))


@app.command()
def train(
    model: Annotated[str, typer.Option(help=f"One of: {', '.join(MODEL_NAMES)}")] = "physics",
) -> None:
    """Fit the chosen meter on all data and save it to data/models/."""
    if model not in MODEL_NAMES:
        raise typer.BadParameter(f"choose from {', '.join(MODEL_NAMES)}")
    pipeline.train(default_paths(), model)


if __name__ == "__main__":
    app()
