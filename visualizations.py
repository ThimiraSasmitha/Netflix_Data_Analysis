"""
Charts for the Netflix analysis.

Each `plot_*` function takes the result table (or the cleaned dataframe),
saves a PNG into `outputs/figures/` and returns the file path.
`create_all_plots` runs them all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, NETFLIX_BLACK, NETFLIX_GRAY, NETFLIX_RED, TYPE_COLORS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100
    }
)

def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path.name)
    return path

def _hide_x_grid(ax: plt.Axes) -> None:
    """Vertical gridlines add noise to column charts."""
    ax.xaxis.grid(False)

def _type_colors(columns) -> list[str]:
    return [TYPE_COLORS.get(col, NETFLIX_GREY) for col in columns]

# Catalog composition

def plot_type_distribution(type_dist: pd.DataFrame) -> Path:
    """Donut chart of Movies vs TV Shows."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, autotexts = ax.pie(
        type_dist["count"],
        labels=type_dist["type"],
        colors=_type_colors(type_dist["type"]),
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
        wedgeprops={"width": 0.4, "edgecolor": "white"},
    )