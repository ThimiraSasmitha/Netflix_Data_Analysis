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

    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.set_title("Movies vs TV Shows")
    return _save(fig, "01_type_distribution.png")

def plot_rating_distribution(table: pd.DataFrame, top: int = 12) -> Path:
    """Grouped bars of the most common ratings."""
    data = table.drop(columns="Total").head(top)
    fig, ax = plt.subplots(figsize=(11, 6))
    data.plot(kind="bar", ax=ax, color=_type_colors(data.columns), width=0.8)
    ax.set_title("Titles by Maturity Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of titles")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _hide_x_grid(ax)
    return _save(fig, "02_rating_distribution.png")

def plot_rating_groups(table: pd.DataFrame) -> Path:
    """Grouped bars of audience groups (Kids / Teens / Adults ...)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    table.plot(kind="bar", ax=ax, color=_type_colors(table.columns0), width=0.75)
    ax.set_title("Who is Netflix Content Made For?")
    ax.set_xlabel("")
    ax.set_ylabel("Number of titles")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _hide_x_grid(ax)
    return _save(fig, "03_rating_group.png")

# Time
def plot_additions_over_time(yearly: pd.DataFrame) -> Path:
    """Stacked bars: titles added per year."""
    data = yearly.drop(columns="Total")
    fig, ax = plt.subplots(figsize=(11, 6))
    data.plot(kind="bar", stacked=True, ax=ax, color=_type_colors(data.columns), width=0.8)
    ax.set_title("Titles Added to Netflix per Year")
    ax.set_xlabel("Year added")
    ax.set_ylabel("Number of titles")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _hide_x_grid(ax)
    return _save(fig, "04_additions_per_year.png")

def plot_release_decades(table: pd.DataFrame) -> Path:
    """How old is the catalog? Titles per release decade."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    table.plot(kind="bar", stacked=True, ax=ax, color=_type_colors(table.columns), width=0.8)
    ax.set_title("Catalog by Release Decade")
    ax.set_xlabel("Release decade")
    ax.set_ylabel("Number of titles")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _hide_x_grid(ax)
    return _save(fig, "06_release_decades.png")


def plot_release_lag(df: pd.DataFrame, max_years: int = 15) -> Path:
    """Histogram of years between release and arrival on Netflix."""
    lag = df["release_lag_years"].dropna().astype(float).clip(upper=max_years)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(lag, bins=np.arange(0, max_years + 2) - 0.5, color=NETFLIX_RED, ax=ax)
    ax.axvline(lag.median(), color=NETFLIX_BLACK, linestyle="--", label=f"Median: {lag.median():.0f} yr")
    ax.set_title("Years Between Release and Netflix Arrival")
    ax.set_xlabel(f"Years (capped at {max_years}+)")
    ax.set_ylabel("Number of titles")
    ax.legend()
    _hide_x_grid(ax)
    return _save(fig, "07_release_lag.png")

def plot_genre_trends(trend: pd.DataFrame) -> Path:
    """Lines: how the top genres grew over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    trend.plot(ax=ax, marker="o", linewidth=2, colormap="Set1")
    ax.set_title("Top Genres Over Time")
    ax.set_xlabel("Year added")
    ax.set_ylabel("Titles added")
    ax.legend(title="", loc="upper left")
    return _save(fig, "08_genre_trends.png")