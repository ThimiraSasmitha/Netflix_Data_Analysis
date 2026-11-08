"""
Analysis functions for the Netflix catalog.
 
Every function takes the cleaned dataframe and returns a pandas object
(DataFrame / Series) so results can be exported, plotted or unit-tested
independently of each other.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from config import (
    MONTH_ABBR,
    RATING_GROUP_ORDER,
    TABLES_DIR,
    TOP_N,
    UNKNOWN,
)

logger = logging.getLogger(__name__)

# Helpers
def explode_multivalue(df: pd.DataFrame, column: str, sep: str = ",") -> pd.DataFrame:
    """
    Turn a comma-separated column into one row per value.
 
    'Dramas, Comedies' becomes two rows (one 'Dramas', one 'Comedies') that
    keep all the other columns of the original title. Empty and 'Unknown'
    entries are dropped.
    """
    exploded = df.assign(**{column: df[column].astype("string").str.split(sep)}).explode(column)
    exploded[column] = exploded[column].str.strip()
    keep = exploded[column].notna() & ~exploded[column].isin(["", UNKNOWN])
    # explode() repeats index labels; reset so crosstab/groupby alignment works
    return exploded[keep].reset_index(drop=True)

def _filter_type(df: pd.DataFrame, content_type: str | None) -> pd.DataFrame:
    return df if content_type is None else df[df["type"] == content_type]


# Catalog overview
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers describing the dataset."""
    metrics = {
        "Total titles": len(df),
        "Movies": int((df["type"] == "Movie").sum()),
        "TV Shows": int((df["type"] == "TV Show").sum()),
        "Unique countries": explode_multivalue(df, "country"["country"].nunique()),
        "Unique genres": explode_multivalue(df, "listed_in"["listed_in"].nunique()),
        "Unique directors": explode_multivalue(df, "director"["director"].nunique()),
        "Unique actors": explode_multivalue(df, "cast"["cast"].nunique()),
        "Earliest release year": int(df["release_year"].min()),
        "Latest release year": int(df["release_year"].max()),
        "First year added": int(df["year_added"].min()),
        "Last year added": int(df["year_added"].max()),
    }
    return pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})

def type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Movies vs TV Shows: count and percentage."""
    out = df["type"].value_counts().rename_axis("type").reset_index(name="count")
    out["percentage"] = (out["count"] / out["count"].sum() * 100).round(1)
    return out

# Time-based analysis
def titles_added_per_year(df: pd.DataFrame) -> pd.DataFrame:
    dated = df.dropna(subset=["year_added"])
    table = dated.groupby([dated["year_added"].astype(int),
"type"]).size().unstack(fill_value=0)
    table = table.reindex(columns=["Movie", "TV Show"], fill_value=0)
    # include years with zero additions so the time axis has no gasps
    table = table.reindex(range(table.index.min(), table.index.max() + 1), fill_value=0)
    table["Total"] = table.sum(axis=1)
    table.index.name = "year_added"
    return table

def monthly_addition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Year x month matrix of titles added - handy for spotting seasonality."""
    dated = df.dropna(subset=["year_added", "month_added"])
    matrix = pd.crosstab(dated["year_added", "month_added"])
    matrix = matrix.reindex(columns=range(1, 13), fill_value=0)
    matrix.columns = MONTH_ABBR
    matrix.index.name = "year_added"
    return matrix

def release_decade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """How old is the catalog? Titles per release decade, split by type."""  
    valid = df.dropna(subset=["release_year"])
    decade = (valid["release_year"].astype(int) // 10) * 10
    table = pd.crosstab(decade, valid["type"])
    table.index.name = "release_decade"
    return table