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

def release_lag_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistics on the gap (years) between release and arrival on Netflix."""
    lag = df["release_lag_years"].dropna().astype(float)
    return lag.describe().round(2).rename_axis("statistic").reset_index(name="years")

# Who? / Where? / What?
def top_values(
    df: pd.DataFrame,
    column: str,
    n: int = TOP_N,
    content_type: str | None = None,
) -> pd.DataFrame:
    """Most frequent values of a (possibly comma-separated) column."""  
    subset = _filter_type(df, content_type)
    counts = explode_multivalue(subset. column)[column].value_counts().head(n)
    return counts.rename_axis(column).reset_index(name="count")

def country_content_mix(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Movies vs TV Shows for the n countries that produce the most titles."""
    exploded = explode_multivalue(df, "country")
    top = exploded["country"].value_counts().head(n).index
    subset = exploded[exploded["country"].isin(top)]
    mix = pd.crosstab(subset["country"], subset["type"]).loc[list(top)]
    mix = mix.reindex(columns=["Movies", "TV Show"], fill_value=0)
    return mix

def genre_trend_over_time(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Titles added per year for the n most common genres."""
    exploded = explode_multivalue(df.dropna(subset=["year_added"]), "listed_in")
    top = list(exploded["listed_in"].value_counts().head(n).index)
    subset = exploded[exploded["listed_in"].isin(top)]
    trend = pd.crosstab(subset["year_added"].astype(int), subset["listed_in"])[top]
    trend.index.name = "year_added"
    return trend

def genre_cooccurrence(
    df: pd.DataFrame, n: int = 12, content_type: str | None = "Movie"
) -> pd.DataFrame:
    """
    Symmetric matrix counting how often two genres are listed on the same
    title. The diagonal holds each genre's total number of titles.
 
    Movie and TV genres never overlap, so the matrix is computed for a single
    content type (movies by default).
    """
    exploded = explode_multivalue(_filter_type(df, content_type), "listed_in")
    top = list(exploded["listed_in"].value_counts().head(n).index)
    presence = pd.crosstab(exploded["show_id"], exploded["listed_in"])[top].gt(0).astype(int)
    return presence.T.dot(presence)

# Ratings and durations
def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count of titles per rating (TV-MA, PG-13, ...), split by type."""
    table = pd.crosstab(df["rating"], df["type"])
    table["Total"] = table.sum(axis=1)
    return table.sort_values("Total", ascending=False)

def rating_group_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Same as above but with ratings collapsed into audience groups."""
    table = pd.crosstab(df["rating_group"], df["type"])
    return table.reindex(RATING_GROUP_ORDER, fill_value=0)

def movie_duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of movie runtimes in minutes."""
    minutes = df.loc[df["type"] == "Movie", "duration_minutes"].dropna().astype(float)
    return minutes.describe().round(1).rename_axis("statistic").reset_index(name="minutes")

def tv_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of the number of seasons of TV shows."""
    seasons = df.loc[df["type"] == "TV Show", "num_seasons"].dropna().astype(float)
    return seasons.describe().round(1).rename_axis("statistic").reset_index(name="seasons")

# Orchestration
def run_all_analyses(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every analysis and return the results keyed by table name."""    
    logger.info("Running analyses...")
    return{
        "overview": dataset_overview(df),
        "type_distribution": type_distribution(df),
        "titles_added_per_year": titles_added_per_year(df),
        "monthly_additions": monthly_addition_matrix(df),
        "release_decades": release_decade_distribution(df),
        "release_lag_summary": release_lag_summary(df),
        "top_countries": top_values(df, "country"),
        "country_mix": country_content_mix(df),
        "top_genres": top_values(df, "listed_in"),
        "top_genres_movies": top_values(df, "listed_in", content_type="Movie"),
        "top_genres_tv": top_values(df, "listed_in", content_type="TV Show"),
        "top_directors": top_values(df, "director"),
        "top_actors": top_values(df, "cast"),
        "genre_trends": genre_trend_over_time(df),
        "genre_cooccurrence_movies": genre_cooccurrence(df, content_type="Movie"),
        "rating_distribution": rating_distribution(df),
        "rating_groups": rating_group_distribution(df),
        "movie_duration_summary": movie_duration_summary(df),
        "tv_season_summary": tv_season_summary(df),
    }

def save_tables(tables: dict[str, pd.DataFrame], directory: Path = TABLES_DIR) -> None:
    """Export every result table to CSV."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        keep_index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(directory / f"{name}.csv", index=keep_index)
    logger.info("Saved %s tables to %s", len(tables), directory)

# Plain-English insights
def generate_insights(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Turn the result tables into a list of human-readable findings."""
    insights: list[str] = []

    # Catalog composition
    dist = tables["type_distribution"].set_index("type")
    movie_pct = dist["percentage"].get("Movie", 0.0)
    show_pct = dist["percentage"].get("TV Show", 0.0)
    insights.append(
        f"The catalog holds {len(df):,} titles: {movie_pct:.1f}% movies and {show_pct:.1f}% TV shows."
    )

    # Growth
    yearly = tables["titles_added_per_year"]
    if not yearly.empty:
        peak_year = int(yearly["Total"].idxmax())
        insights.append(
            f"The busiest year for new additions was {peak_year}, with {int(yearly['Total'].max()):,} titles added."
        )

    # Seasonality
    monthly = tables["monthly_additions"]
    if not monthly.empty:
        month_totals = monthly.sum()
        insights.append(
            f"{month_totals.idxmax()} is the most popular month for adding content "
            f"({int(month_totals.max()):,} titles over all years); "
            f"{month_totals.idxmin()} is the quietest ({int(month_totals.min()):,})."
        )

    # Geography
    countries = tables["top_countries"]
    if not countries.empty:
        top = countries.iloc[0]
        share = top["count"] / len(df) * 100
        insights.append(
            f"{top['country']} leads production with {int(top['count']):,} titles "
            f"({share:.1f}% of the catalog)."
        )

    