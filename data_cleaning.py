"""
Data cleaning and feature engineering for the Netflix titles dataset.
 
Each private helper does one job and returns a new dataframe, so the pipeline
in `clean_data` reads top-to-bottom like a recipe.
"""

from __future__ import annotations

import logging
import pandas as pd
from config import RATING_TO_GROUP, TEXT_COLUMNS, UNKNOWN

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a tidy dataframe."""
    rows_before = len(df)

    df = df.copy()
    df = _strip_text_columns(df)
    df = _fix_misplaced_durations(df)
    df = _drop_invalid_and_duplicate_rows(df)
    df = _parse_dates(df)
    df = _parse_duration(df)
    df = _fill_missing_values(df)
    df = _add_rating_group(df)
    df = _add_helper_features(df)

    logger.info("Cleaning finished: %s -> %s rows", rows_before, len(df))
    return df.reset_index(drop=True)

def _strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and turn empty strings into proper missing values."""
    for col in TEXT_COLUMNS:
       if col in df.columns:
           df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df

def _fix_misplaced_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    A handful of rows in the public dataset have the runtime (e.g. '74 min')
    stored in the `rating` column and an empty `duration`. Move it back.
    """
    looks_like_duration = df["rating"].str.match(r"^\d+\s+min$", case=False, na=False)
    mask = looks_like_duration & df["duration"].isna()

    if mask.any():
        logger.info("Fixing %s rows with the duration stored in `rating`", mask.sum())
        df.loc[mask, "duration"] = df.loc[mask, "rating"]
        df.loc[mask, "rating"] = pd.NA
    return df

def _drop_invalid_and_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without a title/type and any duplicated titles."""
    before = len(df)
    df = df.dropna(subset=["title", "type"])
    df = df.drop_duplicates(subset="show_id")
    df = df.drop_duplicates(subset=["title", "type", "release_year"])

    removed = before - len(df)
    if removed:
        logger.info("Dropped %s invalid/duplicate rows", removed)
    return df

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `date_added` and derive calendar features plus the release lag."""
    df["date_added"] = pd.to_datetime(df["date_added"], format="mixed", errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
 
    df["year_added"] = df["date_added"].dt.year.astype("Int64")
    df["month_added"] = df["date_added"].dt.month.astype("Int64")
    df["month_name_added"] = df["date_added"].dt.month_name()
    df["weekday_added"] = df["date_added"].dt.day_name()
 
    lag = df["year_added"] - df["release_year"]
    df["release_lag_years"] = lag.where(lag >= 0)
    return df

def _parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the text `duration` column ('90 min' / '3 Seasons') into two numeric
    columns: `duration_minutes` for movies and `num_seasons` for TV shows.
    """
    number = pd.to_numeric(
        df["duration"].str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    is_movie = df["type"] == "Movie"
    is_show = df["type"] == "TV Show"

    df["duration_minutes"] = number.where(is_movie)
    df["num_seasons"] = number.where(is_show)
    return df

def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing categorical values with an explicit 'Unknown' label."""
    for col in ["director", "cast", "country", "rating", "listed_in"]:
        df[col] = df[col].fillna(UNKNOWN)
    df["description"] = df["description"].fillna("")
    return df


def _add_rating_group(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the many TV/MPAA ratings into broader audience groups."""
    df["rating_group"] = (
        df["rating"].str.upper().map(RATING_TO_GROUP).fillna("Unrated")
    )
    return df


def _add_helper_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few convenience columns used by the analysis and recommender."""
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    df["num_genres"] = df["listed_in"].str.count(",").add(1).where(df["listed_in"] != UNKNOWN, 0)
    df["description_length"] = df["description"].str.split().str.len()
    return df
