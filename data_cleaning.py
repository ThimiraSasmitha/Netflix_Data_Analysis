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
    df = _fill_missing_values(df)
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

