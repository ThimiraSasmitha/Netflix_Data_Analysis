"""Load and persist Netflix datasets."""

from __future__ import annotations
 
import logging
from pathlib import Path
 
import pandas as pd
 
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

def validate_columns(df: pd.DataFrame) -> None:
    """Raise a helptful error if the dataframe lacks any expected column."""
    missing = [col  for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
            )

def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Netflix CSV and check that its schema is what we expect."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}.\n"
            "Download 'netflix_titles.csv' from "
            "https://www.kaggle.com/datasets/shivamb/netflix-shows "
            "and place it in data/raw/, or run `python main.py --sample` "
            "to use synthetic demo data."
        )

    df = pd.read_csv(path)
    validate_columns(df)
    logger.info(f"Loaded %s rows x %s columns from %s", *df.shape, path)
    return df

def save_processed_data(df: pd.DataFrame, path: str | Path = PROCESSED_DATA_PATH) -> Path:
    """Write the cleaned dataframe to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data (%s rows) to %s", len(df), path)
    return path

