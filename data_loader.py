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
            f"Please download the Netflix dataset from Kaggle and place it in the "
            f"data/raw directory."
            f"See README.md for more details.")
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    validate_columns(df)
    return df
