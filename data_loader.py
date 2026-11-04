"""Load and persist Netflix datasets."""

from __future__ import annotations
 
import logging
from pathlib import Path
 
import pandas as pd
 
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that the DataFrame has the required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")