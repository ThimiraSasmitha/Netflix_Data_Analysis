"""Load and persist Netflix datasets."""

from __future__ import annotations
 
import logging
from pathlib import Path
 
import pandas as pd
 
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, REQUIRED_COLUMNS