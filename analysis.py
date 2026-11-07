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

