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

