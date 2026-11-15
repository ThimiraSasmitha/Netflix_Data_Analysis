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