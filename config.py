"""
Central configuration for the Netflix data analysis project.

Everything that is a path, a constant or a styling choice lives here so the
other modules stay free of "magic values".
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Kaggle "Netflix Movies and TV Shows" dataset:
# https://www.kaggle.com/datasets/shivamb/netflix-shows
RAW_DATA_PATH = RAW_DIR / "netflix_titles.csv"
SAMPLE_DATA_PATH = RAW_DIR / "netflix_titles_sample.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "netflix_clean.csv"
INSIGHTS_PATH = OUTPUT_DIR / "insights.txt"

# Dataset schema
REQUIRED_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]

# Free-text columns that get whitespace-stripped during cleaning
TEXT_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "rating",
    "duration",
    "listed_in",
    "description",
]

# Columns holding comma-separated lists of values
MULTI_VALUE_COLUMNS = ["director", "cast", "country", "listed_in"]

# Placeholder used for missing categorical values
UNKNOWN = "Unknown"

# Domain knowledge
# Maps indivdual maturity ratings to broader audience groups
RATING_TO_GROUP = {
    # Kids
    "TV-Y": "Kids",
    "TV-G": "Kids",
    "G": "Kids",
    "TV-Y7": "Older Kids",
    "TV-Y7-FV": "Older Kids",
    "TV-PG": "Older Kids",
    "PG": "Older Kids",
    # Teens
    "TV-14": "Teens",
    "PG-13": "Teens",
    # Adults
    "TV-MA": "Adults",
    "R": "Adults",
    "NC-17": "Adults",
}
RATING_GROUP_ORDER = ["Kids", "Older Kids", "Teens", "Adults", "Unrated"]

MONTH_ABBR = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
MONTH_FULL = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
# Analysis / modelling settings
TOP_N = 10
RANDOM_SEED = 42

# Plot styling
NETFLIX_RED = "#E50914"
NETFLIX_BLACK = "#141414"
NETFLIX_GRAY = "#8C8C8C"
TYPE_COLORS = {"Movie": NETFLIX_RED, "TV Show": "#564D4D"}

def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
