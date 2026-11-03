"""
Netflix data analysis - end-to-end pipeline.
 
    load -> clean -> analyse -> export tables -> plot -> (optional) recommend
 
Examples
--------
    python main.py --sample                       # try it on synthetic demo data
    python main.py                                # use data/raw/netflix_titles.csv
    python main.py --data path/to/file.csv        # use a custom file
    python main.py --recommend "Stranger Things"  # also print similar titles
    python main.py --skip-plots                   # tables and insights only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis import generate_insights, run_all_analyses, save_tables
from config import INSIGHTS_PATH, RAW_DATA_PATH, SAMPLE_DATA_PATH, ensure_directories
from data_cleaning import clean_data
from data_loader import load_raw_data, save_processed_data
from generate_sample_data import generate_sample_data
from recommender import ContentRecommender
from visualizations import create_all_plots

logger = logging.getLogger("netflix")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyse the Netflix Movie and TV Show dataset and generate insights, tables, and visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        )
    parser.add_argument(
        "--data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to netflix_titles.csv (default: %(default)s)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="generate and use synthetic demo data instead of the real CSV"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="number of rows for --sample (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="skip chart generation",
    )

    parser.add_argument(
        "--recommend",
        metavar="TITLE",
        help="print titles similar to TITLE"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of recommendations to show (default: %(default)s)"
    )
    return parser.parse_args()

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def write_insights(insights: list[str], path: Path = INSIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"- {line}" for line in insights) + "\n", encoding="utf-8")

def main() -> int:
    args = parse_args()
    setup_logging()
    ensure_directories()

    # 1. Load
    data_path = args.data
    if args.sample:
        logger.warning("Using SYNTHETIC data - results are not real Netflix insights.")
        generate_sample_data(n_rows=args.sample_size, path=SAMPLE_DATA_PATH)
        data_path = SAMPLE_DATA_PATH

    try:
        raw = load_raw_data(data_path)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1

    # 2. Clean
    df = clean_data(raw)
    save_processed_data(df)

    # 3. Analyse
    tables = run_all_analyses(df)
    save_tables(tables)

    insights = generate_insights(df, tables)
    write_insights(insights)
    print("\n=== Key insights ===")
    for line in insights:
        print(f"- {line}")

    # 4. Visualize
    if not args.skip_plots:
        create_all_plots(df, tables)

    # 5. Recommendations
    if args.recommend:
        recommender = ContentRecommender(df).fit()
        try:
            recs = recommender.recommend(args.recommend, n=args.top)
        except ValueError as err:
            logger.error("%s", err)
            return 1
        print(f"\n=== Because you watched '{args.recommend}' ===")
        print(recs.to_string(index=False))

    print("\nDone. Outputs are in the 'outputs/' folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
