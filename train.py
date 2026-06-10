from __future__ import annotations

import argparse

from moodtox.config import get_config
from moodtox.trainer import run_experiment, run_grid_search


def main():
    parser = argparse.ArgumentParser(description="Train MoodTOX")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run the hyperparameter grid defined in moodtox/config.py",
    )
    args = parser.parse_args()
    config = get_config()
    if args.grid_search or config.grid_search.enabled:
        run_grid_search(config)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
