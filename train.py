from moodtox.config import get_config
from moodtox.trainer import run_grid_search


def main():
    run_grid_search(get_config())


if __name__ == "__main__":
    main()
