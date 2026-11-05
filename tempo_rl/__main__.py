"""``python -m tempo_rl ...``: the same command-line interface as ``tempo-drl ...``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(prog="python -m tempo_rl"))
