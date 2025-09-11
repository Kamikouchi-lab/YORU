import sys

sys.path.append("../yoru")

# yoru/__main__.py
from yoru.cli import main as cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())

