"""Portable import boundary for the unchanged public paper-output CLI."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'mc_toolkit/operations/paper_outputs'),
                str(ROOT / 'mc_toolkit/operations/fresh_test'),
                str(ROOT / 'mc_toolkit/operations/fresh_test/source')]

if __name__ == '__main__':
    import cli
    cli.DEFAULT_FREEZE = ROOT / "reproduction/configs/frozen"
    raise SystemExit(cli.main())
