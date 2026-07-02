"""Module entrypoint for ``python -m tsam_workflows``."""

from __future__ import annotations

from tsam_workflows.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
