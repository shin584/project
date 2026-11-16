"""PyInstaller entry point for the production desktop and its gate-2 check."""

from desktop.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
