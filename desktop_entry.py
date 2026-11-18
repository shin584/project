"""PyInstaller entry point for the production desktop and its gate-2 check."""

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    from desktop.cli import main

    raise SystemExit(main())
