"""Render saved live DOM snapshots with Edge for the gate-2 visual checklist.

These images are re-renderings of WebView's exported DOM, not native window
captures. No server is created; scripts and external requests are disabled.
Requires playwright on the review PC, never on the demonstration PC.
"""

import argparse
from pathlib import Path


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, java_script_enabled=False)
        page.route("**/*", lambda route: route.abort())
        for path in sorted(args.evidence.glob("*.html")):
            page.set_content(path.read_text(encoding="utf-8"), wait_until="load")
            page.screenshot(path=str(path.with_suffix(".png")), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
