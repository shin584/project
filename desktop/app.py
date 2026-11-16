"""Desktop shell composition. Computation stays behind the #178 boundary."""

from __future__ import annotations

import sys

from desktop.instance import SingleInstance


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("The desktop shell requires Windows and WebView2.")

    # Acquire before importing any UI or renderer, including on a second launch.
    with SingleInstance() as instance:
        if not instance.acquired:
            return 0

        import webview

        from bundle_runtime import is_packaged, resource_root
        from desktop.api import DesktopApi, Dispatcher, webview_emitter
        from desktop.fake import FakeRenderer
        from desktop.screen import SCREEN_HTML

        if is_packaged():
            webview.settings["WEBVIEW2_RUNTIME_PATH"] = str(resource_root() / "webview2")
        webview.settings["REMOTE_DEBUGGING_PORT"] = None
        window = webview.create_window(
            "CRISPR PAM Scanner + XAI",
            html=SCREEN_HTML,
            width=1280,
            height=840,
            min_size=(900, 600),
            text_select=True,
        )
        if window is None:
            raise RuntimeError("Could not create the desktop window.")
        dispatcher = Dispatcher(webview_emitter(window))
        api = DesktopApi(FakeRenderer(step_delay=0.15), dispatcher)
        window.expose(
            api.prepare, api.scan_sites, api.predict, api.explain, api.save_csv, api.cancel
        )
        try:
            # A load handler avoids a startup thread waiting after an early close.
            def start_preparation() -> None:
                if window.events.shown.wait(5) and not window.events.closed.is_set():
                    window.evaluate_js("window.xaiDesktop.start()")

            window.events.loaded += start_preparation
            webview.start(gui="edgechromium", http_server=False, debug=False)
        finally:
            dispatcher.close()
    return 0
