"""Dashboard v3 window composition; computation lives in a spawned child."""

from __future__ import annotations

import sys

from desktop.instance import SingleInstance


def main(*, preview: bool = False) -> int:
    if sys.platform != "win32":
        raise RuntimeError("The desktop shell requires Windows and WebView2.")

    # Acquire before importing any UI or renderer, including on a second launch.
    with SingleInstance() as instance:
        if not instance.acquired:
            return 0

        import webview
        from bundle_runtime import is_packaged, resource_root

        from desktop.api import DesktopApi, webview_emitter
        from desktop.computation import ProcessDispatcher
        from desktop.screen import SCREEN_HTML

        if is_packaged():
            webview.settings["WEBVIEW2_RUNTIME_PATH"] = str(
                resource_root() / "webview2"
            )
        webview.settings["REMOTE_DEBUGGING_PORT"] = None
        window = webview.create_window(
            "CRISPR PAM Scanner + XAI",
            html=SCREEN_HTML.replace('<body>', '<body data-preview="true">')
            if preview
            else SCREEN_HTML.replace('<p class="demo">', '<p class="demo" hidden>'),
            width=1280,
            height=840,
            min_size=(900, 600),
            text_select=True,
        )
        if window is None:
            raise RuntimeError("Could not create the desktop window.")
        def ask_save_path(filename: str) -> str | None:
            """The OS save dialog, standing in for the browser download (#185).

            pywebview answers with a path, a one-item sequence, or nothing when
            the dialog is closed; the boundary wants a path or None.
            """
            chosen = window.create_file_dialog(
                webview.SAVE_DIALOG, save_filename=filename
            )
            if not chosen:
                return None
            return chosen if isinstance(chosen, str) else chosen[0]

        dispatcher = ProcessDispatcher(webview_emitter(window), preview=preview)
        api = DesktopApi(dispatcher.renderer, dispatcher, ask_save_path=ask_save_path)
        window.expose(
            api.prepare,
            api.select_family,
            api.scan_sites,
            api.predict,
            api.explain,
            api.save_context,
            api.save_csv,
            api.cancel,
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
