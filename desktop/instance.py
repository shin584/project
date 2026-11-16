"""Windows session-wide instance guard; no socket or persistent lock file."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from types import TracebackType


class SingleInstance:
    """Keep the named kernel object alive until the application exits.

    Windows closes handles even after a crash, so a killed process cannot leave
    a stale lock. Development and packaged launches share the same name.
    """

    def __init__(self, name: str = r"Local\XaiDashboardDesktop") -> None:
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        self._kernel.CreateMutexW.restype = wintypes.HANDLE
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel.CreateMutexW(None, False, name)
        error = ctypes.get_last_error()
        if not self._handle:
            raise ctypes.WinError(error)
        self.acquired = error != 183  # ERROR_ALREADY_EXISTS
        if not self.acquired:
            self.close()

    def close(self) -> None:
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> SingleInstance:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
