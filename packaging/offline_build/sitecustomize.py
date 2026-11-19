"""Block Python network calls in the offline installer and PyInstaller hooks."""

import sys


def block_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto"}:
        raise PermissionError(f"Offline build blocked {event}")


sys.addaudithook(block_network)
