"""Observe process RSS externally without modifying the frozen computation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = psutil.Process(args.pid)
    created = root.create_time()
    with args.output.open("x", encoding="utf-8") as stream:
        while root.is_running() and root.create_time() == created:
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            for process in processes:
                try:
                    memory = process.memory_info()
                    row = {"observed_at": stamp, "pid": process.pid, "ppid": process.ppid(),
                           "created_at_epoch": process.create_time(), "command": process.cmdline(),
                           "executable": process.exe(), "rss_bytes": memory.rss,
                           "peak_working_set_bytes": getattr(memory, "peak_wset", None)}
                    stream.write(json.dumps(row) + "\n")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            stream.flush()
            time.sleep(0.25)


if __name__ == "__main__":
    main()
