#!/usr/bin/env python3
"""Run a command and sample aggregate RSS for its process tree."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def process_table() -> tuple[dict[int, int], dict[int, int]]:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parents[pid] = int(fields[3])
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss[pid] = int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return parents, rss


def descendants(root: int, parents: dict[int, int]) -> set[int]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in result and parent in result:
                result.add(pid)
                changed = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    process = subprocess.Popen(command, start_new_session=True)
    peak = 0
    samples = 0
    started = time.monotonic()
    try:
        while True:
            parents, rss = process_table()
            pids = descendants(process.pid, parents)
            peak = max(peak, sum(rss.get(pid, 0) for pid in pids))
            samples += 1
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(args.interval)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "peak_memory_bytes": peak,
                "samples": samples,
                "elapsed_seconds": time.monotonic() - started,
                "command": command,
                "exit_code": returncode,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
