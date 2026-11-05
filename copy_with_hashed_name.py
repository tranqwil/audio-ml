#!/usr/bin/env python3
"""Recursively copy every file under a source directory into a flat target
directory, giving each copy a randomly hashed file name.

Usage:
    python copy_with_hashed_names.py SOURCE TARGET
"""

import hashlib
import os
import secrets
import shutil
import sys
import time
from pathlib import Path


def random_hash_name() -> str:
    """Return a random SHA-256 hex digest (derived from 32 random bytes)."""
    return hashlib.sha256(secrets.token_bytes(32)).hexdigest()


def unique_dest(target: Path, ext: str) -> Path:
    """Pick a hashed destination path that doesn't already exist."""
    while True:
        candidate = target / f"{random_hash_name()}{ext}"
        if not candidate.exists():
            return candidate


def iter_files(source: Path, target: Path):
    """Yield every regular file under source, skipping symlinks and the target dir."""
    for dirpath, dirnames, filenames in os.walk(source):
        current = Path(dirpath)

        # Don't descend into the target if it lives inside the source tree.
        dirnames[:] = [d for d in dirnames if (current / d).resolve() != target]

        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python copy_with_hashed_names.py SOURCE TARGET")

    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()

    if not source.is_dir():
        sys.exit(f"Source is not a directory: {source}")

    target.mkdir(parents=True, exist_ok=True)
    print(f"Source: {source}")
    print(f"Target: {target}")

    print("Scanning source directory...")
    total = sum(1 for _ in iter_files(source, target))
    print(f"Found {total} file(s) to copy")
    if total == 0:
        return

    copied = failed = 0
    start = time.monotonic()

    for index, src_file in enumerate(iter_files(source, target), start=1):
        dest = unique_dest(target, src_file.suffix)

        try:
            shutil.copy2(src_file, dest)
        except OSError as e:
            failed += 1
            print(f"FAILED {src_file}: {e}")
            continue

        copied += 1
        elapsed = time.monotonic() - start
        eta = elapsed / index * (total - index)
        print(
            f"[{index}/{total} {index / total * 100:5.1f}% | ETA {format_duration(eta)}] "
            f"{src_file} -> {dest.name}"
        )

    print(f"Done in {format_duration(time.monotonic() - start)}: "
          f"{copied} copied, {failed} failed, {total} total")


if __name__ == "__main__":
    main()