#!/usr/bin/env python3
"""
Recursively search a directory for audio files with a given sample rate
(default 96000 Hz) and copy them to a destination folder.

Requires: pip install mutagen
(WAV files also fall back to Python's built-in `wave` module.)

Examples:
    python copy_by_samplerate.py ~/Music ~/Music_96k
    python copy_by_samplerate.py ~/Music ~/Music_96k --rate 48000 --flat
    python copy_by_samplerate.py ~/Music ~/Music_96k --dry-run
"""

import argparse
import shutil
import sys
import wave
from pathlib import Path

try:
    import mutagen
except ImportError:
    mutagen = None

AUDIO_EXTS = {
    ".wav", ".wave", ".flac", ".aif", ".aiff", ".aifc", ".mp3", ".m4a",
    ".aac", ".ogg", ".oga", ".opus", ".wma", ".wv", ".ape", ".alac", ".mp4",
}


def get_sample_rate(path: Path):
    """Return the sample rate of an audio file, or None if it can't be read."""
    if mutagen is not None:
        try:
            audio = mutagen.File(path)
            if audio is not None and getattr(audio, "info", None):
                rate = getattr(audio.info, "sample_rate", None)
                if rate:
                    return int(rate)
        except Exception:
            pass

    if path.suffix.lower() in {".wav", ".wave"}:
        try:
            with wave.open(str(path), "rb") as w:
                return w.getframerate()
        except Exception:
            pass

    return None


def unique_path(dest: Path) -> Path:
    """Avoid overwriting: file.wav -> file_1.wav, file_2.wav, ..."""
    if not dest.exists():
        return dest
    i = 1
    while True:
        candidate = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="Directory to search (recursively)")
    p.add_argument("dest", type=Path, help="Directory to copy matching files into")
    p.add_argument("--rate", type=int, default=96000, help="Sample rate to match (default: 96000)")
    p.add_argument("--flat", action="store_true",
                   help="Copy everything into dest directly instead of preserving subfolders")
    p.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = p.parse_args()

    if mutagen is None:
        print("Warning: mutagen not installed; only .wav files can be read. "
              "Run: pip install mutagen", file=sys.stderr)

    src = args.source.expanduser().resolve()
    dst = args.dest.expanduser().resolve()

    if not src.is_dir():
        sys.exit(f"Source is not a directory: {src}")

    scanned = copied = skipped = 0

    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        # Don't re-scan files we've already copied if dest is inside source
        if dst == path.parent or dst in path.parents:
            continue

        scanned += 1
        rate = get_sample_rate(path)
        if rate is None:
            print(f"[skip] could not read: {path}")
            skipped += 1
            continue
        if rate != args.rate:
            continue

        target = dst / path.name if args.flat else dst / path.relative_to(src)
        target = unique_path(target)

        print(f"[{'would copy' if args.dry_run else 'copy'}] {path} -> {target}  ({rate} Hz)")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        copied += 1

    verb = "would be copied" if args.dry_run else "copied"
    print(f"\nScanned {scanned} audio files: {copied} {verb}, {skipped} unreadable.")


if __name__ == "__main__":
    main()