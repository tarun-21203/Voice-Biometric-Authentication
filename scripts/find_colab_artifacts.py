#!/usr/bin/env python3
"""Scan repository files for cloud-notebook / drive-mount artifacts and optionally remove them.

Usage:
    python scripts/find_colab_artifacts.py --report
    python scripts/find_colab_artifacts.py --fix  # makes backups (.bak) before editing

This tool looks for common cloud-notebook mount calls and repository paths created
when working in hosted notebook environments. To avoid the scanner reporting
itself, this script does not embed the exact search strings verbatim in its
source; the patterns are constructed at runtime instead.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    # constructed from smaller pieces so the raw literal does not appear in this file
    'from ' + 'google' + '.colab',
    'google' + '.colab',
    'drive' + '.' + 'mount' + '(',
    '/' + 'content' + '/' + 'drive',
    '/' + 'content' + '/' + 'gdrive',
]


def scan_text_file(path: Path) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if any(p in line for p in PATTERNS):
                    hits.append((i, line.rstrip("\n")))
    except Exception:
        pass
    return hits


def scan_ipynb(path: Path) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        cells = data.get("cells", [])
        for ci, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            for li, line in enumerate(src, start=1):
                if any(p in line for p in PATTERNS):
                    hits.append((ci + 1, line.rstrip("\n")))
    except Exception:
        pass
    return hits


def should_skip(dirpath: Path) -> bool:
    # skip virtual envs and git dir
    parts = set(dirpath.parts)
    skip = {".git", "venv", ".venv", "env", "__pycache__", ".ipynb_checkpoints"}
    return not skip.isdisjoint(parts)


def scan_repo() -> Dict[str, List[Tuple[int, str]]]:
    results: Dict[str, List[Tuple[int, str]]] = {}
    for root, dirs, files in os.walk(ROOT):
        rootp = Path(root)
        if should_skip(rootp):
            continue
        for fname in files:
            fp = rootp / fname
            if fp.suffix.lower() in {".py", ".md", ".txt", ".csv", ".json"}:
                hits = scan_text_file(fp)
                if hits:
                    results[str(fp.relative_to(ROOT))] = hits
            elif fp.suffix.lower() == ".ipynb":
                hits = scan_ipynb(fp)
                if hits:
                    results[str(fp.relative_to(ROOT))] = hits
    return results


def fix_text_file(path: Path) -> None:
    # Create a backup and write a cleaned file back to the original path
    backup = path.with_name(path.name + ".bak")
    try:
        if not backup.exists():
            path.replace(backup)
        # read from backup (original) and write cleaned content to original path
        with backup.open("r", encoding="utf-8", errors="replace") as src:
            lines = src.readlines()
        with path.open("w", encoding="utf-8") as out:
            for line in lines:
                if any(p in line for p in PATTERNS):
                    out.write("# REMOVED BY cleanup: " + line)
                else:
                    out.write(line)
    except Exception:
        pass


def fix_ipynb(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    try:
        if not backup.exists():
            path.replace(backup)
        with backup.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        changed = False
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            new_src = []
            for line in src:
                if any(p in line for p in PATTERNS):
                    new_src.append("# REMOVED BY cleanup: " + line)
                    changed = True
                else:
                    new_src.append(line)
            cell["source"] = new_src
        if changed:
            with path.open("w", encoding="utf-8") as out:
                json.dump(data, out, indent=1, ensure_ascii=False)
    except Exception:
        pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Edit files in-place (creates .bak backups)")
    args = parser.parse_args()

    print(f"Scanning repository: {ROOT}")
    results = scan_repo()
    if not results:
        print("No Colab/GDrive artifacts found.")
        return

    total = 0
    for path, hits in sorted(results.items()):
        print(f"\n{path}")
        for loc, snippet in hits[:10]:
            print(f"  {loc}: {snippet}")
        if len(hits) > 10:
            print(f"  ... {len(hits)-10} more hits ...")
        total += len(hits)

    print(f"\nFound {total} Colab/GDrive artifact lines across {len(results)} files.")

    if args.fix:
        print("\nFixing: creating backups with .bak and commenting out offending lines...")
        for path in results:
            fp = ROOT / path
            if fp.suffix.lower() == ".ipynb":
                fix_ipynb(fp)
            else:
                fix_text_file(fp)
        print("Fix complete. Backups have suffix .bak")


if __name__ == "__main__":
    main()
