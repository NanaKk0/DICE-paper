#!/usr/bin/env python3
"""Inspect/reconstruct exact duplicate paths in the optional DICE asset packs.

Offline and CPU-only. Does not infer, train, simulate or replace existing files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def safe(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    result = root / rel
    # Refuse any symlink in the path, even one currently resolving inside root.
    for p in [result, *result.parents]:
        if p == root.parent:
            break
        if p.is_symlink():
            raise ValueError(f"Symlink in archive path: {relative}")
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"Archive path escapes root: {relative}")
    return result


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inspect(root, *, restore=False, verify=False, require_all=False):
    root = Path(root).expanduser().resolve(strict=True)
    with (root / "FILE_INDEX.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Duplicate logical path in FILE_INDEX")
    checked = {}
    pending = []
    missing = {}
    for row in rows:
        source = safe(root, row["canonical_file"])
        target = safe(root, row["file"])
        expected = (int(row["bytes"]), row["sha256"])
        if source not in checked:
            if not source.is_file():
                missing.setdefault(row["package"], set()).add(row["canonical_file"])
                continue
            if source.stat().st_size != expected[0] or sha(source) != expected[1]:
                raise ValueError(f"Canonical file integrity mismatch: {row['canonical_file']}")
            checked[source] = expected
        elif checked[source] != expected:
            raise ValueError("Conflicting canonical identity in FILE_INDEX")
        if target.exists():
            if not target.is_file() or target.stat().st_size != expected[0] or ((verify or restore) and sha(target) != expected[1]):
                raise ValueError(f"Existing logical path differs; refusing replacement: {row['file']}")
        elif target != source:
            pending.append((source, target))
    result = dict(canonical_files_present=len(checked), logical_files=len(rows),
                  aliases_available=len(pending), aliases_restored=0,
                  missing_packages={k: len(v) for k, v in sorted(missing.items())},
                  missing_aliases_are_optional=not require_all,
                  scientific_calculations=False)
    # Fail before any write on incomplete strict downloads or corrupt files.
    if missing and require_all:
        raise ValueError("Required packages missing: " + json.dumps(result["missing_packages"]))
    if restore:
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out, source.open("rb") as inp:
                shutil.copyfileobj(inp, out)
            result["aliases_restored"] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted research-assets directory")
    parser.add_argument("--restore", action="store_true", help="Copy only missing exact duplicate paths")
    parser.add_argument("--verify", action="store_true", help="Also hash existing alias paths")
    parser.add_argument("--require-all", action="store_true", help="Require every indexed canonical payload")
    args = parser.parse_args()
    print(json.dumps(inspect(args.root, restore=args.restore, verify=args.verify,
                             require_all=args.require_all), indent=2))


if __name__ == "__main__":
    main()
