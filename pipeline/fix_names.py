"""input 폴더에서 class_ 접두사 제거 (예: class_2_8 → 2_8)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Strip class_ prefix from input filenames")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"default: {cfg.IMAGES_RAW_DIR}",
    )
    args = parser.parse_args(argv)

    input_dir = args.dir or cfg.IMAGES_RAW_DIR
    count = 0

    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        stem, suffix = p.stem, p.suffix
        if not stem.startswith("class_"):
            continue
        new_stem = stem.replace("class_", "", 1)
        new_name = new_stem + suffix
        new_path = p.with_name(new_name)
        if new_path.exists():
            print(f"SKIP (already exists): {new_name}")
            continue
        p.rename(new_path)
        print(f"{p.name}  →  {new_name}")
        count += 1

    print("\nRenamed:", count)


if __name__ == "__main__":
    main()
