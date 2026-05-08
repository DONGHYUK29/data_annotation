"""중간 산출물 폴더 비우기."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean workspace subfolders")
    parser.add_argument(
        "--mode",
        default="dataset",
        choices=["dataset", "input", "stage1", "stage2", "training", "all"],
    )
    args = parser.parse_args(argv)

    w = cfg.WORKSPACE_ROOT
    by_mode: dict[str, list[Path]] = {
        "dataset": [
            w / "dataset/images",
            w / "dataset/labels",
            w / "dataset/masks",
            w / "dataset/rgbd",
        ],
        "input": [
            w / "input/images",
            w / "input/rgbd",
            ],
        "stage1": [
            w / "output_1/labels",
            w / "output_1/masks",
            w / "output_1/images",
        ],
        "stage2": [
            w / "output_2/images",
            w / "output_2/labels",
            w / "output_2/masks",
        ],
        "training": [
            w / "training/images",
            w / "training/labels",
            w / "training/dataset",
        ],
    }
    all_targets: list[Path] = []
    for k, v in by_mode.items():
        all_targets.extend(v)
    by_mode["all"] = all_targets

    print("WORKSPACE_ROOT =", w)
    for folder in by_mode[args.mode]:
        if not folder.exists():
            print("[skip]", folder)
            continue
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print("[clean]", folder)


if __name__ == "__main__":
    main()
