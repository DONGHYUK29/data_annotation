import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation labels")
    parser.add_argument("label_dir", type=Path, help="검사할 labels 디렉터리")
    return parser.parse_args()


def main() -> int:
    label_dir = parse_args().label_dir.expanduser().resolve()
    invalid_files = []
    invalid_lines = []
    total_lines = 0
    label_paths = sorted(label_dir.rglob("*.txt"))

    print("===== LABEL VALIDATION START =====")
    for label_path in label_paths:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            invalid_files.append((label_path, "empty_file"))

        for line_idx, line in enumerate(lines, start=1):
            total_lines += 1
            parts = line.strip().split()
            if len(parts) < 7:
                invalid_lines.append((label_path, line_idx, "too_short", line))
                continue
            try:
                int(float(parts[0]))
                coords = list(map(float, parts[1:]))
            except ValueError:
                invalid_lines.append((label_path, line_idx, "parse_error", line))
                continue
            if len(coords) % 2:
                invalid_lines.append((label_path, line_idx, "odd_coords", line))
            elif len(coords) < 6:
                invalid_lines.append((label_path, line_idx, "too_few_points", line))
            elif any(value < 0 or value > 1 for value in coords):
                invalid_lines.append((label_path, line_idx, "out_of_range", line))

    print(f"Total label files: {len(label_paths)}")
    print(f"Total lines: {total_lines}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Invalid lines: {len(invalid_lines)}")
    for item in invalid_files[:10]:
        print(item)
    for path, line_idx, reason, content in invalid_lines[:20]:
        print(f"{path} | line {line_idx} | {reason}\n  -> {content}")
    return 1 if invalid_files or invalid_lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
