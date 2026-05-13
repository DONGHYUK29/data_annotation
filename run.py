"""
통합 진입점. 프로젝트 루트에서 실행하세요.

  python run.py segment [--input DIR] [--output DIR]
  python run.py gui [--host 0.0.0.0] [--port 7860]
  python run.py export --bg paper
  python run.py build --num-classes 10 [--val-ratio 0.1]
  python run.py train --weights yolo26l-seg.yaml --name exp1 --epochs 100 --batch 16 --imgsz 640 [...]
  python run.py clean --mode dataset|input|output1|output2|training|all
  python run.py count [--dir PATH]
  python run.py fix-names [--dir PATH]
  python run.py trim stage --dir create/output_1 --keep 200
  python run.py trim dataset [--dir PATH]

입력 PNG는 create/input/images 에 직접 두면 됩니다.
경로·가중치는 config.py 만 수정하면 됩니다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELP = __doc__ or ""


def run_web_gui(argv: list[str]) -> None:
    try:
        import uvicorn
        import config as cfg

        parser = argparse.ArgumentParser(prog="python run.py gui")
        parser.add_argument("--host", default=cfg.WEB_HOST, help="Web GUI host")
        parser.add_argument("--port", type=int, default=cfg.WEB_PORT, help="Web GUI port")
        args = parser.parse_args(argv)

        uvicorn.run("pipeline.gui_app:app", host=args.host, port=args.port, reload=False)
    except ModuleNotFoundError:
        cmd = ["docker", "compose", "up", "annotation"]
        print("로컬 의존성이 없어 Docker로 GUI를 실행합니다:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=False)


def fix_labels_for_segmentation(yaml_path: Path) -> None:
    """학습 직전 라벨 정리: bbox 접두가 있으면 제거."""
    from pipeline.label_utils import normalize_label_text

    dataset_dir = yaml_path.parent
    fixed_count = 0

    labels_root = dataset_dir / "labels"
    if not labels_root.is_dir():
        return

    for label_file in labels_root.rglob("*.txt"):
        try:
            raw = label_file.read_text(encoding="utf-8")
            new_text = normalize_label_text(raw)
            if new_text != raw:
                label_file.write_text(new_text, encoding="utf-8")
                fixed_count += 1
        except OSError:
            pass

    if fixed_count > 0:
        print(
            f"🛠️ [자동 수정 완료] {fixed_count}개 라벨에서 bbox 접두를 제거했습니다 (세그 포맷)."
        )


def run_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    import config as cfg
    from pipeline.train_args import resolve_weights_path, train_kwargs_from_namespace

    if args.project is None:
        args.project = cfg.WEIGHTS_DIR

    # Train 전에 항상 dataset -> training split/build를 수행한다.
    from pipeline.build_split import main as build_main

    build_main(
        [
            "--num-classes",
            str(args.num_classes),
            "--val-ratio",
            str(args.val_ratio),
        ]
    )

    if not args.data:
        # build_split 이 생성하는 yaml (create/training/dataset.yaml)
        args.data = str(cfg.TRAINING_DATASET_YAML)

    yaml_path = Path(args.data)
    if not yaml_path.is_file():
        print(f"❌ [에러] dataset.yaml 을 찾을 수 없습니다: {yaml_path}")
        print("   먼저 build(Train/Val 분할)를 실행해 주세요.")
        sys.exit(1)

    fix_labels_for_segmentation(yaml_path)

    model_source = resolve_weights_path(args.weights, cfg.WEIGHTS_DIR)
    model = YOLO(str(model_source))

    kwargs = train_kwargs_from_namespace(args)
    if str(model_source).lower().endswith((".yaml", ".yml")):
        # yaml로 학습 시작 시 ultralytics가 기본 pt를 내려받아 생성하는 것을 방지
        kwargs["pretrained"] = False
    print("=" * 60)
    print("🚀 YOLO 학습 시작")
    print(f"   - model: {model_source}")
    print(f"   - data:  {args.data}")
    print(f"   - project: {args.project}")
    print(f"   - name: {args.name}")
    print("=" * 60)
    model.train(**kwargs)
    print(f"✅ 학습이 완료되었습니다. 결과는 {args.project}/{args.name} 에서 확인하세요.")


def main() -> None:
    if len(sys.argv) < 2:
        run_web_gui([])
        return

    if sys.argv[1] in ("-h", "--help", "help"):
        print(HELP.strip() + "\n\n(인자 없이 실행하면 GUI가 바로 시작됩니다.)")
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "segment":
        from pipeline.auto_segment import main as run_cmd

        run_cmd(rest)

    elif cmd == "gui":
        run_web_gui(rest)

    elif cmd == "export":
        from pipeline.export import main as run_cmd

        run_cmd(rest)

    elif cmd == "build":
        from pipeline.build_split import main as run_cmd

        run_cmd(rest)

    elif cmd == "train":
        from pipeline.train_args import add_train_arguments

        parser = argparse.ArgumentParser(prog="python run.py train")
        add_train_arguments(parser)
        args = parser.parse_args(rest)
        run_train(args)

    elif cmd == "clean":
        from pipeline.clean import main as run_cmd

        run_cmd(rest)

    elif cmd == "count":
        from pipeline.stats import main as run_cmd

        run_cmd(rest)

    elif cmd == "fix-names":
        from pipeline.fix_names import main as run_cmd

        run_cmd(rest)

    elif cmd == "trim":
        from pipeline.trim import main as run_cmd

        run_cmd(rest)

    else:
        print("알 수 없는 명령:", cmd, file=sys.stderr)
        print(HELP.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
