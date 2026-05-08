"""
통합 진입점. 프로젝트 루트에서 실행하세요.

  python run.py extract --start 1 --end 5 --count 30
  python run.py segment [--input DIR] [--output DIR]
  python run.py gui [--host 0.0.0.0] [--port 7860]
  python run.py export --bg paper [--mode copy|move]
  python run.py build --num-classes 10 [--val-ratio 0.1]
  python run.py train --weights yolov8n-seg.pt [--name custom_model] [--epochs 100] [--batch 16]
  python run.py clean --mode dataset|input|stage1|stage2|training|all
  python run.py count [--dir PATH]
  python run.py fix-names [--dir PATH]
  python run.py trim stage --dir create/output_1 --keep 200
  python run.py trim dataset [--dir PATH]

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


def fix_labels_for_segmentation(yaml_path: Path):
    """
    YOLO Segmentation 학습 전에 기존 라벨(.txt)에서 
    BBox (xc, yc, w, h) 좌표가 포함되어 있다면 폴리곤 정보만 남기고 제거합니다.
    """
    dataset_dir = yaml_path.parent
    fixed_count = 0
    
    for label_file in dataset_dir.rglob("labels/*.txt"):
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 11 and len(parts) % 2 != 0:
                    try:
                        poly = [float(x) for x in parts[5:]]
                        xs = poly[0::2]
                        ys = poly[1::2]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        
                        xc, yc = float(parts[1]), float(parts[2])
                        bw, bh = float(parts[3]), float(parts[4])
                        
                        calc_xc = (min_x + max_x) / 2
                        calc_yc = (min_y + max_y) / 2
                        calc_w = max_x - min_x
                        calc_h = max_y - min_y
                        
                        if abs(xc - calc_xc) < 0.05 and abs(bw - calc_w) < 0.05:
                            parts = [parts[0]] + parts[5:]
                            changed = True
                    except ValueError:
                        pass
                new_lines.append(" ".join(parts))
            
            if changed:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                fixed_count += 1
        except Exception:
            pass
            
    if fixed_count > 0:
        print(f"🛠️ [자동 수정 완료] {fixed_count}개의 라벨 파일에서 BBox 좌표를 제거하여 순수 Segmentation 포맷으로 변경했습니다.")


def main() -> None:
    if len(sys.argv) < 2:
        run_web_gui([])
        return

    if sys.argv[1] in ("-h", "--help", "help"):
        print(HELP.strip() + "\n\n(인자 없이 실행하면 GUI가 바로 시작됩니다.)")
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "extract":
        from pipeline.bag_extract import main as run_cmd
        run_cmd(rest)

    elif cmd == "segment":
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
        parser = argparse.ArgumentParser(prog="python run.py train")
        parser.add_argument("--weights", type=str, required=True, help="학습용 가중치 (파일명만 써도 weights/ 폴더에서 탐색)")
        parser.add_argument("--name", type=str, default="custom_model", help="학습된 모델이 저장될 폴더 이름")
        parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수 (기본 100)")
        parser.add_argument("--batch", type=int, default=16, help="배치 사이즈 (기본 16)")
        parser.add_argument("--data", type=str, default="", help="데이터셋 yaml 파일 경로")
        args = parser.parse_args(rest)

        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics 라이브러리가 설치되어 있지 않습니다.")
            sys.exit(1)
            
        import config as cfg

        # 가중치 경로 설정: 입력값이 절대경로가 아니면 기본적으로 weights/ 폴더에서 찾음
        weights_path = Path(args.weights)
        if not weights_path.exists():
            weights_path = ROOT / "weights" / args.weights
        
        if not weights_path.exists():
            print(f"❌ [에러] 가중치 파일을 찾을 수 없습니다: {weights_path}")
            sys.exit(1)

        print("=" * 60)
        print(f"🚀 YOLO 학습 시작")
        print(f"   - 가중치(Weights): {weights_path}")
        print(f"   - 저장위치(Project): weights/{args.name}")
        print(f"   - 에폭(Epochs): {args.epochs}")
        print(f"   - 배치(Batch): {args.batch}")
        print("=" * 60)
        
        yaml_path = None
        if args.data:
            yaml_path = Path(args.data)
        else:
            candidates = [
                Path("create/training/dataset.yaml"),
                cfg.PROJECT_ROOT / "create" / "training" / "dataset.yaml",
                Path("/app/create/training/dataset.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    yaml_path = candidate
                    print(f"🔍 발견된 데이터셋 설정 파일: {yaml_path}")
                    break
        
        if not yaml_path or not yaml_path.exists():
            print(f"❌ [에러] dataset.yaml 파일을 찾을 수 없습니다!")
            sys.exit(1)

        fix_labels_for_segmentation(yaml_path)

        model = YOLO(str(weights_path))
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            name=args.name,
            project="weights",
            workers=0,
            amp=False
        )
        print(f"✅ 학습이 완료되었습니다. 결과는 weights/{args.name} 에서 확인하세요.")

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