"""
통합 진입점. 프로젝트 루트에서 실행하세요.

  python run.py extract --start 1 --end 5 --count 30
  python run.py segment [--input DIR] [--output DIR]
  python run.py gui [--host 0.0.0.0] [--port 7860]
  python run.py export --bg paper [--mode copy|move]
  python run.py build --num-classes 10 [--val-ratio 0.1]
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