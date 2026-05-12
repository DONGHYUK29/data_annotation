"""YOLO 세그 라벨: 클래스 + (선택) bbox 4개 + 폴리곤 정규화 좌표."""
from __future__ import annotations


def strip_bbox_prefix_if_polygon_line(parts: list[str]) -> list[str]:
    """
    한 줄을 공백 분리한 토큰 리스트에 대해,
    class + bbox(xc,yc,w,h) + polygon 형식이면 bbox를 제거하고 class + polygon만 남긴다.

    순수 세그 라벨(class + 짝수 개의 폴리곤 좌표만 있는 줄)은 그대로 둔다.
    bbox+폴리곤 혼합(토큰 수 11 이상, 폴리곤이 bbox와 일치)일 때만 bbox를 제거한다.
    """
    if len(parts) < 11:
        return parts
    try:
        poly = [float(x) for x in parts[5:]]
        if len(poly) < 6 or len(poly) % 2 != 0:
            return parts
        xc, yc = float(parts[1]), float(parts[2])
        bw, bh = float(parts[3]), float(parts[4])
        xs = poly[0::2]
        ys = poly[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        calc_xc = (min_x + max_x) / 2
        calc_yc = (min_y + max_y) / 2
        calc_w = max_x - min_x
        calc_h = max_y - min_y
        if (
            abs(xc - calc_xc) < 0.05
            and abs(yc - calc_yc) < 0.05
            and abs(bw - calc_w) < 0.05
            and abs(bh - calc_h) < 0.05
        ):
            return [parts[0]] + parts[5:]
    except ValueError:
        pass
    return parts


def normalize_label_text(content: str) -> str:
    """파일 전체: 줄별로 bbox 접두 제거."""
    out_lines: list[str] = []
    for line in content.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        parts = strip_bbox_prefix_if_polygon_line(parts)
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines) + ("\n" if out_lines else "")
