#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track / CDT / Centerline / Raceline visualizer
- Updated to match new C++ CSV outputs (2025-09-30)
  * Supports:
    - *_edges_all_idx.csv
    - *_edges_labeldiff_idx.csv
    - *_edges_labeldiff_kept_idx.csv
  * centerline_with_geom.csv may include v_kappa_mps (9 columns)
  * raceline_with_geom.csv may include v_kappa_mps (7 columns)
  * Coloring options: --color-center-by v_kappa, --color-by v_kappa
  * Info panel shows v_kappa stats when available
  * NEW: annotate inner/outer cone indices (--annotate-inner-idx / --annotate-outer-idx)
"""

import argparse, csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------- small utils ----------
def drop_ext(path):
    base, _ = os.path.splitext(path)
    return base

def parse_figsize(s):
    """'W,H' 문자열을 (float, float)로 파싱. 실패 시 (9,9)."""
    try:
        w, h = s.split(",")
        return float(w), float(h)
    except Exception:
        return (9.0, 9.0)

def polyline_length(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2: return 0.0
    dx = np.diff(x); dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))

# ---------- loaders ----------
def load_xy(path):
    """두 열 (x,y) CSV. 헤더/빈줄/공백 허용."""
    xs, ys = [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                x = float(row[0].strip()); y = float(row[1].strip())
                xs.append(x); ys.append(y)
            except ValueError:
                continue
    if not xs:
        raise ValueError(f"'{path}'에서 (x,y) 데이터를 읽지 못했습니다. CSV 형식(두 열) 확인 필요.")
    return np.array(xs), np.array(ys)

def load_points_labeled(path):
    """id,x,y,label (0-based id). 헤더 허용."""
    ids, xs, ys, labs = [], [], [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header_checked = False
        for row in r:
            if not row: continue
            if not header_checked:
                header_checked = True
                try:
                    _ = int(row[0]); _ = float(row[1]); _ = float(row[2]); _ = int(row[3])
                except Exception:
                    continue
            try:
                i = int(row[0].strip())
                x = float(row[1].strip())
                y = float(row[2].strip())
                lb = int(row[3].strip())
                ids.append(i); xs.append(x); ys.append(y); labs.append(lb)
            except Exception:
                continue
    if not xs:
        raise ValueError(f"'{path}'에서 id,x,y,label 데이터를 읽지 못했습니다.")
    order = np.argsort(ids)
    ids  = np.array(ids)[order]
    xs   = np.array(xs)[order]
    ys   = np.array(ys)[order]
    labs = np.array(labs)[order]
    if not np.array_equal(ids, np.arange(len(ids))):
        print(f"[WARN] {path}: id가 0..N-1 연속이 아닙니다. (정렬은 했지만 인덱싱 주의)", file=sys.stderr)
    return ids, xs, ys, labs

def load_edges_idx(path):
    E = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2: continue
            try:
                u = int(row[0].strip()); v=int(row[1].strip())
                E.append((u,v))
            except Exception:
                continue
    return E

def load_tris_idx(path):
    T = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 3: continue
            try:
                a = int(row[0].strip()); b=int(row[1].strip()); c=int(row[2].strip())
                T.append((a,b,c))
            except Exception:
                continue
    return T

def load_center_with_geom(path):
    """
    centerline_with_geom.csv 지원 포맷(헤더 허용):
      1) s,x,y,heading_rad,curvature
      2) s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width
      3) s,x,y,heading_rad,curvature,dist_to_inner,dist_to_outer,width,v_kappa_mps
    항상 9개 배열을 반환하며, 누락된 항목은 None.
    """
    s_list, xs, ys, heads, curvs = [], [], [], [], []
    d_in_list, d_out_list, width_list, vkap_list = [], [], [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header_skipped = False
        for row in r:
            if not row: continue
            if not header_skipped:
                header_skipped = True
                try:
                    float(row[0]); float(row[1]); float(row[2])
                except Exception:
                    continue
            try:
                vals = [v.strip() for v in row]
                if len(vals) < 5:
                    continue
                s  = float(vals[0]); x=float(vals[1]); y=float(vals[2])
                hd = float(vals[3]); kv=float(vals[4])
                s_list.append(s); xs.append(x); ys.append(y); heads.append(hd); curvs.append(kv)
                if len(vals) >= 8:
                    d_in  = float(vals[5]); d_out = float(vals[6]); w = float(vals[7])
                    d_in_list.append(d_in); d_out_list.append(d_out); width_list.append(w)
                if len(vals) >= 9:
                    vkap = float(vals[8])
                    vkap_list.append(vkap)
            except Exception:
                continue
    if not xs:
        raise ValueError(f"'{path}'에서 centerline_with_geom 데이터를 읽지 못했습니다.")
    # 길이가 안 맞으면 None 처리
    if len(d_in_list)  != len(xs): d_in_arr  = None
    else:                           d_in_arr  = np.array(d_in_list)
    if len(d_out_list) != len(xs): d_out_arr = None
    else:                           d_out_arr = np.array(d_out_list)
    if len(width_list) != len(xs): width_arr = None
    else:                           width_arr = np.array(width_list)
    if len(vkap_list)  != len(xs): vkap_arr  = None
    else:                           vkap_arr  = np.array(vkap_list)
    return (np.array(s_list), np.array(xs), np.array(ys),
            np.array(heads), np.array(curvs),
            d_in_arr, d_out_arr, width_arr, vkap_arr)

def load_raceline_with_geom(path):
    """
    raceline_with_geom.csv 지원 포맷(헤더 허용):
      1) s,x,y,heading_rad,curvature,alpha
      2) s,x,y,heading_rad,curvature,alpha,v_kappa_mps
    항상 7개 배열을 반환하며, 누락된 v_kappa_mps는 None.
    """
    s_list, xs, ys, heads, curvs, alphas, vkap_list = [], [], [], [], [], [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header_skipped = False
        for row in r:
            if not row: continue
            if not header_skipped:
                header_skipped = True
                try:
                    float(row[0]); float(row[1]); float(row[2])
                except Exception:
                    continue
            try:
                vals = [v.strip() for v in row]
                if len(vals) < 6:
                    continue
                s  = float(vals[0]); x=float(vals[1]); y=float(vals[2])
                hd = float(vals[3]); kv=float(vals[4])
                al = float(vals[5])
                s_list.append(s); xs.append(x); ys.append(y)
                heads.append(hd); curvs.append(kv); alphas.append(al)
                if len(vals) >= 7:
                    vkap_list.append(float(vals[6]))
            except Exception:
                continue
    if not xs:
        raise ValueError(f"'{path}'에서 raceline_with_geom 데이터를 읽지 못했습니다.")
    vkap_arr = np.array(vkap_list) if len(vkap_list) == len(xs) else None
    return (np.array(s_list), np.array(xs), np.array(ys),
            np.array(heads), np.array(curvs), np.array(alphas), vkap_arr)

# ---------- drawing ----------
def draw_tris(ax, pts_xy, tris, **kwargs):
    xs, ys = pts_xy
    for (a,b,c) in tris:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], **kwargs)
        ax.plot([xs[b], xs[c]], [ys[b], ys[c]], **kwargs)
        ax.plot([xs[c], xs[a]], [ys[c], ys[a]], **kwargs)

def draw_edges(ax, pts_xy, edges, **kwargs):
    xs, ys = pts_xy
    for (u,v) in edges:
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], **kwargs)

# ---------- guards ----------
def check_index_bounds(name, lst, N):
    """tris/edges가 points 범위(0..N-1)에 들어오는지 검사"""
    bad = []
    if lst is None: return True
    if name.endswith("_tris"):
        for i,(a,b,c) in enumerate(lst):
            if a<0 or b<0 or c<0 or a>=N or b>=N or c>=N:
                bad.append((i,a,b,c))
    else:  # edges
        for i,(u,v) in enumerate(lst):
            if u<0 or v<0 or u>=N or v>=N:
                bad.append((i,u,v))
    if bad:
        print(f"[ERROR] {name}: {len(bad)}개의 인덱스가 points 범위를 벗어났습니다. 예) {bad[:3]}", file=sys.stderr)
        return False
    return True

# ---------- helpers: colorized polyline ----------
def _segments_from_xy(x, y):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.hstack([pts[:-1], pts[1:]])

def plot_colored_line(ax, x, y, values, label, cmap='viridis'):
    """values(길이 N)으로 선분 색을 입혀 그림"""
    if values is None:
        ax.plot(x, y, '-', linewidth=2.2, c="#000000", label=label)
        return None
    segs = _segments_from_xy(x, y)
    v = np.asarray(values)
    vmin = np.nanmin(v); vmax = np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        ax.plot(x, y, '-', linewidth=2.2, c="#000000", label=label)
        return None
    norm = (v - vmin) / (vmax - vmin)
    lc = LineCollection(segs, array=norm, cmap=cmap, linewidth=2.2)
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    return lc

# ===== 화살표/인덱스 라벨 헬퍼 =====
def draw_heading_arrows(ax, x, y, heading, step=20, arrow_len=1.0, color="#444444", alpha=0.9, width=0.004):
    x = np.asarray(x); y = np.asarray(y); h = np.asarray(heading)
    if len(x) == 0: return
    idx = np.arange(0, len(x), max(1, int(step)))
    ux = np.cos(h[idx]) * arrow_len
    uy = np.sin(h[idx]) * arrow_len
    ax.quiver(
        x[idx], y[idx], ux, uy,
        angles='xy', scale_units='xy', scale=1.0,
        color=color, alpha=alpha, width=width, minlength=0
    )

def annotate_polyline(ax, x, y, step, offset, color="#0b6e69"):
    step = max(1, int(step))
    offset = float(offset)
    n = len(x)
    for i in range(0, n, step):
        tx = float(x[i]); ty = float(y[i])
        if offset != 0.0 and i+1 < n:
            dx = float(x[i+1] - x[i]); dy = float(y[i+1] - y[i])
            L = (dx*dx + dy*dy) ** 0.5
            if L > 1e-9:
                tx += offset * dx / L
                ty += offset * dy / L
        ax.text(
            tx, ty, str(i),
            fontsize=7, ha='center', va='center', color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)
        )

# ---------- info panel ----------
def fill_info_panel(ax_info, ax_main,
                    center_geom=None, center_xy=None,
                    raceline_geom=None, raceline_xy=None,
                    points_ctx=None, tris_ctx=None, forced_ctx=None,
                    edges_all_ctx=None, edges_mixed_ctx=None, edges_kept_ctx=None):
    """오른쪽 정보 패널: 범례 + 간단 통계"""
    ax_info.set_axis_off()

    # 1) 메인 범례를 정보패널에 복제
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_info.legend(handles, labels, loc='upper left', frameon=False, fontsize=9)

    # 2) 통계 텍스트 구성
    lines = []
    if points_ctx is not None:
        ids, xs_all, ys_all, labs = points_ctx
        n_in  = int(np.sum(labs == 0)) if labs is not None else 0
        n_out = int(np.sum(labs == 1)) if labs is not None else 0
        lines.append(f"Points: {len(xs_all)} (inner {n_in}, outer {n_out})")
    if edges_all_ctx is not None:
        lines.append(f"Edges(all): {len(edges_all_ctx)}")
    if edges_mixed_ctx is not None:
        lines.append(f"Edges(label-diff): {len(edges_mixed_ctx)}")
    if edges_kept_ctx is not None:
        lines.append(f"Edges(kept label-diff): {len(edges_kept_ctx)}")
    if forced_ctx is not None:
        lines.append(f"Forced edges: {len(forced_ctx)}")
    if tris_ctx is not None:
        tri_raw, faces_kept, faces_drop = tris_ctx
        if tri_raw  is not None:  lines.append(f"Tri raw: {len(tri_raw)}")
        if faces_kept is not None:lines.append(f"Faces kept: {len(faces_kept)}")
        if faces_drop is not None:lines.append(f"Faces drop: {len(faces_drop)}")

    # 길이/곡률/폭/속도 범위
    if center_geom is not None:
        sC, cx, cy, hC, kC, d_inC, d_outC, widthC, vC = center_geom
        Lc = polyline_length(cx, cy)
        lines.append(f"Center length: {Lc:.1f} m")
        if len(kC): lines.append(f"Center κ: [{np.nanmin(kC):.4g}, {np.nanmax(kC):.4g}] 1/m")
        if widthC is not None and len(widthC):
            lines.append(f"Width: [{np.nanmin(widthC):.3g}, {np.nanmax(widthC):.3g}] m")
        if vC is not None and len(vC):
            lines.append(f"v_kappa: [{np.nanmin(vC):.3g}, {np.nanmax(vC):.3g}] m/s")
    elif center_xy is not None:
        cx, cy = center_xy
        Lc = polyline_length(cx, cy)
        lines.append(f"Center length: {Lc:.1f} m")

    if raceline_geom is not None:
        sR, rx, ry, hR, kR, aR, vR = raceline_geom
        Lr = polyline_length(rx, ry)
        lines.append(f"Race length: {Lr:.1f} m")
        if len(kR): lines.append(f"Race κ: [{np.nanmin(kR):.4g}, {np.nanmax(kR):.4g}] 1/m")
        if aR is not None and len(aR): lines.append(f"alpha: [{np.nanmin(aR):.3g}, {np.nanmax(aR):.3g}]")
        if vR is not None and len(vR): lines.append(f"v_kappa: [{np.nanmin(vR):.3g}, {np.nanmax(vR):.3g}] m/s")
    elif raceline_xy is not None:
        rx, ry = raceline_xy
        Lr = polyline_length(rx, ry)
        lines.append(f"Race length: {Lr:.1f} m")

    if lines:
        ax_info.text(0.02, 0.98, "Stats", fontsize=10, fontweight='bold',
                     va='top', transform=ax_info.transAxes)

# ====== CLI / main ======

def main():
    ap = argparse.ArgumentParser(description="CDT / Centerline / Raceline 시각화 (updated for new CSVs)")
    # xy csv
    ap.add_argument("--inner",  help="inner.csv (x,y) (정렬된 inner_from_mids.csv 권장)")
    ap.add_argument("--outer",  help="outer.csv (x,y) (정렬된 outer_from_mids.csv 권장)")
    ap.add_argument("--center", help="centerline.csv (x,y)")
    ap.add_argument("--center-geom", help="centerline_with_geom.csv (s,x,y,heading_rad,curvature[,dist_in,dist_out,width[,v_kappa_mps]])")

    # raceline
    ap.add_argument("--raceline",      help="*_raceline.csv (x,y)")
    ap.add_argument("--raceline-geom", help="*_raceline_with_geom.csv (s,x,y,heading_rad,curvature,alpha[,v_kappa_mps])")

    # dump prefix (CDT 관련)
    ap.add_argument("--prefix", help="C++ 덤프 prefix (예: centerline)")

    # files (개별 지정)
    ap.add_argument("--points",      help="*_all_points.csv (id,x,y,label)")
    ap.add_argument("--tri_raw",     help="*_tri_raw_idx.csv (a,b,c)")
    ap.add_argument("--faces_kept",  help="*_faces_kept_idx.csv (a,b,c)")
    ap.add_argument("--faces_drop",  help="*_faces_drop_idx.csv (a,b,c)")
    # (old) forced edges, kept for backward compatibility
    ap.add_argument("--forced",      help="*_forced_edges_idx.csv (u,v)")
    # (new) edge dumps
    ap.add_argument("--edges-all",        help="*_edges_all_idx.csv (u,v)")
    ap.add_argument("--edges-labeldiff",  help="*_edges_labeldiff_idx.csv (u,v)")
    ap.add_argument("--edges-kept",       help="*_edges_labeldiff_kept_idx.csv (u,v)")

    # toggles
    ap.add_argument("--show-tri-raw",      action="store_true")
    ap.add_argument("--show-faces-kept",   action="store_true")
    ap.add_argument("--show-faces-drop",   action="store_true")
    ap.add_argument("--show-forced",       action="store_true")
    ap.add_argument("--show-edges-all",        action="store_true")
    ap.add_argument("--show-edges-labeldiff",  action="store_true")
    ap.add_argument("--show-edges-kept",       action="store_true")
    ap.add_argument("--show-mids",         action="store_true")
    ap.add_argument("--show-mids-ordered", action="store_true")
    ap.add_argument("--show-raceline",     action="store_true", help="raceline을 평면에 표시")

    # annotation for centerline indices
    ap.add_argument("--annotate-center-idx", action="store_true",
                    help="센터라인 각 점에 인덱스 라벨 표시")
    ap.add_argument("--idx-step", type=int, default=20,
                    help="센터라인 인덱스 라벨 간격(기본 20)")
    ap.add_argument("--idx-offset", type=float, default=0.0,
                    help="센터라인 라벨 위치 오프셋(미터 단위, 0이면 점 위치에 그대로 표시)")

    # NEW: inner/outer index annotation
    ap.add_argument("--annotate-inner-idx", action="store_true",
                    help="정렬된 inner 콘에 인덱스 라벨 표시")
    ap.add_argument("--inner-idx-step", type=int, default=10,
                    help="inner 인덱스 라벨 간격(기본 10)")
    ap.add_argument("--inner-idx-offset", type=float, default=0.0,
                    help="inner 라벨 위치 오프셋(미터)")
    ap.add_argument("--annotate-outer-idx", action="store_true",
                    help="정렬된 outer 콘에 인덱스 라벨 표시")
    ap.add_argument("--outer-idx-step", type=int, default=10,
                    help="outer 인덱스 라벨 간격(기본 10)")
    ap.add_argument("--outer-idx-offset", type=float, default=0.0,
                    help="outer 라벨 위치 오프셋(미터)")

    ap.add_argument("--save", help="결과 저장 파일명(예: track.png)")
    ap.add_argument("--title", default="Track Visualization (CDT / Center / Raceline)")

    # 레이스라인 컬러/메트릭
    ap.add_argument("--color-by", choices=["none","heading","curvature","alpha","v_kappa"], default="none",
                    help="raceline 색상 매핑: heading/curvature/alpha/v_kappa")
    ap.add_argument("--show-metrics", action="store_true",
                    help="raceline의 s-축에 대한 heading/curvature/alpha/v_kappa 곡선 표시")

    # 헤딩 화살표
    ap.add_argument("--heading-arrows", action="store_true",
                    help="heading_rad을 화살표로 시각화")
    ap.add_argument("--arrow-src", choices=["auto","centerline","raceline"], default="auto",
                    help="헤딩 화살표의 데이터 소스 선택 (auto: raceline_with_geom 우선, 없으면 centerline_with_geom)")
    ap.add_argument("--arrow-step", type=int, default=20, help="화살표 간격(샘플 인덱스)")
    ap.add_argument("--arrow-len", type=float, default=1.0, help="화살표 길이(미터)")
    ap.add_argument("--arrow-alpha", type=float, default=0.9, help="화살표 투명도(0~1)")
    ap.add_argument("--arrow-width", type=float, default=0.004, help="화살표 선 두께")
    ap.add_argument("--arrow-color", default="#444444", help="화살표 색상 (예: #333333)")

    # 센터라인 컬러/메트릭
    ap.add_argument("--color-center-by", choices=["none","heading","curvature","width","v_kappa"], default="none",
                    help="centerline 색상 매핑: heading/curvature/width/v_kappa")
    ap.add_argument("--show-center-metrics", action="store_true",
                    help="centerline의 s-축에 대한 heading/curvature/width/v_kappa 곡선 표시")

    # raceline index annotation
    ap.add_argument("--annotate-raceline-idx", action="store_true",
                    help="레이스라인 각 점에 인덱스 라벨 표시")
    ap.add_argument("--raceline-idx-step", type=int, default=20,
                    help="레이스라인 인덱스 라벨 간격(기본 20)")
    ap.add_argument("--raceline-idx-offset", type=float, default=0.0,
                    help="레이스라인 라벨 위치 오프셋(미터 단위, 0이면 점 위치에 그대로 표시)")

    # ----- figure / layout options -----
    ap.add_argument("--figsize", default="9,9",
                    help="Figure size in inches 'W,H' (default: 9,9)")
    ap.add_argument("--legend-outside", action="store_true",
                    help="범례를 플롯 오른쪽 바깥으로 배치")
    ap.add_argument("--side-panel", action="store_true",
                    help="오른쪽에 정보 패널 subplot 추가")
    ap.add_argument("--panel-ratio", type=float, default=0.24,
                    help="side-panel 너비 비율 (0~1, default 0.24)")

    args = ap.parse_args()

    # ===== 경로 자동 추론 =====
    # 1) prefix로 CDT 관련 묶음 채우기
    if args.prefix:
        base = args.prefix
        args.points       = args.points       or f"{base}_all_points.csv"
        args.tri_raw      = args.tri_raw      or f"{base}_tri_raw_idx.csv"
        args.faces_kept   = args.faces_kept   or f"{base}_faces_kept_idx.csv"
        args.faces_drop   = args.faces_drop   or f"{base}_faces_drop_idx.csv"
        args.mids_raw     = getattr(args, 'mids_raw', None) or f"{base}_mids_raw.csv"
        args.mids_ordered = getattr(args, 'mids_ordered', None) or f"{base}_mids_ordered.csv"
        # (old)
        args.forced       = args.forced       or f"{base}_forced_edges_idx.csv"
        # (new)
        args.edges_all        = args.edges_all       or f"{base}_edges_all_idx.csv"
        args.edges_labeldiff  = args.edges_labeldiff or f"{base}_edges_labeldiff_idx.csv"
        args.edges_kept       = args.edges_kept      or f"{base}_edges_labeldiff_kept_idx.csv"

    # 2) center가 있으면 with_geom / raceline도 같은 베이스명에서 유추
    if args.center and os.path.exists(args.center):
        base = drop_ext(args.center)
        if not args.center_geom:
            cand = base + "_with_geom.csv"
            if os.path.exists(cand): args.center_geom = cand
        if not args.raceline:
            cand = base + "_raceline.csv"
            if os.path.exists(cand): args.raceline = cand
        if not args.raceline_geom:
            cand = base + "_raceline_with_geom.csv"
            if os.path.exists(cand): args.raceline_geom = cand

    # ===== load =====
    # points (id,x,y,label) → 인덱스 기준
    ids = xs_all = ys_all = labs = None
    if args.points and os.path.exists(args.points):
        ids, xs_all, ys_all, labs = load_points_labeled(args.points)

    # inner/outer
    inner_xy = outer_xy = None
    if args.inner and os.path.exists(args.inner):
        inner_xy = load_xy(args.inner)
    if args.outer and os.path.exists(args.outer):
        outer_xy = load_xy(args.outer)

    # center/with_geom
    center_xy = None
    center_geom = None
    if args.center and os.path.exists(args.center):
        center_xy = load_xy(args.center)
    if args.center_geom and os.path.exists(args.center_geom):
        center_geom = load_center_with_geom(args.center_geom)  # (s,x,y,heading,curv,[d_in,d_out,width,v_kappa])

    # raceline
    raceline_xy = None
    raceline_geom = None
    if args.raceline and os.path.exists(args.raceline):
        raceline_xy = load_xy(args.raceline)
    if args.raceline_geom and os.path.exists(args.raceline_geom):
        raceline_geom = load_raceline_with_geom(args.raceline_geom)

    # indexed stuff
    forced_edges = tri_raw = faces_kept = faces_drop = None
    edges_all = edges_mixed = edges_kept = None
    if args.forced and os.path.exists(args.forced):
        forced_edges = load_edges_idx(args.forced)
    if args.tri_raw and os.path.exists(args.tri_raw):
        tri_raw = load_tris_idx(args.tri_raw)
    if args.faces_kept and os.path.exists(args.faces_kept):
        faces_kept = load_tris_idx(args.faces_kept)
    if args.faces_drop and os.path.exists(args.faces_drop):
        faces_drop = load_tris_idx(args.faces_drop)

    # (new) edges
    if args.edges_all and os.path.exists(args.edges_all):
        edges_all = load_edges_idx(args.edges_all)
    if args.edges_labeldiff and os.path.exists(args.edges_labeldiff):
        edges_mixed = load_edges_idx(args.edges_labeldiff)
    if args.edges_kept and os.path.exists(args.edges_kept):
        edges_kept = load_edges_idx(args.edges_kept)

    # mids (optional)
    mids_raw_xy = mids_ord_xy = None
    if getattr(args, 'mids_raw', None) and os.path.exists(args.mids_raw):
        mids_raw_xy = load_xy(args.mids_raw)
    if getattr(args, 'mids_ordered', None) and os.path.exists(args.mids_ordered):
        mids_ord_xy = load_xy(args.mids_ordered)

    # ===== sanity checks =====
    if xs_all is not None:
        N = len(xs_all)
        ok = True
        ok &= check_index_bounds("tri_raw_tris",   tri_raw,    N) if tri_raw else True
        ok &= check_index_bounds("faces_kept_tris",faces_kept, N) if faces_kept else True
        ok &= check_index_bounds("faces_drop_tris",faces_drop, N) if faces_drop else True
        ok &= check_index_bounds("forced_edges",   forced_edges,N) if forced_edges else True
        ok &= check_index_bounds("edges_all",      edges_all,   N) if edges_all else True
        ok &= check_index_bounds("edges_labeldiff",edges_mixed, N) if edges_mixed else True
        ok &= check_index_bounds("edges_kept",     edges_kept,  N) if edges_kept else True
        if not ok:
            print("[FATAL] 인덱스 범위 오류. 덤프/매핑 불일치 가능성이 큽니다.", file=sys.stderr)
            sys.exit(2)

    # ===== draw (plan view) =====
    W, H = parse_figsize(args.figsize)
    if args.side_panel:
        fig = plt.figure(figsize=(W, H))
        pr = min(max(args.panel_ratio, 0.08), 0.48)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[1.0 - pr, pr], figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(W, H))
        ax_info = None

    # inner/outer (raw points or polylines)
    if xs_all is not None:
        mask_in  = (labs == 0)
        mask_out = (labs == 1)
        ax.scatter(xs_all[mask_in],  ys_all[mask_in],  s=8, c="#1f77b4", label="Inner (label=0)")
        ax.scatter(xs_all[mask_out], ys_all[mask_out], s=8, c="#ff7f0e", label="Outer (label=1)")
    else:
        if inner_xy is not None:
            ax.plot(inner_xy[0], inner_xy[1], '-', linewidth=1.5, c="#1f77b4", label="Inner(sorted)")
        if outer_xy is not None:
            ax.plot(outer_xy[0], outer_xy[1], '-', linewidth=1.5, c="#ff7f0e", label="Outer(sorted)")

    # (NEW) annotate inner/outer indices
    if getattr(args, 'annotate_inner_idx', False) and inner_xy is not None:
        annotate_polyline(ax, inner_xy[0], inner_xy[1], args.inner_idx_step, args.inner_idx_offset, color="#1f77b4")
    if getattr(args, 'annotate_outer_idx', False) and outer_xy is not None:
        annotate_polyline(ax, outer_xy[0], outer_xy[1], args.outer_idx_step, args.outer_idx_offset, color="#ff7f0e")

    # CDT primitives
    if xs_all is not None:
        pts_xy = (xs_all, ys_all)
        if args.show_tri_raw and tri_raw:
            draw_tris(ax, pts_xy, tri_raw, color="#cccccc", linewidth=0.5, alpha=0.7)
        if args.show_faces_kept and faces_kept:
            draw_tris(ax, pts_xy, faces_kept, color="#d62728", linewidth=0.8, alpha=0.9)
        if args.show_faces_drop and faces_drop:
            draw_tris(ax, pts_xy, faces_drop, color="#999999", linewidth=0.5, alpha=0.4, linestyle="--")
        if args.show_forced and forced_edges:
            draw_edges(ax, pts_xy, forced_edges, color="#2ca02c", linewidth=1.4, alpha=0.9)
        if args.show_edges_all and edges_all:
            draw_edges(ax, pts_xy, edges_all, color="#bbbbbb", linewidth=0.6, alpha=0.6)
        if args.show_edges_labeldiff and edges_mixed:
            draw_edges(ax, pts_xy, edges_mixed, color="#9467bd", linewidth=1.1, alpha=0.9)
        if args.show_edges_kept and edges_kept:
            draw_edges(ax, pts_xy, edges_kept, color="#2ca02c", linewidth=1.8, alpha=0.95)

    # midpoints
    if args.show_mids and 'mids_raw_xy' in locals() and mids_raw_xy is not None:
        ax.scatter(mids_raw_xy[0], mids_raw_xy[1], s=12, c="#9467bd", label="Midpoints (raw)")
    if args.show_mids_ordered and 'mids_ord_xy' in locals() and mids_ord_xy is not None:
        ax.plot(mids_ord_xy[0], mids_ord_xy[1], '-', linewidth=2.0, c="#8c564b", label="Midpoints (ordered)")

    # centerline (with_geom 우선)
    cx = cy = None
    if center_geom is not None:
        s_vals, cx, cy, headings, curvs, d_in, d_out, widthC, vC = center_geom
        if args.color_center_by != "none":
            if args.color_center_by == "heading":
                plot_colored_line(ax, cx, cy, headings, "center heading_rad")
            elif args.color_center_by == "curvature":
                plot_colored_line(ax, cx, cy, curvs, "center curvature [1/m]")
            elif args.color_center_by == "width":
                if widthC is None:
                    print("[WARN] centerline width 데이터가 없어 단색으로 표시합니다.", file=sys.stderr)
                    ax.plot(cx, cy, '-', linewidth=2.2, c="#000000", label='Centerline')
                else:
                    plot_colored_line(ax, cx, cy, widthC, "center width [m]")
            elif args.color_center_by == "v_kappa":
                if vC is None:
                    print("[WARN] centerline v_kappa 데이터가 없어 단색으로 표시합니다.", file=sys.stderr)
                    ax.plot(cx, cy, '-', linewidth=2.2, c="#000000", label='Centerline')
                else:
                    plot_colored_line(ax, cx, cy, vC, "center v_kappa [m/s]")
            ax.plot([], [], '-', c="#000000", linewidth=0, label='Centerline (colored)')
        else:
            ax.plot(cx, cy, '-', linewidth=2.2, c="#000000", label='Centerline')
    elif center_xy is not None:
        cx, cy = center_xy
        ax.plot(cx, cy, '-', linewidth=2.2, c="#000000", label='Centerline')

    # (optional) center index annotation
    if getattr(args, 'annotate_center_idx', False) and (center_geom is not None or center_xy is not None):
        if center_geom is not None:
            _, cxg, cyg, *_ = center_geom
            annotate_polyline(ax, cxg, cyg, args.idx_step, args.idx_offset, color="#000000")
        else:
            annotate_polyline(ax, center_xy[0], center_xy[1], args.idx_step, args.idx_offset, color="#000000")

    # ===== centerline s-축 메트릭 Figure =====
    if args.show_center_metrics and (center_geom is not None):
        sC, cxg, cyg, hC, kC, d_inC, d_outC, widthC, vC = center_geom
        plt.figure(figsize=(10, 6))
        axc = plt.gca()
        axc.plot(sC, hC, '-', color='red', label='heading_rad')
        axc.set_xlabel("s [m]")
        axc.set_ylabel("heading_rad", color='red')
        axc.tick_params(axis='y', colors='red')
        axc.grid(True, linestyle='--', alpha=0.35)
        axc2 = axc.twinx()
        axc2.plot(sC, kC, '-', color='blue', label='curvature [1/m]', alpha=0.9)
        axc2.set_ylabel("curvature [1/m]", color='blue')
        axc2.tick_params(axis='y', colors='blue')
        if widthC is not None:
            axc.plot(sC, widthC, '--', color='green', label='width [m]', alpha=0.9)
        if vC is not None:
            axc.plot(sC, vC, ':', color='purple', label='v_kappa [m/s]', alpha=0.9)
        lines, labels = [], []
        for ax_ in (axc, axc2):
            L = ax_.get_legend_handles_labels()
            lines += L[0]; labels += L[1]
        axc.legend(lines, labels, loc='best')
        plt.title("Centerline metrics vs. distance s")

    # raceline
    if args.show_raceline:
        if raceline_geom is not None:
            sR, rx, ry, hR, kR, aR, vR = raceline_geom
            if args.color_by == "heading":
                plot_colored_line(ax, rx, ry, hR, "heading_rad")
                ax.plot([], [], '-', c="#17becf", lw=2.2, label='Raceline (colored)')
            elif args.color_by == "curvature":
                plot_colored_line(ax, rx, ry, kR, "curvature [1/m]")
                ax.plot([], [], '-', c="#17becf", lw=2.2, label='Raceline (colored)')
            elif args.color_by == "alpha":
                plot_colored_line(ax, rx, ry, aR, "alpha")
                ax.plot([], [], '-', c="#17becf", lw=2.2, label='Raceline (colored)')
            elif args.color_by == "v_kappa":
                if vR is None:
                    print("[WARN] raceline v_kappa 데이터가 없어 단색으로 표시합니다.", file=sys.stderr)
                    ax.plot(rx, ry, '-', linewidth=2.2, c="#17becf", label='Raceline', zorder=3)
                else:
                    plot_colored_line(ax, rx, ry, vR, "v_kappa [m/s]")
                    ax.plot([], [], '-', c="#17becf", lw=2.2, label='Raceline (colored)')
            else:
                ax.plot(rx, ry, '-', linewidth=2.2, c="#17becf", label='Raceline', zorder=3)
        elif raceline_xy is not None:
            ax.plot(raceline_xy[0], raceline_xy[1], '-', linewidth=2.2,
                    c="#17becf", label='Raceline', zorder=3)

    # -------- heading 화살표 --------
    if args.heading_arrows:
        src = args.arrow_src
        drawn = False
        if (src in ["auto","raceline"]) and (raceline_geom is not None):
            sR, rx, ry, hR, kR, aR, vR = raceline_geom
            draw_heading_arrows(ax, rx, ry, hR,
                                step=args.arrow_step,
                                arrow_len=args.arrow_len,
                                color=args.arrow_color,
                                alpha=args.arrow_alpha,
                                width=args.arrow_width)
            drawn = True
        if (not drawn) and (src in ["auto","centerline"]) and (center_geom is not None):
            sC, cxg, cyg, hC, kC, d_inC, d_outC, widthC, vC = center_geom
            draw_heading_arrows(ax, cxg, cyg, hC,
                                step=args.arrow_step,
                                arrow_len=args.arrow_len,
                                color=args.arrow_color,
                                alpha=args.arrow_alpha,
                                width=args.arrow_width)
            drawn = True
        if not drawn:
            print("[WARN] heading_rad 데이터가 없습니다. *_raceline_with_geom.csv 또는 centerline_with_geom.csv 필요.", file=sys.stderr)

    # ----- 레이스라인 인덱스 라벨 표시 -----
    if args.annotate_raceline_idx:
        rx_lab = ry_lab = None
        if raceline_geom is not None:
            sR, rx, ry, hR, kR, aR, vR = raceline_geom
            rx_lab, ry_lab = rx, ry
        elif raceline_xy is not None:
            rx_lab, ry_lab = raceline_xy

        if rx_lab is not None:
            annotate_polyline(ax, rx_lab, ry_lab, args.raceline_idx_step, args.raceline_idx_offset, color="#0b6e69")
        else:
            print("[WARN] 레이스라인 데이터가 없어 인덱스 라벨을 표시할 수 없습니다.", file=sys.stderr)

    # ===== s-축 메트릭 Figure (raceline) =====
    if args.show_metrics and (raceline_geom is not None):
        sR, rx, ry, hR, kR, aR, vR = raceline_geom
        plt.figure(figsize=(10, 6))
        axm = plt.gca()
        axm.plot(sR, hR, '-', color='red', label='heading_rad')
        axm.set_xlabel("s [m]")
        axm.set_ylabel("heading_rad", color='red')
        axm.tick_params(axis='y', colors='red')
        axm.grid(True, linestyle='--', alpha=0.35)
        ax2 = axm.twinx()
        ax2.plot(sR, kR, '-', color='blue', label='curvature [1/m]', alpha=0.9)
        ax2.set_ylabel("curvature [1/m]", color='blue')
        ax2.tick_params(axis='y', colors='blue')
        if aR is not None:
            axm.plot(sR, aR, '--', color='green', label='alpha', alpha=0.9)
        if vR is not None:
            axm.plot(sR, vR, ':', color='purple', label='v_kappa [m/s]', alpha=0.9)
        lines, labels = [], []
        for ax_ in (axm, ax2):
            L = ax_.get_legend_handles_labels()
            lines += L[0]; labels += L[1]
        axm.legend(lines, labels, loc='best')
        plt.title("Raceline metrics vs. distance s")

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_title(args.title)

    # 범례 배치 + 사이드 패널 구성
    if args.legend_outside and not args.side_panel:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        fig.tight_layout()
    else:
        ax.legend(loc='best')

    if args.side_panel and ax_info is not None:
        fill_info_panel(
            ax_info, ax,
            center_geom=center_geom, center_xy=center_xy,
            raceline_geom=raceline_geom, raceline_xy=raceline_xy,
            points_ctx=(ids, xs_all, ys_all, labs) if ids is not None else None,
            tris_ctx=(tri_raw, faces_kept, faces_drop),
            forced_ctx=forced_edges,
            edges_all_ctx=edges_all,
            edges_mixed_ctx=edges_mixed,
            edges_kept_ctx=edges_kept
        )
        fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Saved plan-view to {args.save}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
