"""
duct_engine.py
==============
HVAC Duct Layout Engine — integrates with geometry_engine.py + heat_load.py

Given room polygons + CFM values, this module:
  1. Places AHU at centroid of all room centroids (auto)
  2. Routes trunk + branch ducts using rectilinear (L-shaped) paths
  3. Sizes ducts using Equal Friction / SMACNA velocity method
  4. Renders duct overlay on floor plan (matplotlib)
  5. Exports DXF (CAD-ready), duct sizing CSV, BOQ CSV

Public API
----------
from duct_engine import (
    place_ahu,
    route_ducts,
    size_ducts,
    render_duct_floorplan,
    export_duct_dxf,
    export_duct_csv,
    export_boq_csv,
)
"""

from __future__ import annotations

import io
import csv
import math
import base64
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Point, LineString, Polygon

# ─────────────────────────────────────────────────────────────────────────────
#  SMACNA / ASHRAE DUCT SIZING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Velocity limits (m/s) by duct type
VELOCITY_MAIN  = 7.5   # main trunk duct
VELOCITY_BRANCH= 5.0   # branch duct to room
VELOCITY_MIN   = 2.0   # minimum to avoid settling

# Aspect ratio constraint (SMACNA: max 4:1 preferred, 8:1 absolute max)
MAX_ASPECT = 4.0

# Standard duct height options (mm) — pick nearest
STD_HEIGHTS_MM = [150, 200, 250, 300, 350, 400, 500, 600, 800]

# Friction rate target (Pa/m) — Equal Friction method
FRICTION_RATE = 1.0   # Pa/m (SMACNA standard = 0.8–1.2 Pa/m)

# Air density at ~27°C (kg/m³)
RHO_AIR = 1.18

# ─────────────────────────────────────────────────────────────────────────────
#  DUCT COLOUR MAP  (by width bucket for visual clarity)
# ─────────────────────────────────────────────────────────────────────────────
def _duct_colour(width_mm: float) -> str:
    if width_mm >= 800:  return "#FF4444"   # red   — large main trunk
    if width_mm >= 500:  return "#FF8C00"   # orange — medium trunk
    if width_mm >= 300:  return "#FFD700"   # yellow — sub-trunk
    if width_mm >= 200:  return "#00CED1"   # teal  — branch
    return "#90EE90"                         # green  — small branch / flex


# ─────────────────────────────────────────────────────────────────────────────
#  1. AHU PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _get_poly(room: Dict[str, Any]):
    """Return the shapely Polygon from a room dict — handles both key names."""
    return room.get("_poly") or room.get("polygon")


def place_ahu(rooms: List[Dict[str, Any]], unit_div: float) -> Tuple[float, float]:
    """
    Place AHU at the centroid of all room centroids.

    Parameters
    ----------
    rooms     : list of room dicts (supports '_poly' or 'polygon' key)
    unit_div  : divide raw coords by this to get metres (1000 for mm drawings)

    Returns
    -------
    (ahu_x, ahu_y) in raw drawing coordinates
    """
    if not rooms:
        return (0.0, 0.0)
    xs = [_get_poly(r).centroid.x for r in rooms]
    ys = [_get_poly(r).centroid.y for r in rooms]
    return (float(np.mean(xs)), float(np.mean(ys)))


# ─────────────────────────────────────────────────────────────────────────────
#  2. DUCT SIZING  (Equal Friction + SMACNA velocity check)
# ─────────────────────────────────────────────────────────────────────────────
def _cfm_to_m3s(cfm: float) -> float:
    return cfm * 0.000471947

def _size_duct(cfm: float, is_main: bool = False) -> Dict[str, Any]:
    """
    Size a rectangular duct for the given CFM.

    Returns dict with: cfm, m3s, velocity_ms, width_mm, height_mm, area_m2,
                        equiv_diam_mm, label
    """
    m3s = _cfm_to_m3s(cfm)
    if m3s <= 0:
        return {"cfm": cfm, "m3s": 0, "velocity_ms": 0,
                "width_mm": 100, "height_mm": 100,
                "area_m2": 0.01, "equiv_diam_mm": 113, "label": "100×100"}

    v_target = VELOCITY_MAIN if is_main else VELOCITY_BRANCH
    area_m2  = m3s / v_target

    # Pick nearest standard height
    # Height: shorter dimension (fixed from standard list)
    for h_mm in STD_HEIGHTS_MM:
        h_m = h_mm / 1000.0
        w_m = area_m2 / h_m
        aspect = w_m / h_m if w_m >= h_m else h_m / w_m
        if aspect <= MAX_ASPECT:
            w_mm = int(round(w_m * 1000 / 50) * 50)   # round to 50mm
            w_mm = max(w_mm, 100)
            actual_area = (w_mm / 1000.0) * (h_mm / 1000.0)
            velocity    = m3s / actual_area
            # Equivalent circular diameter (Huebscher formula)
            equiv = 1.3 * ((w_mm * h_mm) ** 0.625) / ((w_mm + h_mm) ** 0.25)
            return {
                "cfm":         cfm,
                "m3s":         round(m3s, 4),
                "velocity_ms": round(velocity, 2),
                "width_mm":    w_mm,
                "height_mm":   h_mm,
                "area_m2":     round(actual_area, 4),
                "equiv_diam_mm": round(equiv, 0),
                "label":       f"{w_mm}×{h_mm}",
            }

    # Fallback: use max height, grow width
    h_mm = STD_HEIGHTS_MM[-1]
    h_m  = h_mm / 1000.0
    w_mm = int(round((area_m2 / h_m) * 1000 / 50) * 50)
    w_mm = max(w_mm, 100)
    actual_area = (w_mm / 1000.0) * (h_mm / 1000.0)
    velocity    = m3s / actual_area
    equiv = 1.3 * ((w_mm * h_mm) ** 0.625) / ((w_mm + h_mm) ** 0.25)
    return {
        "cfm":         cfm,
        "m3s":         round(m3s, 4),
        "velocity_ms": round(velocity, 2),
        "width_mm":    w_mm,
        "height_mm":   h_mm,
        "area_m2":     round(actual_area, 4),
        "equiv_diam_mm": round(equiv, 0),
        "label":       f"{w_mm}×{h_mm}",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  3. ROUTE DUCTS  (rectilinear L-shaped AHU → room centroid)
# ─────────────────────────────────────────────────────────────────────────────
def _rectilinear_path(x0: float, y0: float,
                      x1: float, y1: float) -> List[Tuple[float, float]]:
    """
    Return an L-shaped rectilinear path from (x0,y0) to (x1,y1).
    Goes horizontal first, then vertical (standard HVAC practice).
    """
    return [(x0, y0), (x1, y0), (x1, y1)]


def route_ducts(
    rooms: List[Dict[str, Any]],
    ahu_xy: Tuple[float, float],
    unit_div: float,
) -> List[Dict[str, Any]]:
    """
    For each room, route a branch duct from AHU to room centroid.
    Also compute cumulative CFM along each trunk segment.

    Parameters
    ----------
    rooms    : list of dicts with keys: room_id, polygon, CFM (float), AI Name/Room
    ahu_xy   : (x, y) raw coords of AHU
    unit_div : raw → metres divisor

    Returns
    -------
    list of duct segment dicts:
        segment_id, room_id, room_name, cfm, path_pts (raw coords),
        length_m, is_main (bool)
    """
    ax, ay = ahu_xy
    segments = []

    for i, room in enumerate(rooms):
        poly    = _get_poly(room)
        cx, cy  = poly.centroid.x, poly.centroid.y
        cfm     = float(room.get("CFM", 0))
        name    = room.get("AI Name", room.get("Room", f"Room {i+1}"))
        rid     = room.get("_room_id", room.get("room_id", f"R{i+1}"))

        path = _rectilinear_path(ax, ay, cx, cy)

        # Compute total path length in metres
        total_len = 0.0
        for j in range(len(path) - 1):
            dx = (path[j+1][0] - path[j][0]) / unit_div
            dy = (path[j+1][1] - path[j][1]) / unit_div
            total_len += math.hypot(dx, dy)

        segments.append({
            "segment_id": f"S{i+1:03d}",
            "room_id":    rid,
            "room_name":  name,
            "cfm":        cfm,
            "path_pts":   path,
            "length_m":   round(total_len, 2),
            "is_main":    False,
        })

    # Add a conceptual main trunk segment (AHU icon marker — zero length)
    total_cfm = sum(s["cfm"] for s in segments)
    segments.insert(0, {
        "segment_id": "S000",
        "room_id":    "AHU",
        "room_name":  "AHU — Main Supply",
        "cfm":        total_cfm,
        "path_pts":   [ahu_xy],
        "length_m":   0.0,
        "is_main":    True,
    })

    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  4. SIZE ALL SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def size_ducts(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach duct size data to each segment dict.
    Returns enriched list.
    """
    sized = []
    for seg in segments:
        sz = _size_duct(seg["cfm"], is_main=seg["is_main"])
        sized.append({**seg, **sz})
    return sized


# ─────────────────────────────────────────────────────────────────────────────
#  5. RENDER DUCT FLOOR PLAN  (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def render_duct_floorplan(
    room_data:      List[Dict[str, Any]],
    sized_segments: List[Dict[str, Any]],
    ahu_xy:         Tuple[float, float],
    raw_wall_lines: List,
    use_glass_mode: bool,
    wall_sn:        List,
    glass_sn:       List,
    unit_div:       float,
) -> Tuple[plt.Figure, bytes]:
    """
    Draw floor plan with duct overlay.

    Returns (fig, png_bytes).
    """
    fig, ax = plt.subplots(figsize=(22, 13))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    # ── Walls ────────────────────────────────────────────────────────────────
    if use_glass_mode:
        for ls in wall_sn:
            xs, ys = ls.xy
            ax.plot(xs, ys, color="#3a5a70", lw=0.7, alpha=0.6, zorder=1)
        for ls in glass_sn:
            xs, ys = ls.xy
            ax.plot(xs, ys, color="#00d4ff", lw=1.0, alpha=0.6, zorder=1)
    else:
        for l in raw_wall_lines:
            ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]],
                    color="#3a5a70", lw=0.7, alpha=0.6, zorder=1)

    # ── Room fills (light, so ducts stand out) ───────────────────────────────
    FILL_COLORS = ["#1a3a4a", "#1a4a3a", "#3a2a1a", "#2a1a4a", "#3a3a1a"]
    for i, row in enumerate(room_data):
        poly = _get_poly(row)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=FILL_COLORS[i % len(FILL_COLORS)], alpha=0.35, zorder=2)
        ax.plot(xs, ys, color="#445566", lw=1.0, zorder=2)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy,
                f"{row.get('Room', row.get('_room_id',''))}\n{row.get('CFM',0):.0f} CFM",
                ha="center", va="center", fontsize=6.5, color="#aacccc",
                fontfamily="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000cc",
                          edgecolor="#334455", linewidth=0.8))

    # ── Duct segments ─────────────────────────────────────────────────────────
    drawn_labels = set()
    for seg in sized_segments:
        if seg["room_id"] == "AHU":
            continue
        pts   = seg["path_pts"]
        w_mm  = seg.get("width_mm", 200)
        color = _duct_colour(w_mm)
        lw    = max(1.5, min(8.0, w_mm / 80))   # line width scales with duct size

        for j in range(len(pts) - 1):
            x0, y0 = pts[j]
            x1, y1 = pts[j + 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                    alpha=0.88, zorder=6, solid_capstyle="round")

        # Duct size label at midpoint of last segment
        if len(pts) >= 2:
            mx = (pts[-2][0] + pts[-1][0]) / 2
            my = (pts[-2][1] + pts[-1][1]) / 2
            lbl = seg.get("label", "")
            if lbl not in drawn_labels:
                ax.text(mx, my, lbl, fontsize=5.5, color=color,
                        ha="center", va="bottom", fontfamily="monospace",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="#00000099", edgecolor="none"))
                drawn_labels.add(lbl)

        # Diffuser dot at room centroid
        poly = None
        for row in room_data:
            if row.get("_room_id") == seg["room_id"] or \
               row.get("room_id") == seg["room_id"]:
                poly = _get_poly(row)
                break
        if poly:
            ax.plot(poly.centroid.x, poly.centroid.y, "^",
                    color=color, markersize=8, zorder=8,
                    markeredgecolor="white", markeredgewidth=0.5)

    # ── AHU symbol ───────────────────────────────────────────────────────────
    ax_x, ax_y = ahu_xy
    ahu_size = max(
        (max(_get_poly(r).bounds[2] for r in room_data) -
         min(_get_poly(r).bounds[0] for r in room_data)) * 0.03, 500)

    ahu_rect = plt.Rectangle(
        (ax_x - ahu_size, ax_y - ahu_size * 0.6),
        ahu_size * 2, ahu_size * 1.2,
        linewidth=2, edgecolor="#00FF99", facecolor="#003322",
        zorder=9, alpha=0.92)
    ax.add_patch(ahu_rect)

    # Total CFM
    total_cfm = sum(s["cfm"] for s in sized_segments if s["room_id"] != "AHU")
    main_seg  = next((s for s in sized_segments if s["room_id"] == "AHU"), {})
    main_lbl  = main_seg.get("label", "")
    ax.text(ax_x, ax_y,
            f"AHU\n{total_cfm:.0f} CFM\n{main_lbl}",
            ha="center", va="center", fontsize=7.5, color="#00FF99",
            fontweight="bold", fontfamily="monospace", zorder=10)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#FF4444", label="Main trunk (≥800mm wide)"),
        mpatches.Patch(color="#FF8C00", label="Trunk duct (500–800mm)"),
        mpatches.Patch(color="#FFD700", label="Sub-trunk (300–500mm)"),
        mpatches.Patch(color="#00CED1", label="Branch duct (200–300mm)"),
        mpatches.Patch(color="#90EE90", label="Flex / small branch (<200mm)"),
        mpatches.Patch(color="#00FF99", label="AHU unit"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              facecolor="#0f1117", edgecolor="#334455",
              labelcolor="white", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("HVAC Duct Layout — AI Generated",
                 fontsize=13, color="#00d4ff",
                 fontfamily="monospace", pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0f1117", dpi=150)
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    return fig, png_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  6. DXF EXPORT  (CAD-ready duct lines)
# ─────────────────────────────────────────────────────────────────────────────
def export_duct_dxf(
    sized_segments: List[Dict[str, Any]],
    ahu_xy: Tuple[float, float],
    unit_div: float,
) -> bytes:
    """
    Export duct centerlines as DXF (R2013).
    Each segment gets its own layer named by duct size (e.g. DUCT_400x300).
    AHU is a block reference on layer HVAC_AHU.

    Returns raw DXF bytes.
    """
    try:
        import ezdxf
    except ImportError:
        raise RuntimeError("ezdxf not installed. Run: pip install ezdxf")

    doc = ezdxf.new("R2013")
    msp = doc.modelspace()

    # Ensure AHU layer exists
    doc.layers.new(name="HVAC_AHU",   dxfattribs={"color": 3})   # green
    doc.layers.new(name="HVAC_TRUNK", dxfattribs={"color": 1})   # red
    doc.layers.new(name="HVAC_BRANCH",dxfattribs={"color": 4})   # cyan

    ax_x, ax_y = ahu_xy
    # AHU rectangle
    size = 500.0  # 500 units square
    msp.add_lwpolyline(
        [(ax_x - size, ax_y - size * 0.6),
         (ax_x + size, ax_y - size * 0.6),
         (ax_x + size, ax_y + size * 0.6),
         (ax_x - size, ax_y + size * 0.6),
         (ax_x - size, ax_y - size * 0.6)],
        dxfattribs={"layer": "HVAC_AHU", "lineweight": 50})
    msp.add_text(
        "AHU",
        dxfattribs={"layer": "HVAC_AHU", "insert": (ax_x, ax_y), "height": 200})

    for seg in sized_segments:
        if seg["room_id"] == "AHU":
            continue
        pts   = seg["path_pts"]
        w_mm  = seg.get("width_mm", 200)
        lbl   = seg.get("label", "DUCT")
        layer = f"DUCT_{lbl.replace('×','x')}"
        color = 1 if w_mm >= 500 else 4   # red for main, cyan for branch

        if layer not in [l.dxf.name for l in doc.layers]:
            doc.layers.new(name=layer, dxfattribs={"color": color})

        # Draw centerline segments
        for j in range(len(pts) - 1):
            x0, y0 = pts[j]
            x1, y1 = pts[j + 1]
            msp.add_line(
                start=(x0, y0, 0),
                end=(x1, y1, 0),
                dxfattribs={"layer": layer, "lineweight": max(13, w_mm // 30)})

        # Label midpoint of last segment
        if len(pts) >= 2:
            mx = (pts[-2][0] + pts[-1][0]) / 2
            my = (pts[-2][1] + pts[-1][1]) / 2
            msp.add_text(
                lbl,
                dxfattribs={"layer": layer, "insert": (mx, my), "height": 150,
                             "color": color})

        # Diffuser symbol (small cross at room end)
        ex, ey = pts[-1]
        d = 100.0
        msp.add_line((ex - d, ey), (ex + d, ey), dxfattribs={"layer": layer})
        msp.add_line((ex, ey - d), (ex, ey + d), dxfattribs={"layer": layer})
        msp.add_circle((ex, ey), d * 0.8, dxfattribs={"layer": layer})

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        doc.saveas(tmp_path)
        with open(tmp_path, "rb") as f:
            dxf_bytes = f.read()
    finally:
        os.unlink(tmp_path)
    return dxf_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  7. DUCT SIZING TABLE CSV
# ─────────────────────────────────────────────────────────────────────────────
def export_duct_csv(sized_segments: List[Dict[str, Any]]) -> str:
    """Return duct sizing table as CSV string."""
    fieldnames = [
        "Segment ID", "Room ID", "Room Name",
        "CFM", "Flow (m³/s)", "Velocity (m/s)",
        "Duct W×H (mm)", "Width (mm)", "Height (mm)",
        "Equiv Diam (mm)", "Length (m)",
        "Duct Type",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for seg in sized_segments:
        w.writerow({
            "Segment ID":      seg["segment_id"],
            "Room ID":         seg["room_id"],
            "Room Name":       seg["room_name"],
            "CFM":             round(seg.get("cfm", 0), 1),
            "Flow (m³/s)":     seg.get("m3s", 0),
            "Velocity (m/s)":  seg.get("velocity_ms", 0),
            "Duct W×H (mm)":   seg.get("label", ""),
            "Width (mm)":      seg.get("width_mm", 0),
            "Height (mm)":     seg.get("height_mm", 0),
            "Equiv Diam (mm)": seg.get("equiv_diam_mm", 0),
            "Length (m)":      seg.get("length_m", 0),
            "Duct Type":       "Main Trunk" if seg.get("is_main") else "Branch",
        })
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  8. BOQ CSV
# ─────────────────────────────────────────────────────────────────────────────
def export_boq_csv(sized_segments: List[Dict[str, Any]]) -> str:
    """
    Generate Bill of Quantities CSV.
    Groups segments by duct size, sums total length,
    computes surface area for sheet metal and fittings allowance.
    """
    from collections import defaultdict

    # Group by duct W×H label
    groups: Dict[str, Dict] = defaultdict(lambda: {
        "label": "", "width_mm": 0, "height_mm": 0,
        "total_length_m": 0.0, "count": 0,
        "equiv_diam_mm": 0,
    })

    for seg in sized_segments:
        if seg["room_id"] == "AHU" or seg.get("length_m", 0) == 0:
            continue
        lbl = seg.get("label", "unknown")
        g   = groups[lbl]
        g["label"]          = lbl
        g["width_mm"]       = seg.get("width_mm", 0)
        g["height_mm"]      = seg.get("height_mm", 0)
        g["equiv_diam_mm"]  = seg.get("equiv_diam_mm", 0)
        g["total_length_m"] += seg.get("length_m", 0)
        g["count"]          += 1

    fieldnames = [
        "Duct Size (W×H mm)", "Width (mm)", "Height (mm)",
        "Equiv Diam (mm)", "No. of Runs", "Total Length (m)",
        "Perimeter (m)", "Sheet Metal Area (m²)",
        "Fittings Allow (m²)", "Total Sheet Metal (m²)",
        "Unit", "Notes",
    ]

    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()

    total_sm = 0.0
    for lbl, g in sorted(groups.items(), key=lambda x: -x[1]["width_mm"]):
        wm  = g["width_mm"]  / 1000.0
        hm  = g["height_mm"] / 1000.0
        per = 2 * (wm + hm)
        sm  = per * g["total_length_m"]
        fit = sm * 0.25   # 25% allowance for fittings (elbows, tees, reducers)
        total_sm_row = sm + fit
        total_sm += total_sm_row

        w.writerow({
            "Duct Size (W×H mm)":   g["label"],
            "Width (mm)":           g["width_mm"],
            "Height (mm)":          g["height_mm"],
            "Equiv Diam (mm)":      g["equiv_diam_mm"],
            "No. of Runs":          g["count"],
            "Total Length (m)":     round(g["total_length_m"], 2),
            "Perimeter (m)":        round(per, 3),
            "Sheet Metal Area (m²)": round(sm, 2),
            "Fittings Allow (m²)":  round(fit, 2),
            "Total Sheet Metal (m²)": round(total_sm_row, 2),
            "Unit":                 "m²",
            "Notes":                "GI Sheet, 0.63mm gauge (SMACNA)",
        })

    # Summary row
    w.writerow({
        "Duct Size (W×H mm)":   "TOTAL",
        "Width (mm)":           "",
        "Height (mm)":          "",
        "Equiv Diam (mm)":      "",
        "No. of Runs":          sum(g["count"] for g in groups.values()),
        "Total Length (m)":     round(sum(g["total_length_m"] for g in groups.values()), 2),
        "Perimeter (m)":        "",
        "Sheet Metal Area (m²)": "",
        "Fittings Allow (m²)":  "",
        "Total Sheet Metal (m²)": round(total_sm, 2),
        "Unit":                 "m²",
        "Notes":                "Including 25% fittings allowance",
    })

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  9. CONVENIENCE — all-in-one pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_duct_pipeline(
    room_data:      List[Dict[str, Any]],
    unit_div:       float,
    raw_wall_lines: List,
    use_glass_mode: bool,
    wall_sn:        List,
    glass_sn:       List,
) -> Dict[str, Any]:
    """
    Full pipeline: place AHU → route → size → render → export.

    Parameters
    ----------
    room_data   : list of dicts from heat_load.compute_room_heat_loads()
                  (must have _poly, CFM, Room/_room_id keys)
    unit_div    : raw coords → metres divisor
    raw_wall_lines, use_glass_mode, wall_sn, glass_sn : from detect_rooms result

    Returns
    -------
    dict with keys:
        ahu_xy, segments, fig, png_bytes,
        duct_csv, boq_csv, dxf_bytes
    """
    # Normalise: add _room_id if missing
    for i, r in enumerate(room_data):
        if "_room_id" not in r:
            r["_room_id"] = r.get("room_id", f"R{i+1}")

    # Only rooms with CFM > 0
    active = [r for r in room_data if float(r.get("CFM", 0)) > 0]
    if not active:
        active = room_data

    ahu_xy   = place_ahu(active, unit_div)
    segments = route_ducts(active, ahu_xy, unit_div)
    sized    = size_ducts(segments)

    fig, png_bytes = render_duct_floorplan(
        room_data, sized, ahu_xy,
        raw_wall_lines, use_glass_mode,
        wall_sn, glass_sn, unit_div)

    duct_csv  = export_duct_csv(sized)
    boq_csv   = export_boq_csv(sized)
    dxf_bytes = export_duct_dxf(sized, ahu_xy, unit_div)

    return {
        "ahu_xy":    ahu_xy,
        "segments":  sized,
        "fig":       fig,
        "png_bytes": png_bytes,
        "duct_csv":  duct_csv,
        "boq_csv":   boq_csv,
        "dxf_bytes": dxf_bytes,
    }