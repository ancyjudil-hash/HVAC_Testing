"""
app_patch.py  — v22 patches for app.py
=======================================
This file shows EXACTLY what to change in app.py.
It is NOT a standalone file — copy each section into app.py as indicated.

CHANGES REQUIRED IN app.py:
1. Update the import from geometry_engine (add detect_rooms_mode_d, auto_unit_factor)
2. Replace detect_rooms_cached() entirely with the version below
3. Update the mode badge logic (minor)
4. Add Mode D option to the sidebar Mode selectbox
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 1 — geometry_engine import line (replace the existing one)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH_1_IMPORT = """
from geometry_engine import (
    extract_all_v8, process_entity_v11,
    detect_rooms_mode_a, detect_rooms_mode_b,
    detect_rooms_mode_c, detect_rooms_mode_d,   # ← added mode_d
    extract_all_layers, guess_wall_glass_layers,
    process_furniture_to_objects,
    auto_unit_factor,                            # ← added helper
)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 2 — Sidebar Mode selectbox (replace the existing selectbox)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH_2_SIDEBAR = """
    mode_override = st.selectbox("Mode", [
        "Auto-detect",
        "Force Closed-polyline (Mode D)",        # ← NEW — try first
        "Force Glass-partition (Mode A)",
        "Force Base-layer only (Mode B)",
        "Force All-layer (Mode C)",
    ], index=0)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 3 — CSS additions (add inside the existing <style> block)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH_3_CSS = """
  .mode-d-badge {
    display:inline-block; background:#1a1a2e; border:1px solid #9575cd88;
    border-radius:6px; padding:3px 10px; font-size:11px; color:#b39ddb;
    font-family:monospace; margin-left:8px;
  }
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 4 — REPLACE detect_rooms_cached() ENTIRELY with this version
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import streamlit as st
import ezdxf
import numpy as np
from shapely.geometry import LineString, Point
from shapely import wkb as shapely_wkb

@st.cache_data(show_spinner=False)
def detect_rooms_cached(
    dxf_path, wall_layers, glass_layers, furn_layers, mode_override,
    snap_tol, bridge_tol, glass_edge_thresh, glass_proximity_mult,
    min_area_m2, max_area_m2, min_compact, max_aspect_a,
    outer_area_pct_b, min_solidity, max_aspect_b,
    max_interior_walls, exclude_stairs,
    stair_parallel_min, stair_angle_tol, max_stair_area_m2,
    gap_close_tol, max_door_width, min_wall_len,
    force_mode_c=False,
) -> dict:
    """
    v22: Mode D runs first in Auto-detect mode.
    Falls through to A / B / C if Mode D finds nothing.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    WALL  = wall_layers
    GLASS = glass_layers
    FURN  = furn_layers

    # ── Measure drawing span for unit detection ───────────────────────────
    all_xs_sample = []
    for ent in msp:
        if ent.dxftype() == "LWPOLYLINE":
            all_xs_sample.extend([p[0] for p in ent.get_points()])
        elif ent.dxftype() == "LINE":
            all_xs_sample.extend([ent.dxf.start.x, ent.dxf.end.x])
        if len(all_xs_sample) > 500:
            break

    if all_xs_sample:
        span = max(all_xs_sample) - min(all_xs_sample)
    else:
        span = 1000.0  # fallback assume mm

    unit_factor, unit_div = auto_unit_factor(span)

    # ── Entity coverage check (for auto-mode decisions) ───────────────────
    total_geom_ents = sum(1 for ent in msp
                         if ent.dxftype() in
                         {"LINE","LWPOLYLINE","POLYLINE","ARC","CIRCLE","ELLIPSE","SPLINE","INSERT"})
    wall_ent_count  = sum(1 for ent in msp
                         if ent.dxf.get("layer","0").upper() in WALL)
    coverage_pct    = (wall_ent_count / max(total_geom_ents, 1)) * 100

    insunits   = doc.header.get("$INSUNITS", 0)
    scale_map  = {0:1.0,1:25.4,2:304.8,4:1.0,5:10.0,6:1000.0}
    draw_scale = scale_map.get(insunits, 1.0)

    # ══════════════════════════════════════════════════════════════════════
    #  MODE D — CLOSED POLYLINE FAST PATH  (runs in Auto + Force D)
    # ══════════════════════════════════════════════════════════════════════
    try_mode_d = (mode_override in ("Auto-detect", "Force Closed-polyline (Mode D)"))

    if try_mode_d:
        # Try with wall layers first; if nothing found, try ALL layers
        for allowed in [WALL if WALL else None, None]:
            mode_d_rooms, mode_d_info = detect_rooms_mode_d(
                msp=msp, doc=doc,
                allowed_layers_up=allowed,
                min_area_m2=float(min_area_m2),
                max_area_m2=float(max_area_m2),
                unit_factor=unit_factor,
                unit_div=unit_div,
                log=None,
            )
            if mode_d_rooms:
                break  # found rooms — stop trying

        if mode_d_rooms:
            lyr_str = ", ".join(mode_d_info["layers_seen"][:4])
            rooms_serial = []
            for r in mode_d_rooms:
                poly = r["polygon"]
                minx, miny, maxx, maxy = poly.bounds
                rooms_serial.append({
                    "name":     r["name"],
                    "room_id":  r["room_id"],
                    "poly_wkb": poly.wkb,
                    "gf":       0.0,
                    "is_glass": False,
                    "width":    maxx - minx,
                    "height":   maxy - miny,
                    "area":     poly.area,
                    "objects_inside": [],
                })
            return {
                "rooms_serial":      rooms_serial,
                "unit_factor":       unit_factor,
                "unit_div":          unit_div,
                "use_glass":         False,
                "mode_label":        f"🟣 Mode D: Closed polyline  ({len(mode_d_rooms)} rooms · layers: {lyr_str})",
                "coverage_pct":      100.0,
                "force_c":           False,
                "wall_sn_wkb":       [],
                "glass_sn_wkb":      [],
                "bridges_wkb":       [],
                "wall_cavities_wkb": [],
                "all_segs_wkb":      [],
                "raw_wall_lines":    [],
                "extracted_objects": [],
            }
        # Mode D found nothing → fall through to A / B / C

    # ══════════════════════════════════════════════════════════════════════
    #  MODE DECISION  (A / B / C)
    # ══════════════════════════════════════════════════════════════════════
    glass_found = bool(GLASS)
    if   mode_override == "Force Glass-partition (Mode A)": use_glass=True;  force_c=False
    elif mode_override == "Force Base-layer only (Mode B)": use_glass=False; force_c=False
    elif mode_override == "Force All-layer (Mode C)":       use_glass=False; force_c=True
    else:
        force_c   = force_mode_c or (coverage_pct < 5.0)
        use_glass = bool(GLASS) and glass_found and not force_c

    if force_c:
        mode_label = "🟠 Mode C: All-layer fallback"
    elif use_glass:
        mode_label = "🔵 Mode A: Glass-partition"
    else:
        mode_label = "🟩 Mode B: Base-layer"

    raw_furn_lines = []

    # ══════════════════════════════════════════════════════════════════════
    #  MODE C
    # ══════════════════════════════════════════════════════════════════════
    if force_c:
        _SKIP_KW_C = {"text","dim","hatch","title","vp","defpoint",
                      "annotation","border","viewport"}
        skip_layers_up = set()
        for ent in msp:
            lyr = ent.dxf.get("layer","0")
            if any(kw in lyr.lower() for kw in _SKIP_KW_C):
                skip_layers_up.add(lyr.upper())

        raw_c = extract_all_layers(msp, doc, skip_layers_up)
        all_xs = [c[0] for (ls,_,_) in raw_c for c in ls.coords]
        if not all_xs:
            raise RuntimeError("No geometry found in any layer. Is this a valid DXF?")

        span = max(all_xs) - min(all_xs)
        unit_factor, unit_div = auto_unit_factor(span)

        accepted, bridges_used, all_sn = detect_rooms_mode_c(
            all_layer_segs=raw_c,
            snap_tol=snap_tol, bridge_tol=bridge_tol,
            min_area_m2=min_area_m2, max_area_m2=max_area_m2,
            min_compact=min_compact, max_aspect=max_aspect_a,
            unit_factor=unit_factor, log=None)

        wall_sn=[]; glass_sn=[]; wall_cavities=[]; raw_wall_lines=[]; extracted_objects_serial=[]
        all_segs_d = [ls for (ls,_,_) in raw_c]
        rooms_serial=[]
        for i,(poly,gf,is_glass) in enumerate(accepted):
            minx,miny,maxx,maxy=poly.bounds
            rooms_serial.append({
                "name":f"Room {i+1}","room_id":f"R{i+1}",
                "poly_wkb":poly.wkb,"gf":0.0,"is_glass":False,
                "width":maxx-minx,"height":maxy-miny,"area":poly.area,
                "objects_inside":[],
            })
        return {
            "rooms_serial":rooms_serial,"unit_factor":unit_factor,"unit_div":unit_div,
            "use_glass":False,"mode_label":mode_label,"coverage_pct":coverage_pct,"force_c":True,
            "wall_sn_wkb":[ls.wkb for ls in wall_sn],"glass_sn_wkb":[],
            "bridges_wkb":[ls.wkb for ls in bridges_used],"wall_cavities_wkb":[],
            "all_segs_wkb":[ls.wkb for ls in all_segs_d[:8000]],
            "raw_wall_lines":[],"extracted_objects":[],
        }

    # ══════════════════════════════════════════════════════════════════════
    #  MODE A / B
    # ══════════════════════════════════════════════════════════════════════
    if use_glass:
        ALLOWED_A  = WALL | GLASS
        raw_a      = extract_all_v8(msp, doc, ALLOWED_A, GLASS)
        wall_segs  = [g for (g,ig,ia) in raw_a if not ig and not ia]
        glass_segs = [g for (g,ig,ia) in raw_a if ig and not ia]
        arc_segs   = [g for (g,ig,ia) in raw_a if ia]
        all_segs_d = [g for (g,_,_) in raw_a]
        all_xs     = [c[0] for ls in wall_segs+glass_segs for c in ls.coords]
        if not all_xs:
            raise RuntimeError("No wall/glass geometry found. Check layer names.")
        span = max(all_xs) - min(all_xs)
        unit_factor, unit_div = auto_unit_factor(span)
        raw_wall_lines = []
    else:
        wall_segs=glass_segs=arc_segs=[]; all_segs_d=[]; raw_wall_lines=[]; raw_glass_lines=[]
        for ent in msp.query("LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE"):
            lyr_up = ent.dxf.get("layer","0").upper()
            if   lyr_up in WALL:  process_entity_v11(ent, raw_wall_lines,  draw_scale, True)
            elif lyr_up in GLASS: process_entity_v11(ent, raw_glass_lines, draw_scale, False)
            elif lyr_up in FURN:  process_entity_v11(ent, raw_furn_lines,  draw_scale, False)
        all_segs_d = [LineString(l) for l in raw_wall_lines]

    if use_glass and not wall_segs:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")
    if not use_glass and not raw_wall_lines:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")

    furn_raw = process_furniture_to_objects(raw_furn_lines)
    extracted_objects_serial = [{
        "object_id":obj["object_id"],"length":obj["length"],"width":obj["width"],
        "center_x":obj["center_x"],"center_y":obj["center_y"],
        "pt_xy":(obj["point"].x,obj["point"].y),
    } for obj in furn_raw]

    if use_glass:
        accepted,bridges_used,wall_sn,glass_sn = detect_rooms_mode_a(
            wall_segs=wall_segs,glass_segs=glass_segs,arc_segs=arc_segs,
            snap_tol=snap_tol,bridge_tol=bridge_tol,
            glass_edge_thresh=glass_edge_thresh,glass_proximity_mult=glass_proximity_mult,
            min_area_m2=min_area_m2,max_area_m2=max_area_m2,
            min_compact=min_compact,max_aspect=max_aspect_a,
            unit_factor=unit_factor,log=None)
        wall_cavities=[]
        rooms_serial=[]
        for i,(poly,gf,is_glass) in enumerate(accepted):
            minx,miny,maxx,maxy=poly.bounds
            obj_ids=[o["object_id"] for o in extracted_objects_serial
                     if poly.buffer(10).covers(Point(o["pt_xy"]))]
            rooms_serial.append({
                "name":f"Room {i+1}","room_id":f"R{i+1}","poly_wkb":poly.wkb,
                "gf":gf,"is_glass":is_glass,"width":maxx-minx,"height":maxy-miny,
                "area":poly.area,"objects_inside":obj_ids,
            })
    else:
        room_list,wall_cavities = detect_rooms_mode_b(
            raw_wall_lines=raw_wall_lines,
            extracted_objects=[{**o,"point":Point(o["pt_xy"])} for o in extracted_objects_serial],
            gap_close_tol=gap_close_tol,max_door_width=max_door_width,
            min_wall_len=min_wall_len,min_area_m2=min_area_m2,max_area_m2=max_area_m2,
            unit_factor=unit_factor,outer_area_pct=outer_area_pct_b,
            exclude_stairs=exclude_stairs,stair_parallel_min=int(stair_parallel_min),
            stair_angle_tol=float(stair_angle_tol),max_stair_area_m2=float(max_stair_area_m2),
            min_solidity=float(min_solidity),max_aspect_ratio=float(max_aspect_b),
            max_interior_walls=int(max_interior_walls),log=None)
        bridges_used=[]; wall_sn=[]; glass_sn=[]
        rooms_serial=[]
        for r in room_list:
            poly=r["polygon"]
            obj_ids=[o["object_id"] for o in extracted_objects_serial
                     if poly.buffer(10).covers(Point(o["pt_xy"]))]
            rooms_serial.append({
                "name":r["name"],"room_id":r["room_id"],"poly_wkb":poly.wkb,
                "gf":0.0,"is_glass":False,"width":r["width"],"height":r["height"],
                "area":r["area"],"objects_inside":obj_ids,
            })

    return {
        "rooms_serial":rooms_serial,"unit_factor":unit_factor,"unit_div":unit_div,
        "use_glass":use_glass,"mode_label":mode_label,"coverage_pct":coverage_pct,"force_c":False,
        "wall_sn_wkb":[ls.wkb for ls in wall_sn],"glass_sn_wkb":[ls.wkb for ls in glass_sn],
        "bridges_wkb":[ls.wkb for ls in bridges_used],
        "wall_cavities_wkb":[p.wkb for p in wall_cavities if p.geom_type=="Polygon"],
        "all_segs_wkb":[ls.wkb for ls in all_segs_d[:8000]],
        "raw_wall_lines":raw_wall_lines[:8000],"extracted_objects":extracted_objects_serial,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 5 — Mode badge display (replace existing mode_badge block in app.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH_5_BADGE = """
mode_badge = (
    "<span class='mode-d-badge'>Mode D: Closed polyline</span>"
    if "Mode D" in mode_label else
    "<span class='mode-c-badge'>Mode C: All-layer fallback</span>"
    if is_mode_c else "")
st.markdown(f"**Detection mode**: {mode_label} {mode_badge}",
            unsafe_allow_html=True)
"""