"""
app.py — CAD Room Extractor + Heat Load Calculator  v15
========================================================
Run:  streamlit run app.py
Needs: geometry_engine.py  heat_load.py  (same folder)

HOW RE-RUNS ARE PREVENTED
──────────────────────────
Streamlit re-executes the whole script on every widget interaction.
Three @st.cache_data functions make that harmless for the heavy work:

  ① run_oda_conversion(dwg_bytes, dwg_name, oda_exe)
        key: file content bytes + filename + oda path
        → ODA subprocess runs ONCE per unique DWG upload

  ② scan_dxf_layers(dxf_path)
        key: path returned by ①
        → DXF header/layer scan runs ONCE

  ③ detect_rooms_cached(dxf_path, …all layer & detection params…)
        key: path + every sidebar param that affects detection
        → geometry extraction + room detection run ONCE per config

Room multiselect + OK button only touch the TR-filter logic at the
bottom of the script — they never change any cache key, so ①②③ are
all served from cache with zero re-computation.

SERIALISATION
─────────────
st.cache_data requires all return values to be pickle-able.
Shapely geometry objects are NOT directly pickle-able across some
envs; we store them as WKB bytes (geometry.wkb) inside the cache
and deserialise back to Shapely on every rerun (fast, ~µs per object).
Furniture Point objects are stored as (x, y) tuples, not Shapely Points.
"""

import subprocess
import os

import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point
from shapely import wkb as shapely_wkb

from geometry_engine import (
    extract_all_v8,
    process_entity_v11,
    detect_rooms_mode_a,
    detect_rooms_mode_b,
    process_furniture_to_objects,
)
from heat_load import (
    compute_room_heat_loads,
    summarise_heat_loads,
    display_columns,
)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CAD Room Extractor", layout="wide")
st.markdown("""
<style>
  .main { background:#0f1117 }
  h1    { color:#00d4ff; font-family:'Courier New',monospace }
  .block-container { padding-top:2rem }
  div[data-testid="metric-container"] {
    background:#1a1f2e; border-radius:10px;
    padding:10px; border:1px solid #00d4ff33 }
  /* OK button style */
  button[kind="primary"] {
    background:#00d4ff22 !important; color:#00d4ff !important;
    border:1px solid #00d4ff88 !important; border-radius:8px !important;
    font-weight:700 !important }
</style>
""", unsafe_allow_html=True)

st.title(" ICAD Room Extractor ")


# ─────────────────────────────────────────────────────────────────────────────
#  ① CACHED — ODA DWG → DXF
#     Re-runs only when: file bytes, filename, or oda_exe path change.
#     Room selection NEVER changes these → ODA is never called again.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_oda_conversion(dwg_bytes: bytes, dwg_name: str, oda_exe: str) -> str:
    """Returns absolute path to produced DXF. Raises RuntimeError on failure."""
    work_dir   = os.path.join(os.getcwd(), "_dwg_work")
    dxf_folder = os.path.join(os.getcwd(), "_dxf_out")
    os.makedirs(work_dir,   exist_ok=True)
    os.makedirs(dxf_folder, exist_ok=True)

    dwg_path = os.path.join(work_dir, dwg_name)
    with open(dwg_path, "wb") as fh:
        fh.write(dwg_bytes)

    try:
        subprocess.run(
            [oda_exe, work_dir, dxf_folder, "ACAD2013", "DXF", "0", "1"],
            check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ODA conversion failed:\n{e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "ODAFileConverter.exe not found — check the path in the sidebar.")

    stem    = dwg_name.rsplit(".", 1)[0]
    dxf_out = os.path.join(dxf_folder, stem + ".dxf")
    if not os.path.exists(dxf_out):
        hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
        if not hits:
            raise RuntimeError("ODA produced no DXF file.")
        dxf_out = os.path.join(dxf_folder, hits[0])

    return dxf_out


# ─────────────────────────────────────────────────────────────────────────────
#  ② CACHED — DXF layer scan
#     Re-runs only when dxf_path changes.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def scan_dxf_layers(dxf_path: str):
    """Returns (sorted_layers list, entity_count dict, all_layer_names set)."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    all_lyrs = {layer.dxf.name for layer in doc.layers}
    count    = {}
    for ent in msp:
        lyr = ent.dxf.get("layer", "0")
        count[lyr] = count.get(lyr, 0) + 1
    sorted_lyrs = sorted(all_lyrs, key=lambda l: count.get(l, 0), reverse=True)
    return sorted_lyrs, count, all_lyrs


# ─────────────────────────────────────────────────────────────────────────────
#  ③ CACHED — geometry extraction + room detection
#     Re-runs only when dxf_path OR any detection parameter changes.
#     Room multiselect, OK button, heat-load params (H/U/DT/people) are NOT
#     in the signature → changing them never invalidates this cache.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_rooms_cached(
    dxf_path: str,
    # Layer sets — passed as frozenset so they are hashable
    wall_layers: frozenset,
    glass_layers: frozenset,
    furn_layers: frozenset,
    mode_override: str,
    # Mode A
    snap_tol: float, bridge_tol: float,
    glass_edge_thresh: float, glass_proximity_mult: float,
    min_area_m2: float, max_area_m2: float,
    min_compact: float, max_aspect_a: float,
    # Mode B
    outer_area_pct_b: float, min_solidity: float, max_aspect_b: float,
    max_interior_walls: int, exclude_stairs: bool,
    stair_parallel_min: int, stair_angle_tol: float, max_stair_area_m2: float,
    gap_close_tol: float, max_door_width: float, min_wall_len: float,
) -> dict:
    """
    Full extraction + detection pipeline.
    All Shapely geometry is serialised to WKB bytes so st.cache_data can
    pickle the result.  Furniture Point objects are stored as (x,y) tuples.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    WALL  = wall_layers
    GLASS = glass_layers
    FURN  = furn_layers

    # ── detect mode ──────────────────────────────────────────────────────────
    all_ent_lyrs = {ent.dxf.get("layer", "0").upper() for ent in msp}
    glass_found  = bool(GLASS & all_ent_lyrs)
    if not glass_found and GLASS:
        for ent in msp:
            if ent.dxftype() == "INSERT":
                try:
                    bname = ent.dxf.name
                    if bname in doc.blocks:
                        for sub in doc.blocks[bname]:
                            if sub.dxf.get("layer", "0").upper() in GLASS:
                                glass_found = True; break
                except Exception:
                    pass
            if glass_found:
                break

    if   mode_override == "Force Glass-partition (Mode A)": use_glass = True
    elif mode_override == "Force Base-layer only (Mode B)": use_glass = False
    else:                                                    use_glass = bool(GLASS) and glass_found

    mode_label = "🔵 Mode A: Glass-partition" if use_glass else "🟩 Mode B: Base-layer"

    # ── scale (Mode B) ───────────────────────────────────────────────────────
    insunits   = doc.header.get("$INSUNITS", 0)
    scale_map  = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}
    draw_scale = scale_map.get(insunits, 1.0)

    # ── extraction ───────────────────────────────────────────────────────────
    raw_furn_lines = []

    if use_glass:
        ALLOWED_A  = WALL | GLASS
        raw_a      = extract_all_v8(msp, doc, ALLOWED_A, GLASS)
        wall_segs  = [g for (g, ig, ia) in raw_a if not ig and not ia]
        glass_segs = [g for (g, ig, ia) in raw_a if ig  and not ia]
        arc_segs   = [g for (g, ig, ia) in raw_a if ia]
        all_segs_d = [g for (g, _, _)   in raw_a]

        all_xs = [c[0] for ls in wall_segs + glass_segs for c in ls.coords]
        if not all_xs:
            raise RuntimeError("No wall/glass geometry found. Check layer names.")
        span        = max(all_xs) - min(all_xs)
        unit_factor = 1_000_000 if span > 500 else 1
        unit_div    = 1000      if span > 500 else 1
        raw_wall_lines = []

    else:
        wall_segs = glass_segs = arc_segs = []; all_segs_d = []
        raw_wall_lines = []; raw_glass_lines = []

        for ent in msp.query("LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE"):
            lyr_up = ent.dxf.get("layer", "0").upper()
            if   lyr_up in WALL:  process_entity_v11(ent, raw_wall_lines,  draw_scale, True)
            elif lyr_up in GLASS: process_entity_v11(ent, raw_glass_lines, draw_scale, False)
            elif lyr_up in FURN:  process_entity_v11(ent, raw_furn_lines,  draw_scale, False)

        unit_factor = 1_000_000.0; unit_div = 1_000.0
        all_segs_d  = [LineString(l) for l in raw_wall_lines]

    if use_glass and not wall_segs:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")
    if not use_glass and not raw_wall_lines:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")

    # ── furniture (store as tuples — no Shapely in cache) ────────────────────
    furn_raw = process_furniture_to_objects(raw_furn_lines)
    # Serialise Point → (x, y) tuple
    extracted_objects_serial = []
    for obj in furn_raw:
        extracted_objects_serial.append({
            "object_id": obj["object_id"],
            "length":    obj["length"],
            "width":     obj["width"],
            "center_x":  obj["center_x"],
            "center_y":  obj["center_y"],
            "pt_xy":     (obj["point"].x, obj["point"].y),   # tuple, not Point
        })

    # ── room detection ───────────────────────────────────────────────────────
    if use_glass:
        accepted, bridges_used, wall_sn, glass_sn = detect_rooms_mode_a(
            wall_segs=wall_segs, glass_segs=glass_segs, arc_segs=arc_segs,
            snap_tol=snap_tol, bridge_tol=bridge_tol,
            glass_edge_thresh=glass_edge_thresh,
            glass_proximity_mult=glass_proximity_mult,
            min_area_m2=min_area_m2, max_area_m2=max_area_m2,
            min_compact=min_compact, max_aspect=max_aspect_a,
            unit_factor=unit_factor, log=None)
        wall_cavities = []

        rooms_serial = []
        for i, (poly, gf, is_glass) in enumerate(accepted):
            minx, miny, maxx, maxy = poly.bounds
            obj_ids = []
            buf = poly.buffer(10)
            for obj in extracted_objects_serial:
                if buf.covers(Point(obj["pt_xy"])):
                    obj_ids.append(obj["object_id"])
            rooms_serial.append({
                "name":    f"Room {i+1}", "room_id": f"R{i+1}",
                "poly_wkb": poly.wkb,
                "gf":       gf,  "is_glass": is_glass,
                "width":    maxx - minx, "height": maxy - miny,
                "area":     poly.area,
                "objects_inside": obj_ids,
            })

    else:
        room_list, wall_cavities = detect_rooms_mode_b(
            raw_wall_lines=raw_wall_lines,
            extracted_objects=[{**o, "point": Point(o["pt_xy"])}
                               for o in extracted_objects_serial],
            gap_close_tol=gap_close_tol, max_door_width=max_door_width,
            min_wall_len=min_wall_len, min_area_m2=min_area_m2,
            max_area_m2=max_area_m2, unit_factor=unit_factor,
            outer_area_pct=outer_area_pct_b, exclude_stairs=exclude_stairs,
            stair_parallel_min=int(stair_parallel_min),
            stair_angle_tol=float(stair_angle_tol),
            max_stair_area_m2=float(max_stair_area_m2),
            min_solidity=float(min_solidity),
            max_aspect_ratio=float(max_aspect_b),
            max_interior_walls=int(max_interior_walls),
            log=None)
        bridges_used = []; wall_sn = []; glass_sn = []

        rooms_serial = []
        for r in room_list:
            poly = r["polygon"]
            obj_ids = []
            buf = poly.buffer(10)
            for obj in extracted_objects_serial:
                if buf.covers(Point(obj["pt_xy"])):
                    obj_ids.append(obj["object_id"])
            rooms_serial.append({
                "name":    r["name"], "room_id": r["room_id"],
                "poly_wkb": poly.wkb,
                "gf":       0.0,  "is_glass": False,
                "width":    r["width"], "height": r["height"],
                "area":     r["area"],
                "objects_inside": obj_ids,
            })

    # ── serialise all Shapely objects to WKB bytes ───────────────────────────
    return {
        "rooms_serial":       rooms_serial,          # list[dict] with poly_wkb bytes
        "unit_factor":        unit_factor,
        "unit_div":           unit_div,
        "use_glass":          use_glass,
        "mode_label":         mode_label,
        "wall_sn_wkb":        [ls.wkb for ls in wall_sn],
        "glass_sn_wkb":       [ls.wkb for ls in glass_sn],
        "bridges_wkb":        [ls.wkb for ls in bridges_used],
        "wall_cavities_wkb":  [p.wkb  for p in wall_cavities
                                if p.geom_type == "Polygon"],
        "all_segs_wkb":       [ls.wkb for ls in all_segs_d[:8000]],
        "raw_wall_lines":     raw_wall_lines[:8000],
        "extracted_objects":  extracted_objects_serial,   # tuples, no Shapely
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (all values read once per rerun — cheap)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    oda_path = st.text_input(
        "ODAFileConverter Path",
        r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")

    st.divider()
    st.markdown("**Layers**")
    wall_input  = st.text_area("Wall / Base layers",
        "MAIN\nMAIN-4\nCEN-1\nA-WALL\nWALLS\nWall\nWalls\nBase\n0")
    glass_input = st.text_area("Glass layers  (blank = Mode B)", "GLASS")
    furn_input  = st.text_area("Furniture / Door layers (optional)",
        "Furniture\nDoors\nFURNITURE\nDOOR")

    st.divider()
    st.markdown("**Detection Mode**")
    mode_override = st.selectbox("Mode", [
        "Auto-detect",
        "Force Glass-partition (Mode A)",
        "Force Base-layer only (Mode B)"], index=0)

    st.divider()
    st.markdown("**Heat Load Parameters**")
    H               = st.number_input("Room Height (m)",  value=3.0,  step=0.1)
    U_wall          = st.number_input("Wall U-Value",     value=1.8,  step=0.1)
    U_glass         = st.number_input("Glass U-Value",    value=5.8,  step=0.1)
    DT              = st.number_input("DT (deg C)",       value=10,   step=1)
    people_per_room = st.number_input("People / Room",    value=2,    step=1)
    Q_person        = st.number_input("Heat / Person (W)",value=75,   step=5)

    st.divider()
    st.markdown("**Room Filtering**")
    min_area_m2 = st.number_input("Min Room Area (m2)",  value=2.0,   step=0.5)
    max_area_m2 = st.number_input("Max Room Area (m2)",  value=300.0, step=10.0)

    st.divider()
    st.markdown("**Shape Quality — Mode A**")
    min_compact  = st.number_input("Min Compactness",  value=0.04, step=0.01)
    max_aspect_a = st.number_input("Max Aspect Ratio", value=10.0, step=0.5)

    st.divider()
    st.markdown("**Gap Closing — Mode A**")
    snap_tol   = st.number_input("Snap tolerance",   value=10.0, step=1.0)
    bridge_tol = st.number_input("Bridge tolerance", value=80.0, step=5.0)

    st.divider()
    st.markdown("**Glass Detection — Mode A**")
    glass_edge_thresh    = st.number_input("Glass edge threshold (0–1)", value=0.15, step=0.05)
    glass_proximity_mult = st.number_input("Glass proximity multiplier", value=3.0,  step=0.5,
                                            min_value=1.0, max_value=10.0)

    st.divider()
    st.markdown("**Shape Validation — Mode B**")
    outer_area_pct_b   = st.number_input("Outer envelope threshold (%)", value=25.0, step=5.0)
    min_solidity       = st.number_input("Min solidity (0–1)",           value=0.50, step=0.05,
                                          min_value=0.0, max_value=1.0)
    max_aspect_b       = st.number_input("Max aspect ratio (Mode B)",    value=15.0, step=1.0)
    max_interior_walls = st.number_input("Max interior wall segments",   value=8,    step=1)
    exclude_stairs     = st.checkbox("Exclude staircase regions", value=True)
    stair_parallel_min = st.number_input("Min parallel lines → stair", value=4,    step=1)
    stair_angle_tol    = st.number_input("Stair angle tolerance (deg)", value=8.0,  step=1.0)
    max_stair_area_m2  = st.number_input("Max staircase area (m2)",     value=20.0, step=1.0)

    st.divider()
    st.markdown("**Bridging — Mode B**")
    gap_close_tol  = st.number_input("Snap / merge tolerance (mm)",  value=15.0,   step=5.0)
    max_door_width = st.number_input("Max door / archway width (mm)",value=1500.0, step=100.0)
    min_wall_len   = st.number_input("Min wall segment (mm)",        value=200.0,  step=50.0)

    st.divider()
    show_raw = st.checkbox("Show raw geometry overlay", value=False)


# ═════════════════════════════════════════════════════════════════════════════
#  FILE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
if not uploaded_file:
    st.info("👆 Upload a DWG file to begin.")
    st.stop()

dwg_bytes = uploaded_file.getvalue()   # bytes — stable hash key
dwg_name  = uploaded_file.name


# ═════════════════════════════════════════════════════════════════════════════
#  ① ODA CONVERSION  — cached, never re-runs on room selection
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Converting DWG → DXF…"):
    try:
        dxf_path = run_oda_conversion(dwg_bytes, dwg_name, oda_path)
    except RuntimeError as e:
        st.error(str(e)); st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  ② LAYER SCAN — cached
# ═════════════════════════════════════════════════════════════════════════════
sorted_layers, layer_entity_count, all_layers_in_file = scan_dxf_layers(dxf_path)
active_layers = [l for l in sorted_layers if layer_entity_count.get(l, 0) > 0]


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER RESOLUTION  (runs every rerun — cheap dict ops only)
# ═════════════════════════════════════════════════════════════════════════════
def _parse_layers(text):
    return {l.strip().upper() for l in text.strip().splitlines() if l.strip()}

WALL_LAYERS  = _parse_layers(wall_input)
GLASS_LAYERS = _parse_layers(glass_input)
FURN_LAYERS  = _parse_layers(furn_input)

matched_wall  = WALL_LAYERS  & {x.upper() for x in all_layers_in_file}
matched_glass = GLASS_LAYERS & {x.upper() for x in all_layers_in_file}
matched_furn  = FURN_LAYERS  & {x.upper() for x in all_layers_in_file}

matched_wall_ents = sum(layer_entity_count.get(l, 0)
    for l in all_layers_in_file if l.upper() in matched_wall)
poor_match = matched_wall_ents < (sum(layer_entity_count.values()) or 1) * 0.20

NON_WALL_KW = {"furniture","plant","planter","text","vp","defpoint",
               "dimension","dim","hatch","annotation","title","border",
               "viewport","electrical","elec","plumbing","mech","door"}

# ── Layer Selection expander (unchanged per requirement) ─────────────────────
with st.expander("🔍 Layer Selection", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Layers in DXF (by entity count):**")
        for lyr in sorted_layers:
            cnt = layer_entity_count.get(lyr, 0)
            tag = ("🟩 WALL"  if lyr.upper() in WALL_LAYERS
              else ("🟦 GLASS" if lyr.upper() in GLASS_LAYERS
              else ("🪑 FURN"  if lyr.upper() in FURN_LAYERS else "⚪")))
            st.markdown(f"{tag} `{lyr}` — {cnt} entities")
    with col2:
        if matched_wall and not poor_match:
            st.success(f"Wall layers matched: {sorted(matched_wall)} ✅")
        else:
            st.warning("⚠️ Poor wall layer match — select below")
        if matched_glass:
            st.success(f"Glass layers matched: {sorted(matched_glass)} ✅")
        else:
            st.warning(f"⚠️ Glass layer(s) {sorted(GLASS_LAYERS)} not found in file")

        smart_def    = [l for l in active_layers
                        if not any(kw in l.lower() for kw in NON_WALL_KW)]
        wall_default  = [l for l in active_layers if l.upper() in matched_wall] or \
                        (smart_def if poor_match else [])
        glass_default = [l for l in active_layers if l.upper() in matched_glass]
        furn_default  = [l for l in active_layers if l.upper() in matched_furn]

        sel_wall  = st.multiselect("Wall / Base layers",
            options=active_layers, default=wall_default,  key="sel_wall")
        sel_glass = st.multiselect("Glass layers (optional)",
            options=active_layers, default=glass_default, key="sel_glass")
        sel_furn  = st.multiselect("Furniture / Door layers (optional)",
            options=active_layers, default=furn_default,  key="sel_furn")

        # Override only if user made a non-empty selection
        if sel_wall:  WALL_LAYERS  = {l.upper() for l in sel_wall}
        if sel_glass: GLASS_LAYERS = {l.upper() for l in sel_glass}
        if sel_furn:  FURN_LAYERS  = {l.upper() for l in sel_furn}

        st.caption(f"Active: wall={sorted(WALL_LAYERS)} | glass={sorted(GLASS_LAYERS)}")


# ═════════════════════════════════════════════════════════════════════════════
#  ③ ROOM DETECTION — cached
#     frozenset() makes the layer sets hashable for the cache key.
#     Heat-load params (H, U_wall, DT, people…) are intentionally NOT here —
#     they don't affect detection, only TR math.
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("🔲 Extracting geometry and detecting rooms…"):
    try:
        result = detect_rooms_cached(
            dxf_path          = dxf_path,
            wall_layers       = frozenset(WALL_LAYERS),
            glass_layers      = frozenset(GLASS_LAYERS),
            furn_layers       = frozenset(FURN_LAYERS),
            mode_override     = mode_override,
            snap_tol          = float(snap_tol),
            bridge_tol        = float(bridge_tol),
            glass_edge_thresh = float(glass_edge_thresh),
            glass_proximity_mult = float(glass_proximity_mult),
            min_area_m2       = float(min_area_m2),
            max_area_m2       = float(max_area_m2),
            min_compact       = float(min_compact),
            max_aspect_a      = float(max_aspect_a),
            outer_area_pct_b  = float(outer_area_pct_b),
            min_solidity      = float(min_solidity),
            max_aspect_b      = float(max_aspect_b),
            max_interior_walls= int(max_interior_walls),
            exclude_stairs    = bool(exclude_stairs),
            stair_parallel_min= int(stair_parallel_min),
            stair_angle_tol   = float(stair_angle_tol),
            max_stair_area_m2 = float(max_stair_area_m2),
            gap_close_tol     = float(gap_close_tol),
            max_door_width    = float(max_door_width),
            min_wall_len      = float(min_wall_len),
        )
    except RuntimeError as e:
        st.error(str(e)); st.stop()


# ─── Deserialise WKB back to Shapely objects (µs per object) ─────────────────
USE_GLASS_MODE = result["use_glass"]
mode_label     = result["mode_label"]
unit_factor    = result["unit_factor"]
unit_div       = result["unit_div"]

wall_sn       = [shapely_wkb.loads(b) for b in result["wall_sn_wkb"]]
glass_sn      = [shapely_wkb.loads(b) for b in result["glass_sn_wkb"]]
bridges_used  = [shapely_wkb.loads(b) for b in result["bridges_wkb"]]
wall_cavities = [shapely_wkb.loads(b) for b in result["wall_cavities_wkb"]]
all_segs_d    = [shapely_wkb.loads(b) for b in result["all_segs_wkb"]]
raw_wall_lines = result["raw_wall_lines"]

# Restore rooms — add Polygon object alongside the WKB
rooms_unified = []
for r in result["rooms_serial"]:
    rooms_unified.append({**r, "polygon": shapely_wkb.loads(r["poly_wkb"])})

# Restore furniture — add Point alongside (x,y) tuple
extracted_objects = []
for o in result["extracted_objects"]:
    extracted_objects.append({**o, "point": Point(o["pt_xy"])})

n_rooms = len(rooms_unified)
n_glass = sum(1 for r in rooms_unified if r["is_glass"])
n_wall  = n_rooms - n_glass

if n_rooms == 0:
    st.error("No rooms detected. Try increasing Bridge tolerance or decreasing Min Area.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  HEAT LOAD — all rooms  (runs every rerun but is microseconds)
# ═════════════════════════════════════════════════════════════════════════════
room_data   = compute_room_heat_loads(
    rooms_unified, unit_factor, unit_div,
    H=float(H), U_wall=float(U_wall), U_glass=float(U_glass), DT=float(DT),
    people_per_room=int(people_per_room), Q_person=float(Q_person))

summary_all = summarise_heat_loads(room_data)
disp_cols   = display_columns()
df          = pd.DataFrame(room_data)


# ═════════════════════════════════════════════════════════════════════════════
#  STATUS
# ═════════════════════════════════════════════════════════════════════════════
st.success(
    f"✅  **{n_rooms} rooms detected** — "
    f"{n_glass} glass · {n_wall} wall  |  {mode_label}  |  "
    f"Total **{summary_all['total_tr']:.2f} TR** / **{summary_all['total_kw']:.2f} kW**")


# ═════════════════════════════════════════════════════════════════════════════
#  FLOOR PLAN
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🗺️ Room Detection")

fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333")

if show_raw:
    for ls in all_segs_d:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#1e3a50", lw=0.25, alpha=0.35)

if USE_GLASS_MODE:
    for ls in wall_sn:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)
    for ls in glass_sn:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#00d4ff", lw=1.4, alpha=0.75, zorder=2)
else:
    for l in raw_wall_lines:
        ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]],
                color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

for br in bridges_used:
    xs, ys = br.xy; ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.6, zorder=2)
for cav in wall_cavities:
    xs, ys = cav.exterior.xy; ax.fill(xs, ys, color="dimgray", alpha=0.8, zorder=2)

for row in room_data:
    poly  = row["_poly"]; color = row["_color"]
    ig    = row["_is_glass"]; gf   = row["_gf"]
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, alpha=0.40 if ig else 0.22, color=color, zorder=3)
    ax.plot(xs, ys, color="#00d4ff" if ig else color,
            lw=2.5 if ig else 1.6, zorder=4)
    cx, cy = poly.centroid.x, poly.centroid.y
    label  = f"{'🔵' if ig else '🟩'} {row['Room']}\n{row['Area (m2)']} m2\nTR: {row['TR']}"
    if gf > 0.05:
        label += f"\n{row['Glass % edge']}% glass"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5,
            color="white", fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#000000dd",
                      edgecolor="#00d4ff" if ig else color,
                      linewidth=1.3 if ig else 0.5))

for obj in extracted_objects:
    ax.plot(obj["center_x"], obj["center_y"], "ro", markersize=4, zorder=6)

ax.set_aspect("equal")
ax.set_title(f"{n_rooms} Rooms  ({n_wall} wall · {n_glass} glass)  |  {mode_label}",
             color="#00d4ff", fontfamily="monospace", fontsize=10)
ax.grid(True, color="#1a2a3a", lw=0.25)
legend_items = [mpatches.Patch(color="#3a6a8a", label="Wall lines")]
if bridges_used:
    legend_items.append(mpatches.Patch(color="#ff4444", label="Gap bridges"))
if USE_GLASS_MODE:
    legend_items += [mpatches.Patch(color="#00d4ff", label="Glass lines"),
                     mpatches.Patch(color="#00d4ff", alpha=0.4, label="Glass room")]
ax.legend(handles=legend_items, loc="upper right",
          facecolor="#0f1117", edgecolor="#444", labelcolor="white", fontsize=9)
st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  ROOM MEASUREMENTS TABLE  (all rooms — always shown)
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Room Measurements & Heat Load")
st.caption("Area and Perimeter are from actual polygon geometry, not bounding box.")
st.dataframe(
    df[disp_cols].style
    .format({"TR": "{:.3f}", "Area (m2)": "{:.3f}", "Perimeter (m)": "{:.3f}",
             "Q_total (W)": "{:.1f}", "Glass % edge": "{:.1f}"}),
    use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ROOM SELECTION  +  OK BUTTON
#  ──────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔽 Heat Load Filter")

all_room_names = [r["Room"] for r in room_data]

# Initialise confirmed selection on very first load
if "confirmed_rooms" not in st.session_state:
    st.session_state["confirmed_rooms"] = []

# ── multiselect + Calculate button side by side ───────────────────────────────
sel_col, btn_col = st.columns([5, 1])

with sel_col:
    # `key="room_multiselect"` binds the widget to session_state automatically.
    # We do NOT set `default=` here — the value is managed purely via session_state.
    st.session_state.setdefault("room_multiselect", [])
    live_selection = st.multiselect(
        "Select Rooms  (leave empty = All Rooms)",
        options=all_room_names,
        key="room_multiselect",
        placeholder="Pick rooms, then click  ✅ Calculate TR")

with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)          # align vertically
    ok_clicked = st.button("✅ Calculate TR",
                           key="ok_btn", use_container_width=True, type="primary")

# ── Persist confirmed selection only when button is clicked ──────────────────
if ok_clicked:
    st.session_state["confirmed_rooms"] = list(live_selection)

# ── Derive active selection from confirmed state (not live widget) ────────────
active_selection = st.session_state["confirmed_rooms"]


# ═════════════════════════════════════════════════════════════════════════════
#  HEAT LOAD SUMMARY  (pure Python math — no ODA, no detection)
# ═════════════════════════════════════════════════════════════════════════════
if not active_selection:
    filtered_data   = room_data
    summary         = summary_all
    selection_label = "All Rooms"
    chart_title     = f"TR per Room  |  Total: {summary_all['total_tr']:.2f} TR"
else:
    sel_set         = set(active_selection)
    filtered_data   = [r for r in room_data if r["Room"] in sel_set]
    summary         = summarise_heat_loads(filtered_data)
    n_sel           = len(active_selection)
    selection_label = active_selection[0] if n_sel == 1 else f"{n_sel} rooms selected"
    chart_title     = f"TR — {selection_label}  |  Total: {summary['total_tr']:.2f} TR"

# ── Hint when selection is live but not yet confirmed ────────────────────────
if sorted(live_selection) != sorted(active_selection):
    pending = set(live_selection) - set(active_selection)
    removed = set(active_selection) - set(live_selection)
    parts   = []
    if pending: parts.append(f"+{len(pending)} added")
    if removed: parts.append(f"-{len(removed)} removed")
    if parts:
        st.info(f"ℹ️ Pending changes ({', '.join(parts)}) — click **✅ Calculate TR** to apply.")

st.subheader(f"🌡️ Heat Load Summary — {selection_label}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏠 Rooms",       summary["n_rooms"])
c2.metric("🔵 Glass Rooms", summary["n_glass"])
c3.metric("📐 Total Area",  f"{summary['total_area_m2']:.2f} m2")
c4.metric("⚡ Total Load",  f"{summary['total_kw']:.2f} kW")
c5.metric("❄️ Total TR",    f"{summary['total_tr']:.2f} TR")




# ═════════════════════════════════════════════════════════════════════════════
#  DOWNLOADS
# ═════════════════════════════════════════════════════════════════════════════
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Download All Rooms (CSV)",
        df[disp_cols].to_csv(index=False),
        "rooms_heat_load_all.csv", "text/csv")
with col_dl2:
    if active_selection and filtered_data:
        safe = selection_label.replace(" ", "_").replace("/", "-")
        st.download_button(
            "⬇️ Download Selected Rooms (CSV)",
            pd.DataFrame(filtered_data)[disp_cols].to_csv(index=False),
            f"heat_load_{safe}.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    f"{mode_label}  |  "
    f"Wall: {', '.join(sorted(WALL_LAYERS))}  |  "
    f"Glass: {', '.join(sorted(GLASS_LAYERS)) or 'none'}  |  "
    f"{n_rooms} rooms ({n_glass} glass)  |  "
    f"Total: {summary_all['total_tr']:.2f} TR")