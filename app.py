"""
app.py — CAD Room Extractor + Heat Load + CFM + Duct Layout  v20
================================================================
v20 Changes (on top of v19):
  - NEW: Section ⑩ — Duct Layout Generation
      * AHU auto-placed at centroid of selected rooms
      * Branch ducts routed (rectilinear L-shape) from AHU to every room
      * Duct sized via Equal Friction / SMACNA velocity method
      * Floor plan rendered with duct overlay (colour-coded by duct size)
      * Downloads: Duct Floor Plan PNG, DXF (CAD-ready), Sizing CSV, BOQ CSV

Run:  streamlit run app.py
Needs: geometry_engine.py  heat_load.py  duct_engine.py  (same folder)
"""

import subprocess, os, json, re, io, base64
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point
from shapely import wkb as shapely_wkb

from geometry_engine import (
    extract_all_v8, process_entity_v11,
    detect_rooms_mode_a, detect_rooms_mode_b,
    process_furniture_to_objects,
)
from heat_load import (
    compute_room_heat_loads, summarise_heat_loads, display_columns,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CFM
# ─────────────────────────────────────────────────────────────────────────────
BTU_PER_WATT = 3.412
CFM_FACTOR   = 1.08

def watts_to_cfm(q_watts: float, dt_celsius: float) -> float:
    dt_f = dt_celsius * 9 / 5
    if dt_f <= 0:
        return 0.0
    return (q_watts * BTU_PER_WATT) / (CFM_FACTOR * dt_f)

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
  button[kind="primary"] {
    background:#00d4ff22 !important; color:#00d4ff !important;
    border:1px solid #00d4ff88 !important; border-radius:8px !important;
    font-weight:700 !important }

  .result-box {
    background:linear-gradient(135deg,#0d1b2a,#111827);
    border:1px solid #00d4ff33; border-radius:12px;
    padding:18px 20px; text-align:center; height:100%;
  }
  .result-label { font-size:11px; color:#556677; text-transform:uppercase;
    letter-spacing:.1em; font-family:'Courier New',monospace; }
  .result-value { font-size:28px; font-weight:900; color:#00d4ff;
    font-family:'Courier New',monospace; line-height:1.1; }
  .result-unit  { font-size:13px; color:#8899aa; }

  .total-bar {
    background:#0d1a0d; border:1px solid #00ff9933;
    border-radius:8px; padding:10px 16px; margin-top:8px;
    font-family:monospace; font-size:13px; color:#aaccaa;
  }
  .sdiv { height:1px; background:linear-gradient(90deg,#00d4ff44,transparent);
    margin:20px 0; }

  [data-testid="stForm"] { border:none !important; padding:0 !important; }
</style>
""", unsafe_allow_html=True)

st.title(" ICAD Room Extractor")

# ─────────────────────────────────────────────────────────────────────────────
#  ROOM TYPE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
ROOM_TYPE_EMOJI = {
    "Hall / Living Room":"🛋️","Bedroom":"🛏️","Master Bedroom":"🛏️",
    "Kitchen":"🍳","Washroom / Bathroom":"🚿","Balcony / Terrace":"🌿",
    "Dining Room":"🍽️","Foyer / Entrance":"🚪","Garage / Parking":"🚗",
    "Storage / Utility":"📦","Staircase":"🪜","Corridor / Passage":"🚶",
    "Conference Hall":"💼","Manager Cabin":"💺","Open Office / Workstation":"🖥️",
    "Reception / Lobby":"📋","Pantry / Cafeteria":"☕","Server Room / IT Room":"🖧",
    "Meeting Room":"📝","Toilet / Restroom":"🚽","HOD Cabin":"💼",
    "Director Office":"🏢","Garden / Landscape":"🌳","Courtyard / Open Area":"☀️",
    "Parking Lot":"🅿️","Other":"❓",
}
ROOM_TYPE_COLOR = {
    "Hall / Living Room":"#4ECDC4","Bedroom":"#45B7D1","Master Bedroom":"#1A78C2",
    "Kitchen":"#F7DC6F","Washroom / Bathroom":"#96CEB4","Balcony / Terrace":"#82E0AA",
    "Dining Room":"#F0B27A","Foyer / Entrance":"#FAD7A0","Garage / Parking":"#AAB7B8",
    "Storage / Utility":"#AED6F1","Staircase":"#F1948A","Corridor / Passage":"#FFEAA7",
    "Conference Hall":"#C39BD3","Manager Cabin":"#A9CCE3",
    "Open Office / Workstation":"#A9DFBF","Reception / Lobby":"#FAD7A0",
    "Pantry / Cafeteria":"#FDEBD0","Server Room / IT Room":"#D5D8DC",
    "Meeting Room":"#D7BDE2","Toilet / Restroom":"#96CEB4",
    "HOD Cabin":"#7FB3D3","Director Office":"#5D6D7E",
    "Garden / Landscape":"#ABEBC6","Courtyard / Open Area":"#F9E79F",
    "Parking Lot":"#BFC9CA","Other":"#556677",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_oda_conversion(dwg_bytes: bytes, dwg_name: str, oda_exe: str) -> str:
    work_dir   = os.path.join(os.getcwd(), "_dwg_work")
    dxf_folder = os.path.join(os.getcwd(), "_dxf_out")
    os.makedirs(work_dir, exist_ok=True)
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
        raise RuntimeError("ODAFileConverter.exe not found — check the path in the sidebar.")
    stem    = dwg_name.rsplit(".", 1)[0]
    dxf_out = os.path.join(dxf_folder, stem + ".dxf")
    if not os.path.exists(dxf_out):
        hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
        if not hits:
            raise RuntimeError("ODA produced no DXF file.")
        dxf_out = os.path.join(dxf_folder, hits[0])
    return dxf_out


@st.cache_data(show_spinner=False)
def scan_dxf_layers(dxf_path: str):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    all_lyrs = {layer.dxf.name for layer in doc.layers}
    count = {}
    for ent in msp:
        lyr = ent.dxf.get("layer", "0")
        count[lyr] = count.get(lyr, 0) + 1
    sorted_lyrs = sorted(all_lyrs, key=lambda l: count.get(l, 0), reverse=True)
    return sorted_lyrs, count, all_lyrs


@st.cache_data(show_spinner=False)
def detect_rooms_cached(
    dxf_path, wall_layers, glass_layers, furn_layers, mode_override,
    snap_tol, bridge_tol, glass_edge_thresh, glass_proximity_mult,
    min_area_m2, max_area_m2, min_compact, max_aspect_a,
    outer_area_pct_b, min_solidity, max_aspect_b,
    max_interior_walls, exclude_stairs,
    stair_parallel_min, stair_angle_tol, max_stair_area_m2,
    gap_close_tol, max_door_width, min_wall_len,
) -> dict:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    WALL = wall_layers; GLASS = glass_layers; FURN = furn_layers

    all_ent_lyrs = {ent.dxf.get("layer", "0").upper() for ent in msp}
    glass_found  = bool(GLASS & all_ent_lyrs)
    if not glass_found and GLASS:
        for ent in msp:
            if ent.dxftype() == "INSERT":
                try:
                    bname = ent.dxf.name
                    if bname in doc.blocks:
                        for sub in doc.blocks[bname]:
                            if sub.dxf.get("layer","0").upper() in GLASS:
                                glass_found = True; break
                except Exception:
                    pass
            if glass_found: break

    if   mode_override == "Force Glass-partition (Mode A)": use_glass = True
    elif mode_override == "Force Base-layer only (Mode B)": use_glass = False
    else: use_glass = bool(GLASS) and glass_found

    mode_label = "🔵 Mode A: Glass-partition" if use_glass else "🟩 Mode B: Base-layer"
    insunits   = doc.header.get("$INSUNITS", 0)
    scale_map  = {0:1.0,1:25.4,2:304.8,4:1.0,5:10.0,6:1000.0}
    draw_scale = scale_map.get(insunits, 1.0)
    raw_furn_lines = []

    if use_glass:
        ALLOWED_A  = WALL | GLASS
        raw_a      = extract_all_v8(msp, doc, ALLOWED_A, GLASS)
        wall_segs  = [g for (g,ig,ia) in raw_a if not ig and not ia]
        glass_segs = [g for (g,ig,ia) in raw_a if ig and not ia]
        arc_segs   = [g for (g,ig,ia) in raw_a if ia]
        all_segs_d = [g for (g,_,_)   in raw_a]
        all_xs     = [c[0] for ls in wall_segs+glass_segs for c in ls.coords]
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
            lyr_up = ent.dxf.get("layer","0").upper()
            if   lyr_up in WALL:  process_entity_v11(ent, raw_wall_lines,  draw_scale, True)
            elif lyr_up in GLASS: process_entity_v11(ent, raw_glass_lines, draw_scale, False)
            elif lyr_up in FURN:  process_entity_v11(ent, raw_furn_lines,  draw_scale, False)
        unit_factor = 1_000_000.0; unit_div = 1_000.0
        all_segs_d  = [LineString(l) for l in raw_wall_lines]

    if use_glass and not wall_segs:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")
    if not use_glass and not raw_wall_lines:
        raise RuntimeError(f"No geometry on wall layers: {sorted(WALL)}")

    furn_raw = process_furniture_to_objects(raw_furn_lines)
    extracted_objects_serial = []
    for obj in furn_raw:
        extracted_objects_serial.append({
            "object_id": obj["object_id"], "length": obj["length"],
            "width": obj["width"], "center_x": obj["center_x"],
            "center_y": obj["center_y"], "pt_xy": (obj["point"].x, obj["point"].y),
        })

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
        rooms_serial  = []
        for i, (poly, gf, is_glass) in enumerate(accepted):
            minx,miny,maxx,maxy = poly.bounds
            obj_ids = [o["object_id"] for o in extracted_objects_serial
                       if poly.buffer(10).covers(Point(o["pt_xy"]))]
            rooms_serial.append({
                "name": f"Room {i+1}", "room_id": f"R{i+1}",
                "poly_wkb": poly.wkb, "gf": gf, "is_glass": is_glass,
                "width": maxx-minx, "height": maxy-miny, "area": poly.area,
                "objects_inside": obj_ids,
            })
    else:
        room_list, wall_cavities = detect_rooms_mode_b(
            raw_wall_lines=raw_wall_lines,
            extracted_objects=[{**o,"point":Point(o["pt_xy"])} for o in extracted_objects_serial],
            gap_close_tol=gap_close_tol, max_door_width=max_door_width,
            min_wall_len=min_wall_len, min_area_m2=min_area_m2,
            max_area_m2=max_area_m2, unit_factor=unit_factor,
            outer_area_pct=outer_area_pct_b, exclude_stairs=exclude_stairs,
            stair_parallel_min=int(stair_parallel_min),
            stair_angle_tol=float(stair_angle_tol),
            max_stair_area_m2=float(max_stair_area_m2),
            min_solidity=float(min_solidity),
            max_aspect_ratio=float(max_aspect_b),
            max_interior_walls=int(max_interior_walls), log=None)
        bridges_used=[]; wall_sn=[]; glass_sn=[]
        rooms_serial=[]
        for r in room_list:
            poly = r["polygon"]
            obj_ids = [o["object_id"] for o in extracted_objects_serial
                       if poly.buffer(10).covers(Point(o["pt_xy"]))]
            rooms_serial.append({
                "name": r["name"], "room_id": r["room_id"],
                "poly_wkb": poly.wkb, "gf": 0.0, "is_glass": False,
                "width": r["width"], "height": r["height"], "area": r["area"],
                "objects_inside": obj_ids,
            })

    return {
        "rooms_serial": rooms_serial, "unit_factor": unit_factor,
        "unit_div": unit_div, "use_glass": use_glass, "mode_label": mode_label,
        "wall_sn_wkb":       [ls.wkb for ls in wall_sn],
        "glass_sn_wkb":      [ls.wkb for ls in glass_sn],
        "bridges_wkb":       [ls.wkb for ls in bridges_used],
        "wall_cavities_wkb": [p.wkb  for p in wall_cavities if p.geom_type=="Polygon"],
        "all_segs_wkb":      [ls.wkb for ls in all_segs_d[:8000]],
        "raw_wall_lines":    raw_wall_lines[:8000],
        "extracted_objects": extracted_objects_serial,
    }


@st.cache_data(show_spinner=False)
def classify_rooms_ai_vision(image_b64: str, rooms_json: str, openai_key: str) -> dict:
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("openai package not installed.")

    rooms_list = json.loads(rooms_json)
    if not rooms_list:
        return {}

    room_desc_lines = []
    for r in rooms_list:
        asp = r["aspect_ratio"]; area = r["area_m2"]
        shape_hint = ("very elongated strip" if asp >= 5 else
                      "elongated rectangle"  if asp >= 3 else
                      "rectangular"          if asp >= 1.8 else "compact/square")
        size_hint = ("tiny (<4m2)"     if area < 4  else
                     "small (4-8m2)"   if area < 8  else
                     "medium (8-20m2)" if area < 20 else
                     "large (20-50m2)" if area < 50 else "very large (>50m2)")
        room_desc_lines.append(
            f"  {r['room_id']}: {size_hint}, {shape_hint}, "
            f"area={r['area_m2']}m2, {r['width_m']}x{r['height_m']}m, "
            f"aspect={r['aspect_ratio']}, glass={int(r['glass_fraction']*100)}%, "
            f"furniture_count={r['object_count']}")

    total_area = sum(r["area_m2"] for r in rooms_list)
    avg_area   = total_area / max(len(rooms_list), 1)
    max_area   = max((r["area_m2"] for r in rooms_list), default=0)
    is_office  = (total_area > 300 or max_area > 80 or
                  (len(rooms_list) > 12 and avg_area > 15))
    plan_hint  = "OFFICE / COMMERCIAL" if is_office else "RESIDENTIAL"

    system_prompt = f"""You are a world-class architectural floor plan analyst.
Detected plan type: {plan_hint}
RESIDENTIAL: Hall/Living Room, Bedroom, Master Bedroom, Kitchen, Washroom/Bathroom,
Balcony/Terrace, Dining Room, Foyer/Entrance, Garage/Parking, Storage/Utility,
Staircase, Corridor/Passage
OFFICE: Conference Hall, Manager Cabin, HOD Cabin, Director Office,
Open Office/Workstation, Reception/Lobby, Meeting Room, Pantry/Cafeteria,
Toilet/Restroom, Server Room/IT Room, Staircase, Corridor/Passage, Storage/Utility
OUTDOOR: Garden/Landscape, Courtyard/Open Area, Parking Lot
RULES: Staircase MUST have stepped lines. Conference Hall needs large table.
Confidence: high=clear evidence, medium=shape inference, low=uncertain."""

    residential_types = ["Hall / Living Room","Bedroom","Master Bedroom","Kitchen",
        "Washroom / Bathroom","Balcony / Terrace","Dining Room","Foyer / Entrance",
        "Garage / Parking","Storage / Utility","Staircase","Corridor / Passage"]
    office_types = ["Conference Hall","Manager Cabin","HOD Cabin","Director Office",
        "Open Office / Workstation","Reception / Lobby","Meeting Room",
        "Pantry / Cafeteria","Toilet / Restroom","Server Room / IT Room",
        "Staircase","Corridor / Passage","Storage / Utility"]
    landscape_types = ["Garden / Landscape","Courtyard / Open Area","Parking Lot"]

    user_text = (
        f"PLAN TYPE: {plan_hint}\n"
        f"Room measurements:\n" + "\n".join(room_desc_lines) +
        f"\nValid RESIDENTIAL: {json.dumps(residential_types)}"
        f"\nValid OFFICE: {json.dumps(office_types)}"
        f"\nOutdoor: {json.dumps(landscape_types)}"
        f'\nFallback: "Other"'
        f'\nReturn ONLY valid JSON, no markdown:'
        f'\n{{"R1":{{"type":"<label>","confidence":"high|medium|low","reason":"<brief>"}},...}}'
        f"\nClassify ALL: {', '.join(r['room_id'] for r in rooms_list)}"
    )

    client = _openai.OpenAI(api_key=openai_key)
    user_content = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
        })
    user_content.append({"type": "text", "text": user_text})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_content}],
        temperature=0.1, max_tokens=3000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$",       "", raw, flags=re.MULTILINE).strip()
    result = json.loads(raw)
    for rid, info in result.items():
        if info.get("type") not in ROOM_TYPE_EMOJI:
            info["type"] = "Other"
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  FLOOR PLAN RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_floorplan_for_vision(rooms_unified, raw_wall_lines, use_glass_mode,
                                wall_sn, glass_sn, extracted_objects=None) -> str:
    FILL = ["#AED6F1","#A9DFBF","#FAD7A0","#F1948A","#D2B4DE",
            "#76D7C4","#F0B27A","#85C1E9","#82E0AA","#F9E79F",
            "#FDEBD0","#D5F5E3","#EBF5FB","#F4ECF7","#EAFAF1",
            "#FEF9E7","#FDEDEC","#EAF2FF","#FFF9C4","#E8DAEF"]
    fig, axes = plt.subplots(1, 2, figsize=(24, 13), dpi=150)
    fig.patch.set_facecolor("#F8F9FA")
    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("white"); ax.set_aspect("equal"); ax.axis("off")
        if use_glass_mode:
            for ls in wall_sn:
                xs, ys = ls.xy; ax.plot(xs, ys, color="#1a1a1a", lw=1.2, alpha=0.9, zorder=1)
            for ls in glass_sn:
                xs, ys = ls.xy; ax.plot(xs, ys, color="#1560BD", lw=1.4, alpha=0.9, zorder=1)
        else:
            for l in raw_wall_lines:
                ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
                        color="#1a1a1a", lw=1.0, alpha=0.85, zorder=1)
        for i, room in enumerate(rooms_unified):
            poly = room["polygon"]; color = FILL[i % len(FILL)]
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.60 if ax_idx==0 else 0.30, zorder=2)
            ax.plot(xs, ys, color="#2c2c2c" if ax_idx==0 else "#555555",
                    lw=1.5 if ax_idx==0 else 1.2, zorder=3)
            cx, cy = poly.centroid.x, poly.centroid.y
            area_m2 = room.get("area", 0)
            fsize = max(7, min(13, 7 + (area_m2 ** 0.4)))
            ax.text(cx, cy, room["room_id"], ha="center", va="center",
                    fontsize=fsize, fontweight="bold", color="#0a0a0a", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=color, alpha=0.92, linewidth=1.5))
        if ax_idx == 1 and extracted_objects:
            for obj in extracted_objects:
                ax.plot(obj["center_x"], obj["center_y"], "o", color="#CC2200",
                        markersize=5, markeredgecolor="#880000",
                        markeredgewidth=0.5, alpha=0.85, zorder=5)
        ax.set_title("LEFT: Room Shapes & Wall Geometry" if ax_idx==0
                     else "RIGHT: Furniture Layout (red dots = fixtures/furniture)",
                     fontsize=10, color="#333", pad=6, fontweight="bold")
    fig.suptitle("Floor Plan — Room IDs for AI Classification",
                 fontsize=12, color="#222", y=1.01, fontweight="bold")
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#F8F9FA", dpi=150)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_floorplan_figure(room_data, raw_wall_lines, USE_GLASS_MODE,
                             wall_sn, glass_sn, bridges_used, wall_cavities,
                             all_segs_d, extracted_objects, show_raw=False):
    """Returns (fig, png_bytes)."""
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
            ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
                    color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

    for br in bridges_used:
        xs, ys = br.xy; ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.6, zorder=2)
    for cav in wall_cavities:
        xs, ys = cav.exterior.xy; ax.fill(xs, ys, color="dimgray", alpha=0.8, zorder=2)

    for row in room_data:
        poly     = row["_poly"]; is_glass = row["_is_glass"]; gf = row["_gf"]
        ai_type  = row["AI Type"]; ai_name  = row["AI Name"]
        color = (ROOM_TYPE_COLOR.get(ai_type, "#888888")
                 if ai_type != "Other" and ai_type in ROOM_TYPE_COLOR
                 else row["_color"])
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.42 if is_glass else 0.25, color=color, zorder=3)
        ax.plot(xs, ys, color="#00d4ff" if is_glass else color,
                lw=2.5 if is_glass else 1.6, zorder=4)
        cx, cy = poly.centroid.x, poly.centroid.y
        raw_name     = ai_name if ai_type != "Other" else row["Room"]
        display_name = re.sub(r'[^\x00-\x7F\s\-/().,:]', '', raw_name).strip() or raw_name
        label = f"{display_name}\n Area:{row['Area (m2)']}"
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5,
                color="white", fontfamily="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#000000dd",
                          edgecolor="#00d4ff" if is_glass else color,
                          linewidth=1.3 if is_glass else 0.8))

    for obj in extracted_objects:
        ax.plot(obj["center_x"], obj["center_y"], "ro", markersize=4, zorder=6)

    ax.set_aspect("equal")
    legend_items = [mpatches.Patch(color="#3a6a8a", label="Wall lines")]
    if bridges_used:
        legend_items.append(mpatches.Patch(color="#ff4444", label="Gap bridges"))
    if USE_GLASS_MODE:
        legend_items += [mpatches.Patch(color="#00d4ff", label="Glass lines"),
                         mpatches.Patch(color="#00d4ff", alpha=0.4, label="Glass room")]
    for rtype in sorted(set(r["AI Type"] for r in room_data if r["AI Type"] != "Other")):
        legend_items.append(mpatches.Patch(
            color=ROOM_TYPE_COLOR.get(rtype, "#888"), alpha=0.7, label=rtype))
    ax.legend(handles=legend_items, loc="upper right",
              facecolor="#0f1117", edgecolor="#444", labelcolor="white", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f1117", dpi=150)
    buf.seek(0); png_bytes = buf.read(); buf.close()
    return fig, png_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT METRIC CARD
# ─────────────────────────────────────────────────────────────────────────────
def _metric_card(col, label, value, unit):
    with col:
        st.markdown(
            f"<div class='result-box'>"
            f"<div class='result-label'>{label}</div>"
            f"<div class='result-value'>{value}"
            f"<span class='result-unit'> {unit}</span></div>"
            f"</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    oda_path = st.text_input("ODAFileConverter Path",
        r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")
    st.divider()
    st.markdown("**🤖 AI Room Recognition**")
    enable_ai = st.checkbox("Enable AI Room Classification", value=True)
    _env_key  = os.environ.get("OPENAI_API_KEY", "").strip()
    if _env_key:
        st.success("✅ OpenAI key loaded from .env")
    else:
        st.warning("⚠️ OPENAI_API_KEY not found in .env")
        st.caption("Add to `.env`:\n`OPENAI_API_KEY=sk-...`")
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
    H               = st.number_input("Room Height (m)",   value=3.0,  step=0.1)
    U_wall          = st.number_input("Wall U-Value",      value=1.8,  step=0.1)
    U_glass         = st.number_input("Glass U-Value",     value=5.8,  step=0.1)
    DT              = st.number_input("DT (deg C)",        value=10,   step=1)
    people_per_room = st.number_input("People / Room",     value=2,    step=1)
    Q_person        = st.number_input("Heat / Person (W)", value=75,   step=5)
    st.divider()
    st.markdown("**Room Filtering**")
    min_area_m2 = st.number_input("Min Room Area (m2)", value=2.0,   step=0.5)
    max_area_m2 = st.number_input("Max Room Area (m2)", value=300.0, step=10.0)
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
    stair_parallel_min = st.number_input("Min parallel lines → stair",  value=4,    step=1)
    stair_angle_tol    = st.number_input("Stair angle tolerance (deg)", value=8.0,  step=1.0)
    max_stair_area_m2  = st.number_input("Max staircase area (m2)",     value=20.0, step=1.0)
    st.divider()
    st.markdown("**Bridging — Mode B**")
    gap_close_tol  = st.number_input("Snap / merge tolerance (mm)",   value=15.0,   step=5.0)
    max_door_width = st.number_input("Max door / archway width (mm)", value=1500.0, step=100.0)
    min_wall_len   = st.number_input("Min wall segment (mm)",         value=200.0,  step=50.0)
    st.divider()
    show_raw = st.checkbox("Show raw geometry overlay", value=False)

# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
ss = st.session_state
for key, default in [
    ("show_floorplan", False),
    ("show_results",   False),
    ("conf_rooms",     []),
    ("form_sel",       {}),
    ("duct_result",    None),   # ← NEW: stores duct pipeline output
]:
    if key not in ss:
        ss[key] = default

# ═════════════════════════════════════════════════════════════════════════════
#  ① FILE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
if not uploaded_file:
    st.info("👆 Upload a DWG file to begin.")
    st.stop()

dwg_bytes = uploaded_file.getvalue()
dwg_name  = uploaded_file.name

with st.spinner("🔄 Converting DWG → DXF…"):
    try:
        dxf_path = run_oda_conversion(dwg_bytes, dwg_name, oda_path)
    except RuntimeError as e:
        st.error(str(e)); st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  ② LAYER SCAN + LAYER SELECTION
# ═════════════════════════════════════════════════════════════════════════════
sorted_layers, layer_entity_count, all_layers_in_file = scan_dxf_layers(dxf_path)
active_layers = [l for l in sorted_layers if layer_entity_count.get(l, 0) > 0]

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

with st.expander("🔍 Layer Selection", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Layers in DXF :**")
        for lyr in sorted_layers:
            cnt = layer_entity_count.get(lyr, 0)
            tag = ("🟩 WALL"  if lyr.upper() in WALL_LAYERS else
                   "🟦 GLASS" if lyr.upper() in GLASS_LAYERS else
                   "🪑 FURN"  if lyr.upper() in FURN_LAYERS else "⚪")
            st.markdown(f"{tag} `{lyr}` — {cnt} entities")
    with col2:
        smart_def     = [l for l in active_layers
                         if not any(kw in l.lower() for kw in NON_WALL_KW)]
        wall_default  = ([l for l in active_layers if l.upper() in matched_wall]
                         or (smart_def if poor_match else []))
        glass_default = [l for l in active_layers if l.upper() in matched_glass]
        furn_default  = [l for l in active_layers if l.upper() in matched_furn]

        sel_wall  = st.multiselect("Wall / Base layers",  active_layers, wall_default,  key="sel_wall")
        sel_glass = st.multiselect("Glass layers",        active_layers, glass_default, key="sel_glass")
        sel_furn  = st.multiselect("Furniture / Doors",   active_layers, furn_default,  key="sel_furn")

        if sel_wall:  WALL_LAYERS  = {l.upper() for l in sel_wall}
        if sel_glass: GLASS_LAYERS = {l.upper() for l in sel_glass}
        if sel_furn:  FURN_LAYERS  = {l.upper() for l in sel_furn}

# ═════════════════════════════════════════════════════════════════════════════
#  ③ SHOW AI FLOOR PLAN BUTTON
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
fp_c1, fp_c2, fp_c3 = st.columns([2, 3, 2])
with fp_c2:
    if st.button("🗺️ Show AI Floor Plan", key="show_fp_btn",
                 type="primary", use_container_width=True):
        ss["show_floorplan"] = True
        ss["show_results"] = False
        ss["conf_rooms"]   = []
        ss["form_sel"]     = {}
        ss["duct_result"]  = None   # reset duct on new plan

if not ss["show_floorplan"]:
    st.markdown(
        "<p style='text-align:center;color:#556677;font-size:13px;margin-top:6px'>"
        "☝️ Click the button above to run AI room detection and show the floor plan.</p>",
        unsafe_allow_html=True)
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  ④ ROOM DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("🔲 Extracting geometry and detecting rooms…"):
    try:
        result = detect_rooms_cached(
            dxf_path=dxf_path,
            wall_layers=frozenset(WALL_LAYERS), glass_layers=frozenset(GLASS_LAYERS),
            furn_layers=frozenset(FURN_LAYERS),  mode_override=mode_override,
            snap_tol=float(snap_tol), bridge_tol=float(bridge_tol),
            glass_edge_thresh=float(glass_edge_thresh),
            glass_proximity_mult=float(glass_proximity_mult),
            min_area_m2=float(min_area_m2), max_area_m2=float(max_area_m2),
            min_compact=float(min_compact),  max_aspect_a=float(max_aspect_a),
            outer_area_pct_b=float(outer_area_pct_b), min_solidity=float(min_solidity),
            max_aspect_b=float(max_aspect_b), max_interior_walls=int(max_interior_walls),
            exclude_stairs=bool(exclude_stairs), stair_parallel_min=int(stair_parallel_min),
            stair_angle_tol=float(stair_angle_tol), max_stair_area_m2=float(max_stair_area_m2),
            gap_close_tol=float(gap_close_tol), max_door_width=float(max_door_width),
            min_wall_len=float(min_wall_len),
        )
    except RuntimeError as e:
        st.error(str(e)); st.stop()

USE_GLASS_MODE = result["use_glass"]
mode_label     = result["mode_label"]
unit_factor    = result["unit_factor"]
unit_div       = result["unit_div"]

wall_sn        = [shapely_wkb.loads(b) for b in result["wall_sn_wkb"]]
glass_sn       = [shapely_wkb.loads(b) for b in result["glass_sn_wkb"]]
bridges_used   = [shapely_wkb.loads(b) for b in result["bridges_wkb"]]
wall_cavities  = [shapely_wkb.loads(b) for b in result["wall_cavities_wkb"]]
all_segs_d     = [shapely_wkb.loads(b) for b in result["all_segs_wkb"]]
raw_wall_lines = result["raw_wall_lines"]

rooms_unified = [{**r, "polygon": shapely_wkb.loads(r["poly_wkb"])}
                 for r in result["rooms_serial"]]
extracted_objects = [{**o, "point": Point(o["pt_xy"])}
                     for o in result["extracted_objects"]]

n_rooms = len(rooms_unified)
n_glass = sum(1 for r in rooms_unified if r["is_glass"])
n_wall  = n_rooms - n_glass

if n_rooms == 0:
    st.error("No rooms detected. Try increasing Bridge tolerance or decreasing Min Area.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  ⑤ AI CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
ai_classifications = {}
if enable_ai:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        st.warning("⚠️ OPENAI_API_KEY not found — add it to your `.env` file.")
    else:
        rooms_for_ai = []
        for r in rooms_unified:
            poly    = r["polygon"]; area_m2 = poly.area / unit_factor
            perim_m = poly.length / unit_div
            minx, miny, maxx, maxy = poly.bounds
            w = (maxx - minx) / unit_div; h = (maxy - miny) / unit_div
            aspect = round(max(w, h) / max(min(w, h), 0.01), 2)
            rooms_for_ai.append({
                "room_id": r["room_id"], "area_m2": round(area_m2, 2),
                "perimeter_m": round(perim_m, 2), "width_m": round(w, 2),
                "height_m": round(h, 2), "aspect_ratio": aspect,
                "glass_fraction": round(r["gf"], 3), "is_glass_room": r["is_glass"],
                "object_count": len(r["objects_inside"]),
            })
        rooms_json_str = json.dumps(rooms_for_ai)

        with st.spinner("🖼️ Rendering floor plan for AI vision…"):
            try:
                floorplan_b64 = render_floorplan_for_vision(
                    rooms_unified, raw_wall_lines,
                    USE_GLASS_MODE, wall_sn, glass_sn,
                    extracted_objects=extracted_objects)
            except Exception as e:
                st.warning(f"⚠️ Floor plan render failed ({e}) — text-only fallback.")
                floorplan_b64 = None

        with st.spinner("🤖 GPT-4o is analysing your floor plan…"):
            try:
                ai_classifications = classify_rooms_ai_vision(
                    floorplan_b64 or "", rooms_json_str, openai_key)
                st.success(f"✅ AI classified {len(ai_classifications)} rooms")
            except Exception as e:
                st.warning(f"⚠️ AI classification failed: {e}")
                ai_classifications = {}

def get_ai_label(room):
    rid = room["room_id"]; info = ai_classifications.get(rid, {})
    rtype = info.get("type", "")
    if rtype and rtype != "Other":
        return f"{ROOM_TYPE_EMOJI.get(rtype,'')} {rtype}", rtype
    return room["name"], "Other"

for room in rooms_unified:
    ai_name, ai_type = get_ai_label(room)
    room["ai_name"]       = ai_name
    room["ai_type"]       = ai_type
    room["ai_reason"]     = ai_classifications.get(room["room_id"], {}).get("reason", "")
    room["ai_confidence"] = ai_classifications.get(room["room_id"], {}).get("confidence", "")

# ═════════════════════════════════════════════════════════════════════════════
#  ⑥ HEAT LOAD FOR ALL ROOMS
# ═════════════════════════════════════════════════════════════════════════════
room_data = compute_room_heat_loads(
    rooms_unified, unit_factor, unit_div,
    H=float(H), U_wall=float(U_wall), U_glass=float(U_glass),
    DT=float(DT), people_per_room=int(people_per_room), Q_person=float(Q_person))

for i, row in enumerate(room_data):
    ru = rooms_unified[i]
    row["AI Type"]  = ru["ai_type"]
    row["AI Name"]  = ru["ai_name"]
    row["_room_id"] = ru["room_id"]
    row["Room"]     = ru["ai_name"] if ru["ai_type"] != "Other" else ru["name"]
    row["CFM"]      = round(watts_to_cfm(row["Q_total (W)"], float(DT)), 1)

summary_all = summarise_heat_loads(room_data)
summary_all["total_cfm"] = round(sum(r["CFM"] for r in room_data), 1)
disp_cols = display_columns()
df = pd.DataFrame(room_data)
n_ai_classified = sum(1 for r in rooms_unified if r["ai_type"] != "Other")

st.success(f"✅ **{n_rooms} rooms detected**  |  🤖 {n_ai_classified} AI-classified")

# ═════════════════════════════════════════════════════════════════════════════
#  ⑦ FLOOR PLAN IMAGE + DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("AI-Recognised Rooms : Floor Plan")

with st.spinner("🎨 Rendering floor plan…"):
    fp_fig, fp_png = render_floorplan_figure(
        room_data, raw_wall_lines, USE_GLASS_MODE,
        wall_sn, glass_sn, bridges_used, wall_cavities,
        all_segs_d, extracted_objects, show_raw=show_raw)

st.pyplot(fp_fig)
plt.close(fp_fig)

dl_c1, dl_c2, dl_c3 = st.columns([3, 2, 3])
with dl_c2:
    st.download_button(
        label="⬇️ Download Floor Plan (PNG)",
        data=fp_png,
        file_name="floorplan_ai.png",
        mime="image/png",
        use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
#  ⑧ ROOM MEASUREMENTS TABLE
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Room Measurements")
st.caption("Area and Perimeter are from actual polygon geometry.")
meas_cols = ["Room", "Area (m2)", "Perimeter (m)", "Length ref (m)", "Breadth ref (m)"]
meas_cols = [c for c in meas_cols if c in df.columns]
st.dataframe(
    df[meas_cols].style.format({
        "Area (m2)": "{:.3f}",
        "Perimeter (m)": "{:.3f}",
        "Length ref (m)": "{:.2f}",
        "Breadth ref (m)": "{:.2f}"}),
    use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
#  ⑨ ROOM SELECTION & TR / CFM CALCULATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Room Selection & TR / CFM Calculation")

all_room_ids = [r.get("_room_id", f"R{i}") for i, r in enumerate(room_data)]
n_total      = len(room_data)

for rid in all_room_ids:
    if rid not in ss["form_sel"]:
        ss["form_sel"][rid] = False

# ══════════════════════════════════════════════════════════════════════════════
#  STATE A — SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if not ss["show_results"]:

    st.markdown(
        "<p style='color:#8899aa;font-size:13px;margin-bottom:16px'>"
        "Select rooms to include and click <b style='color:#00d4ff'>Calculate TR &amp; CFM</b>, "
        "or use the quick button to calculate all rooms at once.</p>",
        unsafe_allow_html=True)

    all_c1, all_c2, all_c3 = st.columns([2, 3, 2])
    with all_c2:
        if st.button(
            f"⚡ Calculate TR & CFM for ALL {n_total} Rooms",
            key="calc_all_btn",
            type="primary",
            use_container_width=True,
        ):
            ss["conf_rooms"]   = list(range(n_total))
            ss["show_results"] = True
            ss["duct_result"]  = None
            st.rerun()

    st.markdown(
        "<div style='text-align:center;color:#334455;font-size:12px;"
        "margin:6px 0 16px 0'>— or select individual rooms below —</div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#0d1117;border:1px solid #1a2535;"
        "border-radius:10px;padding:16px 18px;margin-bottom:16px'>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#556677;font-size:11px;font-family:monospace;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px'>"
        "Individual Room Selection</p>",
        unsafe_allow_html=True)

    with st.form(key="room_selection_form", border=False):
        cols = st.columns(3)
        checkbox_vals = {}
        for i, (rid, row) in enumerate(zip(all_room_ids, room_data)):
            ai_type  = row.get("AI Type", "Other")
            ai_emoji = ROOM_TYPE_EMOJI.get(ai_type, "")
            tc       = ROOM_TYPE_COLOR.get(ai_type, "#556677")
            area     = row["Area (m2)"]
            with cols[i % 3]:
                val = st.checkbox(
                    label=f"{ai_emoji} {row['Room']}",
                    value=ss["form_sel"].get(rid, False),
                    key=f"form_cb_{rid}")
                checkbox_vals[rid] = val
                st.markdown(
                    f"<div style='font-size:10px;color:{tc};font-family:monospace;"
                    f"margin-top:-10px;margin-bottom:10px;padding-left:24px'>"
                    f"{ai_type} &nbsp;·&nbsp; {area} m²"
                    f"</div>",
                    unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        bc1, bc2, bc3 = st.columns([3, 2, 3])
        with bc2:
            submitted = st.form_submit_button(
                "✅ Calculate TR & CFM",
                type="primary",
                use_container_width=True)

    if submitted:
        for rid, val in checkbox_vals.items():
            ss["form_sel"][rid] = val
        selected = [i for i, rid in enumerate(all_room_ids)
                    if checkbox_vals.get(rid, False)]
        if selected:
            ss["conf_rooms"]   = selected
            ss["show_results"] = True
            ss["duct_result"]  = None
            st.rerun()
        else:
            st.warning("☝️ Tick at least one room above to calculate.")

# ══════════════════════════════════════════════════════════════════════════════
#  STATE B — RESULTS + DUCT LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
else:
    confirmed_idx = ss["conf_rooms"]
    filtered_rows = [room_data[i] for i in confirmed_idx if i < len(room_data)]
    n_conf        = len(filtered_rows)

    if n_conf == 0:
        st.warning("No rooms were selected. Click Change Selection.")
    else:
        TR_PER_WATT  = 1.0 / 3517.0
        total_q      = sum(r["Q_total (W)"] for r in filtered_rows)
        filt_summary = {
            "n_rooms":       n_conf,
            "total_area_m2": round(sum(r["Area (m2)"]  for r in filtered_rows), 2),
            "total_kw":      round(total_q / 1000,      3),
            "total_tr":      round(total_q * TR_PER_WATT, 3),
            "total_cfm":     round(sum(r["CFM"]         for r in filtered_rows), 1),
        }

        m1, m2, m3, m4, m5 = st.columns(5)
        _metric_card(m1, "🏠 Rooms",      str(filt_summary["n_rooms"]),           "")
        _metric_card(m2, "📐 Total Area", f"{filt_summary['total_area_m2']:.1f}", "m²")
        _metric_card(m3, "⚡ Heat Load",  f"{filt_summary['total_kw']:.2f}",      "kW")
        _metric_card(m4, "❄️ Cooling",   f"{filt_summary['total_tr']:.2f}",      "TR")
        _metric_card(m5, "💨 Airflow",   f"{filt_summary['total_cfm']:.0f}",     "CFM")

        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)

        st.markdown("#### 📋 Room-wise Breakdown")
        filt_df  = pd.DataFrame(filtered_rows)
        tbl_cols = ["Room", "AI Type", "Area (m2)", "TR", "CFM"]
        tbl_cols = [c for c in tbl_cols if c in filt_df.columns]
        st.dataframe(
            filt_df[tbl_cols].style.format({
                "TR": "{:.3f}", "CFM": "{:.1f}", "Area (m2)": "{:.2f}"}),
            use_container_width=True,
            height=min(460, 65 + 35 * n_conf))

        st.markdown(
            f"<div class='total-bar'>"
            f"<b style='color:#00ff99'>GRAND TOTAL</b>"
            f" &nbsp;—&nbsp; {filt_summary['n_rooms']} rooms"
            f" &nbsp;|&nbsp; {filt_summary['total_area_m2']:.1f} m²"
            f" &nbsp;|&nbsp; <b style='color:#00d4ff'>{filt_summary['total_tr']:.3f} TR</b>"
            f" &nbsp;|&nbsp; <b style='color:#76d7c4'>{filt_summary['total_cfm']:.0f} CFM</b>"
            f" &nbsp;|&nbsp; {filt_summary['total_kw']:.2f} kW"
            f"</div>",
            unsafe_allow_html=True)

        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        exp_c = ["Room", "AI Type", "Area (m2)", "TR", "CFM",
                 "Q_wall (W)", "Q_glass (W)", "Q_people (W)", "Q_total (W)",
                 "Glass % edge"]
        with dl1:
            ec = [c for c in exp_c if c in filt_df.columns]
            st.download_button("⬇️ Download Selected Rooms (CSV)",
                filt_df[ec].to_csv(index=False),
                "heat_cfm_selected.csv", "text/csv", use_container_width=True)
        with dl2:
            ec_all = [c for c in exp_c if c in df.columns]
            st.download_button("⬇️ Download All Rooms (CSV)",
                df[ec_all].to_csv(index=False),
                "rooms_all.csv", "text/csv", use_container_width=True)

        # ═════════════════════════════════════════════════════════════════════
        #  ⑩  DUCT LAYOUT SECTION  ← NEW in v20
        # ═════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🌬️ Duct Layout")
        st.markdown(
            "<p style='color:#8899aa;font-size:13px'>"
            "AHU is auto-placed at the centroid of all selected rooms. "
            "Branch ducts are routed (L-shaped) from AHU to each room centroid. "
            "Sizing follows SMACNA Equal Friction method.</p>",
            unsafe_allow_html=True)

        dc1, dc2, dc3 = st.columns([2, 3, 2])
        with dc2:
            gen_duct = st.button(
                "⚡ Generate Duct Layout",
                key="gen_duct_btn",
                type="primary",
                use_container_width=True,
            )

        if gen_duct:
            with st.spinner("🔧 Routing ducts and sizing… (SMACNA Equal Friction)"):
                from duct_engine import run_duct_pipeline
                ss["duct_result"] = run_duct_pipeline(
                    room_data=filtered_rows,
                    unit_div=unit_div,
                    raw_wall_lines=raw_wall_lines,
                    use_glass_mode=USE_GLASS_MODE,
                    wall_sn=wall_sn,
                    glass_sn=glass_sn,
                )

        if ss["duct_result"]:
            dr = ss["duct_result"]

            # AHU placement info
            ax_x, ax_y = dr["ahu_xy"]
            ax_m = round(ax_x / unit_div, 1)
            ay_m = round(ax_y / unit_div, 1)
            st.info(
                f"🏭 **AHU placed** at drawing coords ({ax_x:.0f}, {ax_y:.0f}) "
                f"≈ **({ax_m} m, {ay_m} m)** — centroid of all selected rooms"
            )

            # Floor plan with duct overlay
            st.subheader("Duct Layout — Floor Plan")
            st.pyplot(dr["fig"])
            plt.close(dr["fig"])

            ddl1, ddl2, ddl3 = st.columns(3)
            with ddl1:
                st.download_button(
                    "⬇️ Download Duct Floor Plan (PNG)",
                    data=dr["png_bytes"],
                    file_name="duct_layout.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with ddl2:
                st.download_button(
                    "⬇️ Download DXF (CAD-ready)",
                    data=dr["dxf_bytes"],
                    file_name="duct_layout.dxf",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

            # Duct sizing table
            st.markdown("---")
            st.subheader("📐 Duct Sizing Table")
            segs_df = pd.DataFrame([
                {
                    "Segment":       s["segment_id"],
                    "Room":          s["room_name"],
                    "CFM":           round(s.get("cfm", 0), 1),
                    "Size W×H (mm)": s.get("label", ""),
                    "Width (mm)":    s.get("width_mm", 0),
                    "Height (mm)":   s.get("height_mm", 0),
                    "Velocity (m/s)":s.get("velocity_ms", 0),
                    "Equiv ⌀ (mm)":  s.get("equiv_diam_mm", 0),
                    "Length (m)":    s.get("length_m", 0),
                    "Type":          "Main Trunk" if s.get("is_main") else "Branch",
                }
                for s in dr["segments"]
            ])
            st.dataframe(segs_df, use_container_width=True,
                         height=min(500, 65 + 35 * len(segs_df)))

            with ddl3:
                st.download_button(
                    "⬇️ Download Duct Sizes (CSV)",
                    data=dr["duct_csv"],
                    file_name="duct_sizing.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # BOQ
            st.markdown("---")
            st.subheader("📦 Bill of Quantities (BOQ)")
            st.caption(
                "Sheet metal area = 2×(W+H)×Length. "
                "+25% allowance for fittings (elbows, tees, reducers). "
                "Material: GI Sheet 0.63 mm gauge — SMACNA standard."
            )

            boq_lines = dr["boq_csv"].splitlines()
            if len(boq_lines) > 1:
                boq_cols = [
                    "Duct Size", "Width mm", "Height mm", "Equiv ⌀ mm",
                    "Runs", "Length m", "Perimeter m", "Sheet Metal m²",
                    "Fittings m²", "Total m²", "Unit", "Notes",
                ]
                import csv as _csv, io as _io
                boq_reader = list(_csv.reader(_io.StringIO(dr["boq_csv"])))
                boq_df = pd.DataFrame(boq_reader[1:], columns=boq_reader[0])
                st.dataframe(boq_df, use_container_width=True)

            boq_dl1, _, _ = st.columns(3)
            with boq_dl1:
                st.download_button(
                    "⬇️ Download BOQ (CSV)",
                    data=dr["boq_csv"],
                    file_name="duct_boq.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Engineering disclaimer
            st.markdown(
                "<div style='background:#1a1000;border:1px solid #664400;"
                "border-radius:8px;padding:12px 16px;margin-top:16px;"
                "font-size:12px;color:#aa8844;font-family:monospace'>"
                "⚠️ <b>Engineering Disclaimer</b>: This duct layout is AI-generated "
                "for preliminary estimation only. Final sizing, routing, pressure drop "
                "calculations, and construction drawings must be reviewed and signed off "
                "by a qualified HVAC / MEP engineer. "
                "Standards reference: ASHRAE Fundamentals, SMACNA HVAC Duct Construction, NBC."
                "</div>",
                unsafe_allow_html=True,
            )

    # Change selection button
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns([3, 2, 3])
    with ch2:
        if st.button("🔄 Change Selection", key="re_select", use_container_width=True):
            ss["show_results"] = False
            ss["duct_result"]  = None
            st.rerun()
