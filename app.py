"""
app.py — Streamlit UI  v15
===========================
CAD Room Extractor + Heat Load Calculator

Run with:
    streamlit run app.py

Requires (same directory):
    geometry_engine.py
    heat_load.py
"""

import subprocess
import os

import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString

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

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLE
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CAD Room Extractor", layout="wide")
st.markdown("""
<style>
  .main { background: #0f1117 }
  h1 { color: #00d4ff; font-family: 'Courier New', monospace }
  .block-container { padding-top: 2rem }
  div[data-testid="metric-container"] {
    background: #1a1f2e; border-radius: 10px;
    padding: 10px; border: 1px solid #00d4ff33 }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ CAD Room Extractor + Heat Load Calculator")

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    oda_path = st.text_input(
        "ODAFileConverter Path",
        r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")

    st.divider()
    st.markdown("**Layers**")
    wall_input  = st.text_area("Wall / Base layers",
        "MAIN\nMAIN-4\nCEN-1\nA-WALL\nWALLS\nWall\nWalls\nBase\n0")
    glass_input = st.text_area("Glass layers (leave blank = Mode B)", "GLASS")
    furn_input  = st.text_area("Furniture / Door layers (optional)",
        "Furniture\nDoors\nFURNITURE\nDOOR")

    st.divider()
    st.markdown("**Detection Mode**")
    mode_override = st.selectbox(
        "Mode",
        ["Auto-detect",
         "Force Glass-partition (Mode A)",
         "Force Base-layer only (Mode B)"],
        index=0)

    st.divider()
    st.markdown("**Heat Load Parameters**")
    H               = st.number_input("Room Height (m)",  value=3.0,  step=0.1)
    U_wall          = st.number_input("Wall U-Value",     value=1.8,  step=0.1)
    U_glass         = st.number_input("Glass U-Value",    value=5.8,  step=0.1)
    DT              = st.number_input("DT (deg C)",       value=10,   step=1)
    people_per_room = st.number_input("People / Room",    value=2,    step=1)
    Q_person        = st.number_input("Heat/Person (W)",  value=75,   step=5)

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
    glass_edge_thresh    = st.number_input(
        "Glass edge threshold (0-1)", value=0.15, step=0.05)
    glass_proximity_mult = st.number_input(
        "Glass proximity multiplier", value=3.0, step=0.5,
        min_value=1.0, max_value=10.0)

    st.divider()
    st.markdown("**Shape Validation — Mode B**")
    outer_area_pct_b   = st.number_input("Outer envelope threshold (%)", value=25.0, step=5.0)
    min_solidity       = st.number_input("Min solidity (0-1)", value=0.50, step=0.05,
                                          min_value=0.0, max_value=1.0)
    max_aspect_b       = st.number_input("Max aspect ratio (Mode B)", value=15.0, step=1.0)
    max_interior_walls = st.number_input("Max interior wall segments", value=8, step=1)
    exclude_stairs     = st.checkbox("Exclude staircase regions", value=True)
    stair_parallel_min = st.number_input("Min parallel lines for stair", value=4, step=1)
    stair_angle_tol    = st.number_input("Stair angle tolerance (deg)", value=8.0, step=1.0)
    max_stair_area_m2  = st.number_input("Max staircase area (m2)",     value=20.0, step=1.0)

    st.divider()
    st.markdown("**Bridging — Mode B**")
    gap_close_tol  = st.number_input("Snap/merge tolerance (mm)",   value=15.0,   step=5.0)
    max_door_width = st.number_input("Max door/archway width (mm)", value=1500.0, step=100.0)
    min_wall_len   = st.number_input("Min wall segment (mm)",       value=200.0,  step=50.0)

    st.divider()
    show_raw = st.checkbox("Show raw geometry overlay", value=False)

# ──────────────────────────────────────────────────────────────────────────────
#  FILE UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
if not uploaded_file:
    st.info("👆 Upload a DWG file to begin.")
    st.stop()

dwg_filename  = uploaded_file.name
dwg_path_file = os.path.join(os.getcwd(), dwg_filename)
with open(dwg_path_file, "wb") as f:
    f.write(uploaded_file.getbuffer())

# ──────────────────────────────────────────────────────────────────────────────
#  ODA CONVERSION
# ──────────────────────────────────────────────────────────────────────────────
dxf_folder = os.path.join(os.getcwd(), "converted_dxf")
os.makedirs(dxf_folder, exist_ok=True)

with st.spinner("🔄 Converting DWG to DXF..."):
    try:
        subprocess.run(
            [oda_path, os.getcwd(), dxf_folder, "ACAD2013", "DXF", "0", "1"],
            check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        st.error(f"ODA conversion failed: {e.stderr}"); st.stop()
    except FileNotFoundError:
        st.error("ODAFileConverter.exe not found. Check the path in sidebar."); st.stop()

dxf_name      = dwg_filename.rsplit(".", 1)[0] + ".dxf"
dxf_path_conv = os.path.join(dxf_folder, dxf_name)
if not os.path.exists(dxf_path_conv):
    hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
    if not hits:
        st.error("No DXF file found after conversion."); st.stop()
    dxf_path_conv = os.path.join(dxf_folder, hits[0])

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD DXF
# ──────────────────────────────────────────────────────────────────────────────
try:
    doc = ezdxf.readfile(dxf_path_conv)
except Exception as e:
    st.error(f"DXF load error: {e}"); st.stop()

msp                = doc.modelspace()
all_layers_in_file = {layer.dxf.name for layer in doc.layers}

layer_entity_count = {}
for ent in msp:
    lyr = ent.dxf.get("layer", "0")
    layer_entity_count[lyr] = layer_entity_count.get(lyr, 0) + 1

sorted_layers = sorted(all_layers_in_file,
    key=lambda l: layer_entity_count.get(l, 0), reverse=True)
active_layers = [l for l in sorted_layers if layer_entity_count.get(l, 0) > 0]

# ──────────────────────────────────────────────────────────────────────────────
#  LAYER RESOLUTION
# ──────────────────────────────────────────────────────────────────────────────
WALL_LAYERS_DEFAULT  = {l.strip().upper() for l in wall_input.strip().splitlines()  if l.strip()}
GLASS_LAYERS_DEFAULT = {l.strip().upper() for l in glass_input.strip().splitlines() if l.strip()}
FURN_LAYERS_DEFAULT  = {l.strip().upper() for l in furn_input.strip().splitlines()  if l.strip()}

WALL_LAYERS  = WALL_LAYERS_DEFAULT.copy()
GLASS_LAYERS = GLASS_LAYERS_DEFAULT.copy()
FURN_LAYERS  = FURN_LAYERS_DEFAULT.copy()

matched_wall  = WALL_LAYERS  & {x.upper() for x in all_layers_in_file}
matched_glass = GLASS_LAYERS & {x.upper() for x in all_layers_in_file}
matched_furn  = FURN_LAYERS  & {x.upper() for x in all_layers_in_file}

matched_wall_ents = sum(layer_entity_count.get(l, 0)
    for l in all_layers_in_file if l.upper() in matched_wall)
total_ents = sum(layer_entity_count.values()) or 1
poor_match = matched_wall_ents < total_ents * 0.20

NON_WALL_KW = {"furniture","plant","planter","text","vp","defpoint",
               "dimension","dim","hatch","annotation","title","border",
               "viewport","electrical","elec","plumbing","mech","door"}

# ── Layer Selection (unchanged) ───────────────────────────────────────────────
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
        sel_furn  = st.multiselect("Furniture/Door layers (optional)",
                                    options=active_layers, default=furn_default,  key="sel_furn")

        if sel_wall:  WALL_LAYERS  = {l.upper() for l in sel_wall}
        if sel_glass: GLASS_LAYERS = {l.upper() for l in sel_glass}
        if sel_furn:  FURN_LAYERS  = {l.upper() for l in sel_furn}

        matched_glass = GLASS_LAYERS & {x.upper() for x in all_layers_in_file}
        st.caption(f"Active: wall={sorted(WALL_LAYERS)} | glass={sorted(GLASS_LAYERS)}")

# ──────────────────────────────────────────────────────────────────────────────
#  MODE DETECTION  (silent)
# ──────────────────────────────────────────────────────────────────────────────
all_entity_layers_upper = {ent.dxf.get("layer","0").upper() for ent in msp}
glass_found = bool(GLASS_LAYERS & all_entity_layers_upper)
if not glass_found and GLASS_LAYERS:
    for ent in msp:
        if ent.dxftype() == "INSERT":
            try:
                bname = ent.dxf.name
                if bname in doc.blocks:
                    for sub in doc.blocks[bname]:
                        if sub.dxf.get("layer","0").upper() in GLASS_LAYERS:
                            glass_found = True
                            break
            except Exception:
                pass
        if glass_found:
            break

if   mode_override == "Force Glass-partition (Mode A)": USE_GLASS_MODE = True
elif mode_override == "Force Base-layer only (Mode B)": USE_GLASS_MODE = False
else:                                                    USE_GLASS_MODE = bool(GLASS_LAYERS) and glass_found

mode_label = ("🔵 Mode A: Glass-partition" if USE_GLASS_MODE
              else "🟩 Mode B: Base-layer")

ALLOWED_LAYERS_A = WALL_LAYERS | GLASS_LAYERS
ALLOWED_LAYERS_B = WALL_LAYERS | GLASS_LAYERS | FURN_LAYERS

# Scale (Mode B — silent)
insunits        = doc.header.get("$INSUNITS", 0)
scale_map       = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}
DRAWING_SCALE_B = scale_map.get(insunits, 1.0)

# ──────────────────────────────────────────────────────────────────────────────
#  GEOMETRY EXTRACTION  (silent)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🔍 Analysing floor plan..."):
    if USE_GLASS_MODE:
        raw_a = extract_all_v8(msp, doc, ALLOWED_LAYERS_A, GLASS_LAYERS)

        wall_segs  = [g for (g, ig, ia) in raw_a if not ig and not ia]
        glass_segs = [g for (g, ig, ia) in raw_a if ig  and not ia]
        arc_segs   = [g for (g, ig, ia) in raw_a if ia]
        all_segs_display = [g for (g, _, _) in raw_a]

        all_xs = [c[0] for ls in wall_segs + glass_segs for c in ls.coords]
        if not all_xs:
            st.error("No wall/glass geometry found. Check layer names in sidebar."); st.stop()

        span        = max(all_xs) - min(all_xs)
        unit_guess  = "mm" if span > 500 else "m"
        unit_factor = 1_000_000 if unit_guess == "mm" else 1
        unit_div    = 1000      if unit_guess == "mm" else 1

        raw_wall_lines = []
        raw_furn_lines = []

    else:
        wall_segs = glass_segs = arc_segs = []
        all_segs_display = []
        raw_wall_lines  = []
        raw_glass_lines = []
        raw_furn_lines  = []

        for ent in msp.query("LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE"):
            layer_up = ent.dxf.get("layer", "0").upper()
            if   layer_up in WALL_LAYERS:  process_entity_v11(ent, raw_wall_lines,  DRAWING_SCALE_B, exclude_arcs=True)
            elif layer_up in GLASS_LAYERS: process_entity_v11(ent, raw_glass_lines, DRAWING_SCALE_B, False)
            elif layer_up in FURN_LAYERS:  process_entity_v11(ent, raw_furn_lines,  DRAWING_SCALE_B, False)

        unit_factor      = 1_000_000.0
        unit_div         = 1_000.0
        all_segs_display = [LineString(l) for l in raw_wall_lines]

has_geometry = (USE_GLASS_MODE and len(wall_segs) > 0) or \
               (not USE_GLASS_MODE and len(raw_wall_lines) > 0)
if not has_geometry:
    st.error(f"No geometry found on wall layers: {sorted(WALL_LAYERS)}"); st.stop()

# Furniture
extracted_objects = process_furniture_to_objects(raw_furn_lines)

# ──────────────────────────────────────────────────────────────────────────────
#  ROOM DETECTION  (silent — log=None)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🔲 Detecting rooms..."):
    if USE_GLASS_MODE:
        accepted, bridges_used, wall_sn, glass_sn = detect_rooms_mode_a(
            wall_segs=wall_segs, glass_segs=glass_segs, arc_segs=arc_segs,
            snap_tol=snap_tol, bridge_tol=bridge_tol,
            glass_edge_thresh=glass_edge_thresh,
            glass_proximity_mult=glass_proximity_mult,
            min_area_m2=min_area_m2, max_area_m2=max_area_m2,
            min_compact=min_compact, max_aspect=max_aspect_a,
            unit_factor=unit_factor, log=None)
        wall_cavities = []

        rooms_unified = []
        for i, (poly, gf, is_glass) in enumerate(accepted):
            minx, miny, maxx, maxy = poly.bounds
            rooms_unified.append({
                "name": f"Room {i+1}", "room_id": f"R{i+1}",
                "polygon": poly, "gf": gf, "is_glass": is_glass,
                "width": maxx-minx, "height": maxy-miny,
                "area": poly.area, "objects_inside": [],
            })
    else:
        room_list, wall_cavities = detect_rooms_mode_b(
            raw_wall_lines=raw_wall_lines, extracted_objects=extracted_objects,
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

        rooms_unified = []
        for r in room_list:
            rooms_unified.append({
                "name": r["name"], "room_id": r["room_id"],
                "polygon": r["polygon"], "gf": 0.0, "is_glass": False,
                "width": r["width"], "height": r["height"],
                "area": r["area"], "objects_inside": r["objects_inside"],
            })

# Attach objects to rooms
for room in rooms_unified:
    poly_buf = room["polygon"].buffer(10)
    for obj in extracted_objects:
        if poly_buf.covers(obj["point"]):
            room["objects_inside"].append(obj["object_id"])

n_rooms = len(rooms_unified)
n_glass = sum(1 for r in rooms_unified if r.get("is_glass"))
n_wall  = n_rooms - n_glass

if not rooms_unified:
    st.error("No rooms detected. Try increasing Bridge tolerance or decreasing Min Area.")
    fig0, ax0 = plt.subplots(figsize=(14, 7))
    fig0.patch.set_facecolor("#0f1117"); ax0.set_facecolor("#0f1117")
    draw_segs = wall_sn if USE_GLASS_MODE else [LineString(l) for l in raw_wall_lines[:5000]]
    for ls in draw_segs[:5000]:
        xs, ys = ls.xy; ax0.plot(xs, ys, color="#00d4ff", lw=0.5, alpha=0.5)
    ax0.set_aspect("equal")
    ax0.set_title("Raw geometry — no rooms found", color="white")
    st.pyplot(fig0); plt.close(); st.stop()

# ──────────────────────────────────────────────────────────────────────────────
#  HEAT LOAD — compute for ALL rooms upfront
# ──────────────────────────────────────────────────────────────────────────────
room_data   = compute_room_heat_loads(
    rooms_unified, unit_factor, unit_div,
    H=H, U_wall=U_wall, U_glass=U_glass, DT=DT,
    people_per_room=int(people_per_room), Q_person=Q_person)

summary_all = summarise_heat_loads(room_data)
disp_cols   = display_columns()
df          = pd.DataFrame(room_data)

# ──────────────────────────────────────────────────────────────────────────────
#  SINGLE STATUS LINE
# ──────────────────────────────────────────────────────────────────────────────
st.success(
    f"✅  **{n_rooms} rooms detected** — "
    f"{n_glass} glass · {n_wall} wall  |  {mode_label}  |  "
    f"Total **{summary_all['total_tr']:.2f} TR**  /  **{summary_all['total_kw']:.2f} kW**")

# ──────────────────────────────────────────────────────────────────────────────
#  FLOOR PLAN
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Floor Plan")

fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333")

if show_raw:
    for ls in all_segs_display[:8000]:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#1e3a50", lw=0.25, alpha=0.35)

if USE_GLASS_MODE:
    for ls in wall_sn:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)
    for ls in glass_sn:
        xs, ys = ls.xy; ax.plot(xs, ys, color="#00d4ff", lw=1.4, alpha=0.75, zorder=2)
else:
    for l in raw_wall_lines[:8000]:
        ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]],
                color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

for br in bridges_used:
    xs, ys = br.xy; ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.6, zorder=2)
for cav in wall_cavities:
    if cav.geom_type == "Polygon":
        xs, ys = cav.exterior.xy; ax.fill(xs, ys, color="dimgray", alpha=0.8, zorder=2)

for row in room_data:
    poly  = row["_poly"]; color = row["_color"]
    ig    = row["_is_glass"]; gf = row["_gf"]
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
ax.set_title(
    f"{n_rooms} Rooms  ({n_wall} wall · {n_glass} glass)  |  {mode_label}",
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

# ──────────────────────────────────────────────────────────────────────────────
#  ROOM MEASUREMENTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Room Measurements & Heat Load")
st.caption("Area and Perimeter are from actual polygon geometry, not bounding box.")
st.dataframe(
    df[disp_cols].style
    .format({"TR": "{:.3f}", "Area (m2)": "{:.3f}", "Perimeter (m)": "{:.3f}",
             "Q_total (W)": "{:.1f}", "Glass % edge": "{:.1f}"})
    .background_gradient(subset=["TR"],           cmap="YlOrRd")
    .background_gradient(subset=["Glass % edge"], cmap="Blues")
    .background_gradient(subset=["Area (m2)"],    cmap="Greens"),
    use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
#  ROOM MULTI-SELECT
#  No selection = all rooms. Any combination of rooms = filtered view.
# ──────────────────────────────────────────────────────────────────────────────
all_room_names = [r["Room"] for r in room_data]

selected_rooms = st.multiselect(
    "🔽 Select Rooms for Heat Load Summary  (leave empty = All Rooms)",
    options=all_room_names,
    default=[],
    placeholder="Select one or more rooms…",
    key="room_multiselect")

# Resolve filtered dataset
if not selected_rooms:
    # Nothing selected → show all rooms
    filtered_data  = room_data
    summary        = summary_all
    selection_label = "All Rooms"
else:
    selected_set   = set(selected_rooms)
    filtered_data  = [r for r in room_data if r["Room"] in selected_set]
    summary        = summarise_heat_loads(filtered_data)
    n_sel          = len(selected_rooms)
    selection_label = (f"{selected_rooms[0]}"
                       if n_sel == 1
                       else f"{n_sel} rooms selected")

chart_title = (
    f"TR per Room  |  Total: {summary['total_tr']:.2f} TR"
    if not selected_rooms
    else f"TR — {selection_label}  |  Total: {summary['total_tr']:.2f} TR")

# ──────────────────────────────────────────────────────────────────────────────
#  HEAT LOAD SUMMARY METRICS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"🌡️ Heat Load Summary — {selection_label}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏠 Rooms",       summary["n_rooms"])
c2.metric("🔵 Glass Rooms", summary["n_glass"])
c3.metric("📐 Total Area",  f"{summary['total_area_m2']:.2f} m2")
c4.metric("⚡ Total Load",  f"{summary['total_kw']:.2f} kW")
c5.metric("❄️ Total TR",    f"{summary['total_tr']:.2f} TR")

# Component breakdown table — shown when 1 or more rooms are selected
if selected_rooms and filtered_data:
    st.markdown("**Component Breakdown**")
    breakdown_df = pd.DataFrame([{
        "Room":            r["Room"],
        "Type":            r["Type"],
        "Area (m2)":       r["Area (m2)"],
        "Glass % edge":    r["Glass % edge"],
        "Wall Heat (W)":   r["Q_wall (W)"],
        "Glass Heat (W)":  r["Q_glass (W)"],
        "People Heat (W)": r["Q_people (W)"],
        "Total (W)":       r["Q_total (W)"],
        "TR":              r["TR"],
    } for r in filtered_data])
    st.dataframe(
        breakdown_df.style
        .format({"Area (m2)": "{:.3f}", "Glass % edge": "{:.1f}",
                 "Wall Heat (W)": "{:.1f}", "Glass Heat (W)": "{:.1f}",
                 "People Heat (W)": "{:.1f}", "Total (W)": "{:.1f}",
                 "TR": "{:.3f}"})
        .background_gradient(subset=["TR"],        cmap="YlOrRd")
        .background_gradient(subset=["Total (W)"], cmap="Oranges"),
        use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
#  TR BAR CHART  (filtered)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 TR Chart")

n_bars     = len(filtered_data)
bar_labels = [r["Room"]   for r in filtered_data]
bar_values = [r["TR"]     for r in filtered_data]
bar_colors = [r["_color"] for r in filtered_data]

# Highlight selected bars when a subset is chosen
if selected_rooms:
    bar_alphas = [0.95 if r["Room"] in set(selected_rooms) else 0.25
                  for r in filtered_data]
else:
    bar_alphas = [0.85] * n_bars

fig2, ax2 = plt.subplots(figsize=(max(6, n_bars * 1.2), 4))
fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
ax2.tick_params(colors="#888")
for sp in ax2.spines.values(): sp.set_edgecolor("#333")

for i, (lbl, val, col, alpha) in enumerate(zip(bar_labels, bar_values, bar_colors, bar_alphas)):
    bar = ax2.bar(lbl, val, color=col, alpha=alpha, edgecolor="#ffffff22")
    ax2.text(i, val + max(bar_values, default=1) * 0.02,
             f"{val:.3f}",
             ha="center", va="bottom", color="white", fontsize=9,
             fontfamily="monospace")

ax2.set_xlabel("Room", color="#aaa")
ax2.set_ylabel("TR",   color="#aaa")
ax2.set_title(chart_title, color="#00d4ff", fontfamily="monospace")
ax2.grid(axis="y", color="#1a2a3a", lw=0.4)
plt.xticks(rotation=30 if n_bars > 6 else 0, ha="right", color="#aaa")
st.pyplot(fig2); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD BUTTONS
# ──────────────────────────────────────────────────────────────────────────────
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Download All Rooms (CSV)",
        df[disp_cols].to_csv(index=False),
        "rooms_heat_load_all.csv",
        "text/csv")
with col_dl2:
    if selected_rooms and filtered_data:
        df_sel      = pd.DataFrame(filtered_data)[disp_cols]
        safe_label  = selection_label.replace(" ", "_").replace("/", "-")
        st.download_button(
            f"⬇️ Download Selected Rooms (CSV)",
            df_sel.to_csv(index=False),
            f"heat_load_{safe_label}.csv",
            "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"{mode_label}  |  "
    f"Wall: {', '.join(sorted(WALL_LAYERS))}  |  "
    f"Glass: {', '.join(sorted(GLASS_LAYERS)) or 'none'}  |  "
    f"{n_rooms} rooms ({n_glass} glass)  |  "
    f"Total: {summary_all['total_tr']:.2f} TR")