# import streamlit as st
# import subprocess
# import os
# import ezdxf
# import numpy as np
# from shapely.geometry import LineString
# from shapely.ops import polygonize, unary_union
# import pandas as pd
# import math

# # ---------------------- Streamlit Config ----------------------

# st.set_page_config(page_title="DWG Room Extractor + Heat Load", layout="wide")
# st.title("🏗️ CAD Room Extractor (Block Explode Mode Enabled)")

# # ---------------------- Upload DWG + ODA ----------------------

# uploaded_file = st.file_uploader("Upload DWG File", type=["dwg"])

# oda_path = st.text_input(
#     "Path to ODAFileConverter.exe",
#     r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe"
# )

# if not uploaded_file:
#     st.stop()

# dwg_path = uploaded_file.name
# with open(dwg_path, "wb") as f:
#     f.write(uploaded_file.getbuffer())

# dxf_folder = "converted_dxf"
# os.makedirs(dxf_folder, exist_ok=True)

# try:
#     subprocess.run([
#         oda_path,
#         os.getcwd(),
#         dxf_folder,
#         "ACAD2013",
#         "DXF",
#         "0",
#         "1"
#     ], check=True)
# except Exception as e:
#     st.error(f"ODA conversion failed: {e}")
#     st.stop()

# dxf_path = os.path.join(dxf_folder, dwg_path.replace(".dwg", ".dxf"))

# # ---------------------- Load DXF ----------------------

# try:
#     doc = ezdxf.readfile(dxf_path)
# except:
#     st.error("DXF load failed.")
#     st.stop()

# msp = doc.modelspace()

# # ---------------------- Recursive Block Explode ----------------------

# def extract_entities_recursive(layout, doc):
#     entities = []
#     for e in layout:
#         t = e.dxftype()
#         if t in ["LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE"]:
#             entities.append(e)
#         elif t == "INSERT":
#             block_name = e.dxf.name
#             if block_name not in doc.blocks:
#                 continue
#             block = doc.blocks[block_name]
#             block_ents = extract_entities_recursive(block, doc)
#             x, y = e.dxf.insert.x, e.dxf.insert.y
#             for be in block_ents:
#                 be_copy = be.copy()
#                 if hasattr(be_copy.dxf, "start"):
#                     be_copy.dxf.start = (be_copy.dxf.start.x + x, be_copy.dxf.start.y + y)
#                 if hasattr(be_copy.dxf, "end"):
#                     be_copy.dxf.end = (be_copy.dxf.end.x + x, be_copy.dxf.end.y + y)
#                 entities.append(be_copy)
#     return entities

# st.info("Extracting ALL geometry (modelspace + blocks)…")
# all_entities = extract_entities_recursive(msp, doc)

# if len(all_entities) == 0:
#     st.error("Still no geometry found. The drawing may contain 3D surfaces only.")
#     st.stop()

# st.success(f"Found {len(all_entities)} total geometry entities!")

# # ---------------------- Convert Entities to Lines ----------------------

# lines = []

# def arc_to_lines(arc, segments=20):
#     start = math.radians(arc.dxf.start_angle)
#     end = math.radians(arc.dxf.end_angle)
#     if end < start:
#         end += 2 * math.pi
#     ang = np.linspace(start, end, segments)
#     pts = [
#         (
#             arc.dxf.center.x + arc.dxf.radius * math.cos(a),
#             arc.dxf.center.y + arc.dxf.radius * math.sin(a)
#         ) for a in ang
#     ]
#     return LineString(pts)

# for e in all_entities:
#     t = e.dxftype()
#     if t == "LINE":
#         lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
#     elif t == "LWPOLYLINE":
#         pts = [(p[0], p[1]) for p in e]
#         lines.append(LineString(pts))
#     elif t == "POLYLINE":
#         pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
#         lines.append(LineString(pts))
#     elif t == "ARC":
#         lines.append(arc_to_lines(e))
#     elif t == "CIRCLE":
#         ang = np.linspace(0, 2 * math.pi, 40)
#         pts = [
#             (
#                 e.dxf.center.x + e.dxf.radius * math.cos(a),
#                 e.dxf.center.y + e.dxf.radius * math.sin(a)
#             )
#             for a in ang
#         ]
#         lines.append(LineString(pts))

# if len(lines) == 0:
#     st.error("Extracted geometry does not contain any linework.")
#     st.stop()

# st.success(f"Converted into {len(lines)} line segments")

# # ---------------------- Polygonize Rooms ----------------------

# merged = unary_union(lines)
# polys = list(polygonize(merged))

# # Filter small polygons
# rooms = [p for p in polys if p.area > 2000]  # assuming DXF in mm²

# if not rooms:
#     st.error("No closed rooms detected even after exploding blocks.")
#     st.info("This means walls do not form closed loops.")
#     st.stop()

# st.success(f"Detected {len(rooms)} rooms!")

# # ---------------------- Room Dimensions (convert mm → m) ----------------------

# data = []
# for i, p in enumerate(rooms):
#     minx, miny, maxx, maxy = p.bounds
#     length_m = (maxx - minx)/1000
#     breadth_m = (maxy - miny)/1000
#     area_m2 = p.area / 1e6
#     perimeter_m = p.length / 1000
#     data.append({
#         "Room": f"Room {i+1}",
#         "Length (m)": round(length_m, 2),
#         "Breadth (m)": round(breadth_m, 2),
#         "Area (m²)": round(area_m2, 2),
#         "Perimeter (m)": round(perimeter_m, 2)
#     })

# df = pd.DataFrame(data)
# st.dataframe(df, use_container_width=True)

# # ---------------------- Heat Load Inputs ----------------------

# st.subheader("Heat Load Inputs")

# H = st.number_input("Room Height (m)", value=3.0)
# U = st.number_input("Wall U-Value", value=1.8)
# DT = st.number_input("ΔT (°C)", value=10)

# df["Wall Area"] = df["Perimeter (m)"] * H
# df["Q_wall (W)"] = df["Wall Area"] * U * DT

# st.subheader("Heat Load Output")
# st.dataframe(df)

# # Total
# total_w = df["Q_wall (W)"].sum()
# total_kw = total_w / 1000
# total_tr = total_kw / 3.517

# st.metric("Total TR", f"{total_tr:.2f}")

# st.download_button("Download CSV", df.to_csv(index=False), "results.csv")






























































# """
# CAD Room Extractor + Heat Load  v8
# ====================================
# KEY FIX — Glass rooms in bottom-right corner not detected as separate rooms:

# ROOT CAUSE of v7 failure:
#   In Pass 2 (containment check), sub-polygons inside a parent room were only kept
#   if glass_fraction >= glass_edge_thresh (default 0.15).
#   BUT the bottom-right glass window room:
#     - Has glass on only ONE side (the partition wall between it and the main room)
#     - Other 3 sides are outer walls → glass_fraction was ~0.25 or less
#     - If threshold was too high, it got dropped
  
#   ADDITIONALLY: The edge_glass_fraction() used snap_tol*1.5 as proximity check,
#   which sometimes missed glass lines that were slightly offset from polygon edges.

# NEW FIXES in v8:
#   1. edge_glass_fraction() now uses a smarter adaptive buffer:
#        tol * 3.0 instead of tol * 1.5, catching slightly offset glass walls
#   2. Sub-room inside parent: if ANY glass detected (gf > 0.0), keep it.
#        The old rule (gf >= glass_edge_thresh) was too strict for corner rooms.
#   3. New helper: has_any_glass_edge() — binary check, more reliable than fraction
#        for deciding "does this sub-room touch any glass wall at all?"
#   4. Sub-rooms with zero glass → still excluded (furniture, columns, etc.)
#   5. Standalone rooms → always accepted (unchanged from v7)

# RULES (updated):
#   1. Actual polygon area/perimeter for all measurements
#   2. Sub-polygon inside parent with ANY glass edge → separate glass room ✓
#   3. Sub-polygon inside parent with ZERO glass → ignored (furniture etc.) ✗
#   4. Glass forming standalone enclosed space → new room
#   5. Shapes outside all rooms / not enclosed → ignored
#   6. Outer envelope → excluded
# """

# import streamlit as st
# import subprocess, os, math
# import ezdxf
# from ezdxf.math import Matrix44
# import numpy as np
# from shapely.geometry import LineString, Polygon, MultiLineString, Point
# from shapely.ops import polygonize, unary_union
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ──────────────────────────────────────────────────────
# st.set_page_config(page_title="CAD Room Extractor", layout="wide")
# st.markdown("""
# <style>
#   .main{background:#0f1117}
#   h1{color:#00d4ff;font-family:'Courier New',monospace}
#   .block-container{padding-top:2rem}
#   div[data-testid="metric-container"]{
#     background:#1a1f2e;border-radius:10px;
#     padding:10px;border:1px solid #00d4ff33}
# </style>
# """, unsafe_allow_html=True)
# st.title("🏗️ CAD Room Extractor + Heat Load Calculator")
# st.caption("DWG → DXF → Layer filter → Edge-tracked room detection → TR  [v8 — glass sub-room fix]")

# # ──────────────────────────────────────────────────────
# #  SIDEBAR
# # ──────────────────────────────────────────────────────
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     oda_path = st.text_input(
#         "ODAFileConverter Path",
#         r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")

#     st.divider()
#     st.markdown("**Layers**")
#     wall_input  = st.text_area("Wall layers",  "MAIN\nMAIN-4\nCEN-1")
#     glass_input = st.text_area("Glass layers", "GLASS")
#     WALL_LAYERS  = {l.strip().upper() for l in wall_input.strip().splitlines()  if l.strip()}
#     GLASS_LAYERS = {l.strip().upper() for l in glass_input.strip().splitlines() if l.strip()}
#     ALLOWED_LAYERS = WALL_LAYERS | GLASS_LAYERS

#     st.divider()
#     st.markdown("**Heat Load**")
#     H               = st.number_input("Room Height (m)",    value=3.0,  step=0.1)
#     U_wall          = st.number_input("Wall U-Value",       value=1.8,  step=0.1)
#     U_glass         = st.number_input("Glass U-Value",      value=5.8,  step=0.1)
#     DT              = st.number_input("ΔT (°C)",            value=10,   step=1)
#     people_per_room = st.number_input("People / Room",      value=2,    step=1)
#     Q_person        = st.number_input("Heat/Person (W)",    value=75,   step=5)

#     st.divider()
#     st.markdown("**Room Filtering**")
#     min_area_m2 = st.number_input("Min Room Area (m²)",  value=2.0,   step=0.5)
#     max_area_m2 = st.number_input("Max Room Area (m²)",  value=300.0, step=10.0)
#     min_compact = st.number_input("Min Compactness",     value=0.04,  step=0.01)
#     max_aspect  = st.number_input("Max Aspect Ratio",    value=10.0,  step=0.5)

#     st.divider()
#     st.markdown("**Gap Closing**")
#     snap_tol   = st.number_input("Snap tolerance",   value=10.0, step=1.0)
#     bridge_tol = st.number_input("Bridge tolerance", value=80.0, step=5.0)

#     st.divider()
#     st.markdown("**Glass Detection**")
#     glass_edge_thresh = st.number_input(
#         "Glass room edge threshold (0–1)", value=0.15, step=0.05,
#         help="Standalone rooms: fraction of edges that must be glass to label as 'glass room'. "
#              "Sub-rooms inside a parent: ANY glass edge (>0) is enough to be kept as separate room.")
    
#     # NEW in v8: extra proximity multiplier for glass edge detection
#     glass_proximity_mult = st.number_input(
#         "Glass proximity multiplier", value=3.0, step=0.5, min_value=1.0, max_value=10.0,
#         help="Multiplier on snap_tol for glass edge detection. "
#              "Increase if glass walls are slightly offset from room edges. "
#              "Default 3.0 (= snap_tol × 3). Was 1.5 in v7.")

#     show_debug = st.checkbox("Show debug info",   value=True)
#     show_raw   = st.checkbox("Show raw geometry", value=False)

# # ──────────────────────────────────────────────────────
# #  FILE UPLOAD
# # ──────────────────────────────────────────────────────
# uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
# if not uploaded_file:
#     st.info("👆 Upload a DWG file to begin."); st.stop()

# dwg_filename = uploaded_file.name
# dwg_path     = os.path.join(os.getcwd(), dwg_filename)
# with open(dwg_path, "wb") as f:
#     f.write(uploaded_file.getbuffer())
# st.success(f"Uploaded: **{dwg_filename}**")

# # ──────────────────────────────────────────────────────
# #  ODA CONVERSION
# # ──────────────────────────────────────────────────────
# dxf_folder = os.path.join(os.getcwd(), "converted_dxf")
# os.makedirs(dxf_folder, exist_ok=True)
# with st.spinner("🔄 Converting DWG → DXF…"):
#     try:
#         subprocess.run(
#             [oda_path, os.getcwd(), dxf_folder, "ACAD2013", "DXF", "0", "1"],
#             check=True, capture_output=True, text=True, timeout=120)
#         st.success("✅ ODA conversion OK")
#     except subprocess.CalledProcessError as e:
#         st.error(f"ODA failed: {e.stderr}"); st.stop()
#     except FileNotFoundError:
#         st.error("ODAFileConverter.exe not found."); st.stop()

# dxf_name = dwg_filename.rsplit(".", 1)[0] + ".dxf"
# dxf_path = os.path.join(dxf_folder, dxf_name)
# if not os.path.exists(dxf_path):
#     hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
#     if not hits: st.error("No DXF found."); st.stop()
#     dxf_path = os.path.join(dxf_folder, hits[0])

# # ──────────────────────────────────────────────────────
# #  LOAD DXF
# # ──────────────────────────────────────────────────────
# try:
#     doc = ezdxf.readfile(dxf_path)
# except Exception as e:
#     st.error(f"DXF load: {e}"); st.stop()

# msp = doc.modelspace()
# all_layers_in_file = {layer.dxf.name for layer in doc.layers}

# if show_debug:
#     with st.expander("🔍 Layer debug", expanded=True):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown("**All layers in DXF:**")
#             for l in sorted(all_layers_in_file):
#                 tag = ("🟩 WALL"  if l.upper() in WALL_LAYERS
#                   else ("🟦 GLASS" if l.upper() in GLASS_LAYERS else "⚪"))
#                 st.markdown(f"{tag} `{l}`")
#         with c2:
#             missing = {l for l in ALLOWED_LAYERS
#                        if l not in {x.upper() for x in all_layers_in_file}}
#             if missing: st.warning(f"Not in file: {missing}")
#             else:       st.success("All layers found ✅")

# # ──────────────────────────────────────────────────────
# #  GEOMETRY EXTRACTION
# # ──────────────────────────────────────────────────────
# GEOM_TYPES = {"LINE","LWPOLYLINE","POLYLINE","ARC","CIRCLE","ELLIPSE","SPLINE"}

# def apply_m44(mat, x, y):
#     from ezdxf.math import Vec3
#     v = mat.transform(Vec3(x, y, 0))
#     return (v.x, v.y)

# def effective_layer(entity, parent_layer):
#     own = entity.dxf.get("layer", "0")
#     return parent_layer if own == "0" else own

# def build_matrix(ent):
#     try:
#         return ent.matrix44()
#     except Exception:
#         ix  = ent.dxf.insert.x; iy = ent.dxf.insert.y
#         rot = math.radians(ent.dxf.get("rotation", 0))
#         sx  = ent.dxf.get("xscale", 1); sy = ent.dxf.get("yscale", 1)
#         c, s = math.cos(rot), math.sin(rot)
#         return Matrix44([[sx*c,-sy*s,0,ix],[sx*s,sy*c,0,iy],[0,0,1,0],[0,0,0,1]])

# def ent_to_segments(entity, mat, layer):
#     """
#     Returns list of (LineString, is_glass, is_arc).
#     Breaks LWPOLYLINEs into individual segments for per-edge glass tracking.
#     """
#     t        = entity.dxftype()
#     is_glass = layer.upper() in GLASS_LAYERS
#     res      = []
#     try:
#         if t == "LINE":
#             p1 = apply_m44(mat, entity.dxf.start.x, entity.dxf.start.y)
#             p2 = apply_m44(mat, entity.dxf.end.x,   entity.dxf.end.y)
#             if p1 != p2:
#                 res.append((LineString([p1, p2]), is_glass, False))

#         elif t == "LWPOLYLINE":
#             pts = [apply_m44(mat, p[0], p[1]) for p in entity.get_points()]
#             closed = entity.closed or (len(pts) > 2 and pts[0] == pts[-1])
#             if closed and pts and pts[0] != pts[-1]:
#                 pts.append(pts[0])
#             for a, b in zip(pts[:-1], pts[1:]):
#                 if a != b:
#                     res.append((LineString([a, b]), is_glass, False))

#         elif t == "POLYLINE":
#             pts = [apply_m44(mat, v.dxf.location.x, v.dxf.location.y)
#                    for v in entity.vertices if hasattr(v.dxf, "location")]
#             for a, b in zip(pts[:-1], pts[1:]):
#                 if a != b:
#                     res.append((LineString([a, b]), is_glass, False))

#         elif t == "ARC":
#             cx, cy = entity.dxf.center.x, entity.dxf.center.y
#             r      = entity.dxf.radius
#             sa = math.radians(entity.dxf.start_angle)
#             ea = math.radians(entity.dxf.end_angle)
#             if ea <= sa: ea += 2 * math.pi
#             angles = np.linspace(sa, ea, 32)
#             pts = [apply_m44(mat, cx+r*math.cos(a), cy+r*math.sin(a)) for a in angles]
#             if len(pts) >= 2:
#                 for a, b in zip(pts[:-1], pts[1:]):
#                     if a != b:
#                         res.append((LineString([a, b]), is_glass, True))
#                 res.append((LineString([pts[0], pts[-1]]), False, True))

#         elif t == "CIRCLE":
#             cx, cy = entity.dxf.center.x, entity.dxf.center.y
#             r      = entity.dxf.radius
#             angles = np.linspace(0, 2*math.pi, 64)
#             pts = [apply_m44(mat, cx+r*math.cos(a), cy+r*math.sin(a)) for a in angles]
#             for a, b in zip(pts[:-1], pts[1:]):
#                 if a != b:
#                     res.append((LineString([a, b]), is_glass, False))

#         elif t == "SPLINE":
#             try:    raw = [(p[0],p[1]) for p in entity.control_points]
#             except: raw = []
#             if len(raw) < 2:
#                 try:    raw = [(p[0],p[1]) for p in entity.fit_points]
#                 except: raw = []
#             pts = [apply_m44(mat, x, y) for x, y in raw]
#             for a, b in zip(pts[:-1], pts[1:]):
#                 if a != b:
#                     res.append((LineString([a, b]), is_glass, False))
#     except Exception:
#         pass
#     return res

# def extract_all(layout, doc, allowed_up, parent_mat=None, parent_layer="0", depth=0):
#     if depth > 30: return []
#     if parent_mat is None: parent_mat = Matrix44()
#     out = []
#     for ent in layout:
#         et = ent.dxftype()
#         if et == "INSERT":
#             bname = ent.dxf.name
#             if bname not in doc.blocks: continue
#             ins_layer = ent.dxf.get("layer", parent_layer)
#             combined  = parent_mat @ build_matrix(ent)
#             out.extend(extract_all(doc.blocks[bname], doc, allowed_up,
#                                     combined, ins_layer, depth+1))
#         elif et in GEOM_TYPES:
#             eff = effective_layer(ent, parent_layer).upper()
#             if eff not in allowed_up: continue
#             out.extend(ent_to_segments(ent, parent_mat, eff))
#     return out

# with st.spinner("🔍 Extracting geometry…"):
#     raw = extract_all(msp, doc, ALLOWED_LAYERS)

# if not raw:
#     st.error("No geometry on allowed layers."); st.stop()

# wall_segs      = [g for (g, ig, ia) in raw if not ig and not ia]
# glass_segs     = [g for (g, ig, ia) in raw if ig  and not ia]
# arc_segs       = [g for (g, ig, ia) in raw if ia]
# all_segs_display = [g for (g,_,_) in raw]

# st.success(f"✅ {len(raw)} segments  "
#            f"(wall:{len(wall_segs)}  glass:{len(glass_segs)}  arc/chord:{len(arc_segs)})")

# if show_debug:
#     with st.expander("📊 Segment counts", expanded=True):
#         st.markdown(f"- 🟩 Wall segments: **{len(wall_segs)}**")
#         st.markdown(f"- 🟦 Glass segments: **{len(glass_segs)}**")
#         st.markdown(f"- 🚪 Arc/chord segments: **{len(arc_segs)}**")

# # ──────────────────────────────────────────────────────
# #  UNIT DETECTION
# # ──────────────────────────────────────────────────────
# all_xs = [c[0] for ls in wall_segs + glass_segs for c in ls.coords]
# span   = max(all_xs) - min(all_xs) if all_xs else 1
# unit_guess  = "mm" if span > 500 else "m"
# unit_factor = 1_000_000 if unit_guess == "mm" else 1.0
# unit_div    = 1000      if unit_guess == "mm" else 1
# st.info(f"📏 Units: **{unit_guess}** | Span: {span:.0f}")

# # ──────────────────────────────────────────────────────
# #  SNAP
# # ──────────────────────────────────────────────────────
# def node_snap_segs(segs, tol):
#     out = []
#     for ls in segs:
#         try:
#             coords = [(round(x/tol)*tol, round(y/tol)*tol) for x,y in ls.coords]
#             dedup  = [coords[0]]
#             for c in coords[1:]:
#                 if c != dedup[-1]: dedup.append(c)
#             if len(dedup) >= 2:
#                 out.append(LineString(dedup))
#         except Exception:
#             pass
#     return out

# wall_snapped  = node_snap_segs(wall_segs,  snap_tol)
# glass_snapped = node_snap_segs(glass_segs, snap_tol)
# arc_snapped   = node_snap_segs(arc_segs,   snap_tol)

# boundary_snapped = wall_snapped + glass_snapped + arc_snapped

# # ──────────────────────────────────────────────────────
# #  BRIDGE DANGLING ENDPOINTS
# # ──────────────────────────────────────────────────────
# def bridge_gaps(lines, tol):
#     from collections import defaultdict
#     ep = defaultdict(int)
#     for ls in lines:
#         coords = list(ls.coords)
#         ep[coords[0]]  += 1
#         ep[coords[-1]] += 1
#     dangling = [pt for pt, cnt in ep.items() if cnt == 1]
#     if not dangling: return []
#     bridges, used = [], set()
#     arr = np.array(dangling)
#     for i, pt in enumerate(dangling):
#         if i in used: continue
#         diffs = arr - np.array(pt)
#         dists = np.hypot(diffs[:,0], diffs[:,1])
#         dists[i] = np.inf
#         j = int(np.argmin(dists))
#         if dists[j] <= tol and j not in used:
#             bridges.append(LineString([pt, dangling[j]]))
#             used.add(i); used.add(j)
#     return bridges

# bridges = bridge_gaps(boundary_snapped, bridge_tol)
# lines_for_poly = boundary_snapped + bridges

# if show_debug:
#     st.info(f"Gap bridges: **{len(bridges)}**")

# # ──────────────────────────────────────────────────────
# #  POLYGONIZE
# # ──────────────────────────────────────────────────────
# with st.spinner("🔲 Polygonizing…"):
#     merged    = unary_union(lines_for_poly)
#     all_polys = list(polygonize(merged))

# all_polys.sort(key=lambda p: p.area, reverse=True)
# if show_debug:
#     st.info(f"Raw polygons: **{len(all_polys)}**")

# # ──────────────────────────────────────────────────────
# #  GLASS EDGE DETECTION  (v8 — improved proximity)
# # ──────────────────────────────────────────────────────
# glass_union = unary_union(glass_snapped) if glass_snapped else None

# def edge_glass_fraction(poly, glass_u, tol, proximity_mult):
#     """
#     Walk each edge of the polygon exterior.
#     An edge is 'glass' if its midpoint lies within (tol * proximity_mult) of any glass line.
#     Returns fraction (0.0–1.0) of perimeter that is glass.
    
#     v8 change: proximity_mult is now configurable (default 3.0 vs 1.5 in v7)
#     This catches glass walls that are slightly offset from polygon edges.
#     """
#     if glass_u is None or glass_u.is_empty:
#         return 0.0

#     coords     = list(poly.exterior.coords)
#     total_len  = 0.0
#     glass_len  = 0.0
#     check_dist = tol * proximity_mult   # v8: wider search radius

#     for i in range(len(coords) - 1):
#         p1, p2   = coords[i], coords[i+1]
#         seg_len  = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
#         # Use BOTH midpoint AND quarter-points for better coverage
#         # (v8 fix: midpoint alone missed some glass edges in corner rooms)
#         mid  = ((p1[0]+p2[0])/2,   (p1[1]+p2[1])/2)
#         qtr1 = ((p1[0]*3+p2[0])/4, (p1[1]*3+p2[1])/4)
#         qtr2 = ((p1[0]+p2[0]*3)/4, (p1[1]+p2[1]*3)/4)
#         total_len += seg_len

#         # Edge is glass if ANY of the 3 sample points is near glass
#         if (glass_u.distance(Point(mid))  <= check_dist or
#             glass_u.distance(Point(qtr1)) <= check_dist or
#             glass_u.distance(Point(qtr2)) <= check_dist):
#             glass_len += seg_len

#     if total_len == 0:
#         return 0.0
#     return min(glass_len / total_len, 1.0)


# def has_any_glass_edge(poly, glass_u, tol, proximity_mult):
#     """
#     Binary check: does this polygon have AT LEAST ONE edge near a glass line?
#     Used in v8 for sub-room detection — any glass contact = keep as separate room.
#     More reliable than relying purely on fraction for corner rooms.
#     """
#     if glass_u is None or glass_u.is_empty:
#         return False

#     coords     = list(poly.exterior.coords)
#     check_dist = tol * proximity_mult

#     for i in range(len(coords) - 1):
#         p1, p2 = coords[i], coords[i+1]
#         mid    = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
#         qtr1   = ((p1[0]*3+p2[0])/4, (p1[1]*3+p2[1])/4)
#         qtr2   = ((p1[0]+p2[0]*3)/4, (p1[1]+p2[1]*3)/4)
#         if (glass_u.distance(Point(mid))  <= check_dist or
#             glass_u.distance(Point(qtr1)) <= check_dist or
#             glass_u.distance(Point(qtr2)) <= check_dist):
#             return True
#     return False

# # ──────────────────────────────────────────────────────
# #  SHAPE QUALITY HELPERS
# # ──────────────────────────────────────────────────────
# def compactness(poly):
#     if poly.length == 0: return 0
#     return (4 * math.pi * poly.area) / (poly.length ** 2)

# def aspect_ratio(poly):
#     minx, miny, maxx, maxy = poly.bounds
#     w = maxx - minx; h = maxy - miny
#     if min(w, h) == 0: return 999
#     return max(w, h) / min(w, h)

# def is_outer_envelope(poly, all_p, threshold=0.35):
#     others = [p for p in all_p if p is not poly]
#     if not others: return False
#     n = sum(1 for p in others if poly.contains(p.centroid))
#     return (n / len(others)) >= threshold

# # ──────────────────────────────────────────────────────
# #  FILTER + CLASSIFY ROOMS  (v8 — fixed sub-room logic)
# #
# #  KEY CHANGE from v7:
# #    Sub-room inside parent:
# #      v7: kept only if gf >= glass_edge_thresh  (too strict for corner rooms)
# #      v8: kept if has_any_glass_edge() is True  (binary — any glass = keep)
# #           → glass_fraction still computed for labeling/heat-load purposes
# #           → Only purely wall-bounded sub-shapes (furniture etc.) are dropped
# # ──────────────────────────────────────────────────────
# outer_ids = set()
# for i, poly in enumerate(all_polys[:5]):
#     if is_outer_envelope(poly, all_polys):
#         outer_ids.add(id(poly))
#         if show_debug:
#             st.info(f"🏛️ Outer envelope ({poly.area/unit_factor:.1f} m²) → excluded")
#         break

# # Pass 1: filter by area + shape + not outer
# candidates = []
# for poly in all_polys:
#     if id(poly) in outer_ids: continue
#     area_m2 = poly.area / unit_factor
#     if area_m2 < min_area_m2 or area_m2 > max_area_m2: continue
#     if compactness(poly) < min_compact: continue
#     if aspect_ratio(poly) > max_aspect: continue
#     gf      = edge_glass_fraction(poly, glass_union, snap_tol, glass_proximity_mult)
#     has_gls = has_any_glass_edge(poly, glass_union, snap_tol, glass_proximity_mult)
#     candidates.append((poly, gf, has_gls))

# if show_debug:
#     st.info(f"Candidates after shape filter: **{len(candidates)}**")

# # Sort large → small
# candidates.sort(key=lambda x: x[0].area, reverse=True)

# accepted = []   # list of (poly, glass_fraction, is_glass_room)

# def is_mostly_inside(small, large, tol=0.90):
#     try:
#         return small.intersection(large).area / small.area >= tol
#     except Exception:
#         return False

# for (poly, gf, has_gls) in candidates:
#     parent = next(
#         ((ap, apgf, apig) for (ap, apgf, apig) in accepted
#          if is_mostly_inside(poly, ap)),
#         None
#     )

#     if parent is None:
#         # ── Standalone enclosed space → always a room ──
#         is_glass_room = gf >= glass_edge_thresh
#         accepted.append((poly, gf, is_glass_room))
#     else:
#         # ── Sub-polygon inside a parent room ──
#         # v8 FIX: Use binary has_any_glass_edge instead of fraction threshold.
#         # This correctly handles corner rooms where glass is on only 1 side,
#         # giving a low fraction that v7's threshold would reject.
#         if has_gls:
#             # Has at least one glass wall → separate glass-partition room
#             accepted.append((poly, gf, True))
#         # else: no glass contact at all → furniture/column/artifact → ignore

# # Deduplicate near-identical polygons
# def deduplicate(room_list):
#     keep  = list(room_list)
#     flags = [False] * len(keep)
#     for i in range(len(keep)):
#         if flags[i]: continue
#         for j in range(i+1, len(keep)):
#             if flags[j]: continue
#             a, b = keep[i][0], keep[j][0]
#             if abs(a.area - b.area) / max(a.area, b.area) > 0.06: continue
#             try:
#                 inter = a.intersection(b).area
#                 if inter / min(a.area, b.area) >= 0.90:
#                     flags[j] = True
#             except Exception:
#                 pass
#     return [r for r, f in zip(keep, flags) if not f]

# accepted = deduplicate(accepted)

# # Sort top-left → bottom-right
# accepted.sort(key=lambda r: (-r[0].centroid.y, r[0].centroid.x))

# n_glass = sum(1 for _, _, ig in accepted if ig)
# n_wall  = len(accepted) - n_glass

# if show_debug:
#     st.success(f"Final rooms: **{len(accepted)}**  "
#                f"({n_wall} wall-bounded  +  {n_glass} glass-bounded)")

# if not accepted:
#     st.error("❌ No rooms found. Try increasing Bridge tolerance or lowering Min Area.")
#     fig0, ax0 = plt.subplots(figsize=(14, 7))
#     fig0.patch.set_facecolor("#0f1117"); ax0.set_facecolor("#0f1117")
#     for ls in lines_for_poly[:4000]:
#         xs, ys = ls.xy
#         ax0.plot(xs, ys, color="#00d4ff", lw=0.5, alpha=0.5)
#     ax0.set_aspect("equal")
#     ax0.set_title("Raw geometry — no rooms found", color="white")
#     st.pyplot(fig0); plt.close()
#     st.stop()

# st.success(f"✅ **{len(accepted)} rooms** detected  "
#            f"({n_glass} glass  +  {n_wall} wall-bounded)")

# # ──────────────────────────────────────────────────────
# #  HEAT LOAD
# # ──────────────────────────────────────────────────────
# COLORS = [
#     "#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
#     "#DDA0DD","#98D8C8","#F7DC6F","#82E0AA","#F1948A",
#     "#85C1E9","#F0B27A","#C39BD3","#76D7C4","#F9E79F",
#     "#AED6F1","#A9DFBF","#FAD7A0","#D2B4DE","#FFB3BA",
# ]

# room_data = []
# for i, (poly, gf, is_glass_room) in enumerate(accepted):
#     area_m2  = poly.area   / unit_factor
#     perim_m  = poly.length / unit_div
#     minx, miny, maxx, maxy = poly.bounds
#     length_m  = (maxx - minx) / unit_div
#     breadth_m = (maxy - miny) / unit_div

#     glass_perim_m = perim_m * gf
#     wall_perim_m  = perim_m * (1 - gf)
#     wall_area_m2  = wall_perim_m  * H
#     glass_area_m2 = glass_perim_m * H

#     q_wall   = wall_area_m2  * U_wall  * DT
#     q_glass  = glass_area_m2 * U_glass * DT
#     q_people = people_per_room * Q_person
#     q_total  = q_wall + q_glass + q_people
#     tr       = q_total / 3517

#     room_data.append({
#         "Room":             f"Room {i+1}",
#         "Type":             "🔵 Glass" if is_glass_room else "🟩 Wall",
#         "Area (m²)":        round(area_m2,      3),
#         "Perimeter (m)":    round(perim_m,       3),
#         "Length ref (m)":   round(length_m,      2),
#         "Breadth ref (m)":  round(breadth_m,     2),
#         "Glass % edge":     round(gf * 100,      1),
#         "Wall Area (m²)":   round(wall_area_m2,  3),
#         "Glass Area (m²)":  round(glass_area_m2, 3),
#         "Q_wall (W)":       round(q_wall,        1),
#         "Q_glass (W)":      round(q_glass,       1),
#         "Q_people (W)":     round(q_people,      1),
#         "Q_total (W)":      round(q_total,       1),
#         "TR":               round(tr,            3),
#         "_poly":            poly,
#         "_gf":              gf,
#         "_is_glass":        is_glass_room,
#         "_color":           "#00d4ff" if is_glass_room else COLORS[i % len(COLORS)],
#     })

# df           = pd.DataFrame(room_data)
# display_cols = [c for c in df.columns if not c.startswith("_")]

# # ──────────────────────────────────────────────────────
# #  FLOOR PLAN
# # ──────────────────────────────────────────────────────
# st.subheader("🗺️ Detected Floor Plan")
# fig, ax = plt.subplots(figsize=(20, 11))
# fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
# ax.tick_params(colors="#888")
# for sp in ax.spines.values(): sp.set_edgecolor("#333")

# if show_raw:
#     for ls in all_segs_display:
#         xs, ys = ls.xy
#         ax.plot(xs, ys, color="#1e3a50", lw=0.25, alpha=0.35)

# for ls in wall_snapped:
#     xs, ys = ls.xy
#     ax.plot(xs, ys, color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

# for ls in glass_snapped:
#     xs, ys = ls.xy
#     ax.plot(xs, ys, color="#00d4ff", lw=1.4, alpha=0.75, zorder=2)

# for ls in bridges:
#     xs, ys = ls.xy
#     ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.7, zorder=2)

# for row in room_data:
#     poly   = row["_poly"]
#     color  = row["_color"]
#     ig     = row["_is_glass"]
#     gf     = row["_gf"]
#     xs, ys = poly.exterior.xy
#     ax.fill(xs, ys, alpha=0.40 if ig else 0.22, color=color, zorder=3)
#     ax.plot(xs, ys, color="#00d4ff" if ig else color,
#             lw=2.5 if ig else 1.6, zorder=4)
#     cx, cy = poly.centroid.x, poly.centroid.y
#     label  = (f"{'🔵' if ig else '🟩'} {row['Room']}\n"
#               f"{row['Area (m²)']} m²\n"
#               f"TR:{row['TR']}")
#     if gf > 0.05:
#         label += f"\n{row['Glass % edge']}% glass"
#     ax.text(cx, cy, label,
#             ha="center", va="center", fontsize=6.5,
#             color="white", fontfamily="monospace", zorder=5,
#             bbox=dict(boxstyle="round,pad=0.28",
#                       facecolor="#000000dd",
#                       edgecolor="#00d4ff" if ig else color,
#                       linewidth=1.3 if ig else 0.5))

# ax.set_aspect("equal")
# ax.set_title(
#     f"Floor Plan — {len(accepted)} rooms  "
#     f"({n_wall} wall  +  {n_glass} glass-bounded)  "
#     f"[v8 — glass sub-room fix | prox×{glass_proximity_mult}]",
#     color="#00d4ff", fontfamily="monospace", fontsize=10)
# ax.grid(True, color="#1a2a3a", lw=0.25)
# ax.legend(handles=[
#     mpatches.Patch(color="#3a6a8a", label="Wall lines"),
#     mpatches.Patch(color="#00d4ff", label="Glass lines"),
#     mpatches.Patch(color="#ff4444", label=f"Bridges ({len(bridges)})"),
#     mpatches.Patch(color="#00d4ff", alpha=0.4, label="Glass room fill"),
# ], loc="upper right", facecolor="#0f1117", edgecolor="#444",
#    labelcolor="white", fontsize=9)
# st.pyplot(fig); plt.close()

# # ──────────────────────────────────────────────────────
# #  TABLE
# # ──────────────────────────────────────────────────────
# st.subheader("📊 Room Measurements & Heat Load")
# st.caption("✅ Area and Perimeter = actual polygon geometry (not bounding box)")
# st.dataframe(
#     df[display_cols].style
#       .format({
#           "TR":            "{:.3f}",
#           "Area (m²)":     "{:.3f}",
#           "Perimeter (m)": "{:.3f}",
#           "Q_total (W)":   "{:.1f}",
#           "Glass % edge":  "{:.1f}",
#       })
#       .background_gradient(subset=["TR"],           cmap="YlOrRd")
#       .background_gradient(subset=["Glass % edge"], cmap="Blues")
#       .background_gradient(subset=["Area (m²)"],    cmap="Greens"),
#     use_container_width=True)

# # ──────────────────────────────────────────────────────
# #  METRICS
# # ──────────────────────────────────────────────────────
# st.subheader("🌡️ Heat Load Summary")
# total_area = df["Area (m²)"].sum()
# total_kw   = df["Q_total (W)"].sum() / 1000
# total_tr   = df["TR"].sum()

# c1,c2,c3,c4,c5 = st.columns(5)
# c1.metric("🏠 Total Rooms",  len(accepted))
# c2.metric("🔵 Glass Rooms",  n_glass)
# c3.metric("📐 Total Area",   f"{total_area:.2f} m²")
# c4.metric("⚡ Total Load",   f"{total_kw:.2f} kW")
# c5.metric("❄️ Total TR",     f"{total_tr:.2f} TR")

# st.subheader("📈 TR per Room")
# fig2, ax2 = plt.subplots(figsize=(max(10, len(accepted)), 4))
# fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
# ax2.tick_params(colors="#888")
# for sp in ax2.spines.values(): sp.set_edgecolor("#333")
# bars = ax2.bar(df["Room"], df["TR"],
#                color=[r["_color"] for r in room_data],
#                alpha=0.85, edgecolor="#ffffff11")
# for bar, row in zip(bars, room_data):
#     ax2.text(bar.get_x()+bar.get_width()/2,
#              bar.get_height()+total_tr*0.003,
#              f"{row['TR']:.3f}",
#              ha="center", va="bottom", color="white", fontsize=8)
# ax2.set_xlabel("Room", color="#aaa")
# ax2.set_ylabel("TR", color="#aaa")
# ax2.set_title("TR per Room  (🔵 = glass-bounded)", color="#00d4ff", fontfamily="monospace")
# ax2.grid(axis="y", color="#1a2a3a", lw=0.4)
# plt.xticks(rotation=45, ha="right", color="#aaa")
# st.pyplot(fig2); plt.close()

# st.download_button("⬇️ Download CSV",
#                    df[display_cols].to_csv(index=False),
#                    "rooms_heat_load.csv", "text/csv")
# st.divider()
# st.caption(
#     f"Wall:{','.join(sorted(WALL_LAYERS))}  Glass:{','.join(sorted(GLASS_LAYERS))}  |  "
#     f"v8  |  {len(accepted)} rooms ({n_glass} glass)  |  {total_tr:.2f} TR  |  "
#     f"snap={snap_tol}  bridge={bridge_tol}  glass_thresh={glass_edge_thresh}  "
#     f"prox_mult={glass_proximity_mult}")

























# """
# CAD Room Extractor + Heat Load  v11
# =====================================
# ROOM DETECTION ENGINE — merged from reference implementation

# KEY APPROACH (from reference code):
#   1. Extract wall geometry per layer using ezdxf path flattening
#   2. Detect scale from $INSUNITS header
#   3. Merge wall lines → find dangling endpoints
#   4. Orthogonal bridging: seal open doorways/archways by connecting
#      close endpoints (gap <= max_door_width) that are axis-aligned
#   5. polygonize the fully-bridged geometry → raw room polygons
#   6. Filter by area, remove outer envelope, remove nested duplicates
#   7. Furniture/door layers → cluster into objects → place inside rooms

# MODE A: Glass-partition (office) → glass edge tracking (v8 logic)
# MODE B: Base-layer only (residential) → endpoint bridging polygonize (v11)
# """

# import streamlit as st
# import subprocess, os, math
# import ezdxf
# from ezdxf import path as dxf_path
# from ezdxf.math import Matrix44
# import numpy as np
# from shapely.geometry import LineString, Polygon, Point, MultiLineString, MultiPoint
# from shapely.ops import polygonize, unary_union
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import networkx as nx

# # Auto-install networkx if missing
# import importlib, subprocess as _sp
# for _pkg, _imp in [('networkx','networkx')]:
#     if importlib.util.find_spec(_imp) is None:
#         _sp.run(['pip','install',_pkg,'--break-system-packages','-q'], check=False)

# # ──────────────────────────────────────────────────────
# st.set_page_config(page_title="CAD Room Extractor", layout="wide")
# st.markdown("""
# <style>
#   .main{background:#0f1117}
#   h1{color:#00d4ff;font-family:'Courier New',monospace}
#   .block-container{padding-top:2rem}
#   div[data-testid="metric-container"]{
#     background:#1a1f2e;border-radius:10px;
#     padding:10px;border:1px solid #00d4ff33}
# </style>
# """, unsafe_allow_html=True)
# st.title("🏗️ CAD Room Extractor + Heat Load Calculator")
# st.caption("v11 — Endpoint-bridging polygonize engine | Glass-partition + Base-layer unified")

# # ──────────────────────────────────────────────────────
# #  SIDEBAR
# # ──────────────────────────────────────────────────────
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     oda_path = st.text_input(
#         "ODAFileConverter Path",
#         r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")

#     st.divider()
#     st.markdown("**Layers**")
#     wall_input  = st.text_area("Wall / Base layers",
#         "MAIN\nMAIN-4\nCEN-1\nA-WALL\nWALLS\nWall\nWalls\nBase\n0")
#     glass_input = st.text_area("Glass layers (leave blank = base-only mode)", "GLASS")
#     furn_input  = st.text_area("Furniture / Door layers (optional)",
#         "Furniture\nDoors\nFURNITURE\nDOOR")

#     st.divider()
#     st.markdown("**Detection Mode**")
#     mode_override = st.selectbox(
#         "Mode",
#         ["Auto-detect", "Force Glass-partition (Mode A)", "Force Base-layer only (Mode B)"],
#         index=0)

#     st.divider()
#     st.markdown("**Heat Load**")
#     H               = st.number_input("Room Height (m)",    value=3.0, step=0.1)
#     U_wall          = st.number_input("Wall U-Value",       value=1.8, step=0.1)
#     U_glass         = st.number_input("Glass U-Value",      value=5.8, step=0.1)
#     DT              = st.number_input("ΔT (°C)",            value=10,  step=1)
#     people_per_room = st.number_input("People / Room",      value=2,   step=1)
#     Q_person        = st.number_input("Heat/Person (W)",    value=75,  step=5)

#     st.divider()
#     st.markdown("**Room Filtering**")
#     min_area_m2 = st.number_input("Min Room Area (m²)", value=2.0,   step=0.5)
#     max_area_m2 = st.number_input("Max Room Area (m²)", value=500.0, step=10.0)
#     outer_area_pct = st.number_input(
#         "Outer envelope threshold (%)", value=25.0, step=5.0,
#         help="Polygons larger than this % of total bounding box area → excluded as outer border. "
#              "Lower = more aggressive exclusion. Default 25%.")

#     st.divider()
#     st.markdown("**Shape Validation (Mode B)**")
#     min_solidity = st.number_input(
#         "Min solidity (0–1)", value=0.50, step=0.05, min_value=0.0, max_value=1.0,
#         help="Area ÷ convex hull area. Real rooms ≥ 0.5. Stair artifacts and "
#              "corridor slivers are jagged/hollow → lower solidity → excluded.")
#     max_aspect = st.number_input(
#         "Max aspect ratio", value=15.0, step=1.0,
#         help="Width÷Height of bounding box. Extreme thin strips = border artifacts → excluded.")
#     max_interior_walls = st.number_input(
#         "Max interior wall segments", value=8, step=1,
#         help="Wall segments fully INSIDE a polygon. Real rooms have walls on their "
#              "boundary, not cutting through. High count = artifact polygon.")
#     exclude_stairs = st.checkbox("Exclude staircase regions", value=True)
#     stair_parallel_min = st.number_input(
#         "Min parallel lines to flag as stair", value=4, step=1,
#         help="If a polygon contains ≥ this many parallel interior line segments, "
#              "it's treated as a staircase and excluded.")
#     stair_angle_tol = st.number_input(
#         "Stair angle tolerance (°)", value=8.0, step=1.0,
#         help="Two lines are 'parallel' if their angles differ by less than this.")
#     max_stair_area_m2 = st.number_input(
#         "Max staircase area (m²)", value=20.0, step=1.0,
#         help="Only polygons smaller than this are checked for stairs. "
#              "Large rooms won't be excluded even if they contain parallel lines.")

#     st.divider()
#     st.markdown("**Bridging (Mode B — gap sealing)**")
#     gap_close_tol = st.number_input(
#         "Snap/merge tolerance (mm)", value=15.0, step=5.0,
#         help="Lines within this distance are merged into one wall cluster.")
#     max_door_width = st.number_input(
#         "Max door/archway width (mm)", value=1500.0, step=100.0,
#         help="Endpoints closer than this and axis-aligned → bridged to seal the opening.")
#     min_wall_len   = st.number_input(
#         "Min wall segment length (mm)", value=200.0, step=50.0,
#         help="Segments shorter than this are ignored (noise/hatching).")

#     st.divider()
#     st.markdown("**Glass Detection (Mode A only)**")
#     glass_edge_thresh    = st.number_input("Glass edge threshold", value=0.15, step=0.05)
#     glass_proximity_mult = st.number_input("Glass proximity mult", value=3.0,  step=0.5,
#                                             min_value=1.0, max_value=10.0)
#     snap_tol_a = st.number_input("Snap tolerance (Mode A)", value=10.0, step=1.0)
#     bridge_tol_a = st.number_input("Bridge tolerance (Mode A)", value=80.0, step=5.0)

#     st.divider()
#     show_debug = st.checkbox("Show debug info",   value=True)
#     show_raw   = st.checkbox("Show raw geometry", value=False)

# # ──────────────────────────────────────────────────────
# #  FILE UPLOAD
# # ──────────────────────────────────────────────────────
# uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
# if not uploaded_file:
#     st.info("👆 Upload a DWG file to begin."); st.stop()

# dwg_filename = uploaded_file.name
# dwg_path_file = os.path.join(os.getcwd(), dwg_filename)
# with open(dwg_path_file, "wb") as f:
#     f.write(uploaded_file.getbuffer())
# st.success(f"Uploaded: **{dwg_filename}**")

# # ──────────────────────────────────────────────────────
# #  ODA CONVERSION
# # ──────────────────────────────────────────────────────
# dxf_folder = os.path.join(os.getcwd(), "converted_dxf")
# os.makedirs(dxf_folder, exist_ok=True)
# with st.spinner("🔄 Converting DWG → DXF…"):
#     try:
#         subprocess.run(
#             [oda_path, os.getcwd(), dxf_folder, "ACAD2013", "DXF", "0", "1"],
#             check=True, capture_output=True, text=True, timeout=120)
#         st.success("✅ ODA conversion OK")
#     except subprocess.CalledProcessError as e:
#         st.error(f"ODA failed: {e.stderr}"); st.stop()
#     except FileNotFoundError:
#         st.error("ODAFileConverter.exe not found."); st.stop()

# dxf_name = dwg_filename.rsplit(".", 1)[0] + ".dxf"
# dxf_path_conv = os.path.join(dxf_folder, dxf_name)
# if not os.path.exists(dxf_path_conv):
#     hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
#     if not hits: st.error("No DXF found."); st.stop()
#     dxf_path_conv = os.path.join(dxf_folder, hits[0])

# # ──────────────────────────────────────────────────────
# #  LOAD DXF + LAYER DISCOVERY
# # ──────────────────────────────────────────────────────
# try:
#     doc = ezdxf.readfile(dxf_path_conv)
# except Exception as e:
#     st.error(f"DXF load: {e}"); st.stop()

# msp = doc.modelspace()
# all_layers_in_file = {layer.dxf.name for layer in doc.layers}

# # Count entities per layer
# layer_entity_count = {}
# for ent in msp:
#     l = ent.dxf.get("layer", "0")
#     layer_entity_count[l] = layer_entity_count.get(l, 0) + 1

# sorted_layers = sorted(all_layers_in_file,
#     key=lambda l: layer_entity_count.get(l, 0), reverse=True)
# active_layers = [l for l in sorted_layers if layer_entity_count.get(l, 0) > 0]

# WALL_LAYERS  = {l.strip().upper() for l in wall_input.strip().splitlines()  if l.strip()}
# GLASS_LAYERS = {l.strip().upper() for l in glass_input.strip().splitlines() if l.strip()}
# FURN_LAYERS  = {l.strip().upper() for l in furn_input.strip().splitlines()  if l.strip()}

# matched_wall  = WALL_LAYERS  & {x.upper() for x in all_layers_in_file}
# matched_glass = GLASS_LAYERS & {x.upper() for x in all_layers_in_file}
# matched_furn  = FURN_LAYERS  & {x.upper() for x in all_layers_in_file}

# # Entity count on matched wall layers
# matched_wall_ents = sum(layer_entity_count.get(l,0)
#     for l in all_layers_in_file if l.upper() in matched_wall)
# total_ents = sum(layer_entity_count.values()) or 1
# poor_match = matched_wall_ents < total_ents * 0.20

# NON_WALL_KW = {"furniture","plant","planter","text","vp","defpoint",
#                "dimension","dim","hatch","annotation","title","border",
#                "viewport","pplne","electrical","elec","plumbing","mech","door"}

# # ── Layer selector ──
# with st.expander("🔍 Layer debug & selector", expanded=True):
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**Layers in DXF (by entity count):**")
#         for l in sorted_layers:
#             cnt = layer_entity_count.get(l, 0)
#             tag = ("🟩 WALL" if l.upper() in WALL_LAYERS
#               else ("🟦 GLASS" if l.upper() in GLASS_LAYERS
#               else ("🪑 FURN"  if l.upper() in FURN_LAYERS else "⚪")))
#             st.markdown(f"{tag} `{l}` — {cnt} entities")
#     with col2:
#         if matched_wall and not poor_match:
#             st.success(f"Wall layers matched: {sorted(matched_wall)} ✅")
#         else:
#             st.warning("⚠️ Poor or no wall layer match — select below")

#         smart_def = [l for l in active_layers
#                      if not any(kw in l.lower() for kw in NON_WALL_KW)]
#         if not smart_def and active_layers:
#             smart_def = [active_layers[0]]

#         sel_wall = st.multiselect("Wall / Base layers", options=active_layers,
#             default=smart_def if poor_match else
#                     [l for l in active_layers if l.upper() in matched_wall],
#             key="sel_wall")
#         sel_glass = st.multiselect("Glass layers (optional)", options=active_layers,
#             default=[l for l in active_layers if l.upper() in matched_glass],
#             key="sel_glass")
#         sel_furn = st.multiselect("Furniture/Door layers (optional)", options=active_layers,
#             default=[l for l in active_layers if l.upper() in matched_furn],
#             key="sel_furn")

#         if sel_wall:
#             WALL_LAYERS  = {l.upper() for l in sel_wall}
#             GLASS_LAYERS = {l.upper() for l in sel_glass}
#             FURN_LAYERS  = {l.upper() for l in sel_furn}
#             matched_wall  = WALL_LAYERS
#             matched_glass = GLASS_LAYERS
#             matched_furn  = FURN_LAYERS
#             st.success(f"✅ Using: {sorted(WALL_LAYERS)}")
#         elif active_layers:
#             WALL_LAYERS  = {l.upper() for l in active_layers}
#             GLASS_LAYERS = set()
#             FURN_LAYERS  = set()
#             matched_wall  = WALL_LAYERS

# # ──────────────────────────────────────────────────────
# #  MODE DETECTION
# # ──────────────────────────────────────────────────────
# glass_found = bool(matched_glass)
# if mode_override == "Force Glass-partition (Mode A)":
#     USE_GLASS_MODE = True
# elif mode_override == "Force Base-layer only (Mode B)":
#     USE_GLASS_MODE = False
# else:
#     USE_GLASS_MODE = glass_found and bool(GLASS_LAYERS)

# mode_label = "🔵 Mode A: Glass-partition" if USE_GLASS_MODE else "🟩 Mode B: Base-layer (endpoint-bridging)"
# st.info(f"**Detection Mode:** {mode_label}")

# # ──────────────────────────────────────────────────────
# #  SCALE DETECTION  (from reference code $INSUNITS)
# # ──────────────────────────────────────────────────────
# insunits  = doc.header.get('$INSUNITS', 0)
# scale_map = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}
# unit_names= {0:"Unitless",1:"Inches",2:"Feet",4:"mm",5:"cm",6:"m"}
# DRAWING_SCALE = scale_map.get(insunits, 1.0)
# st.info(f"📏 $INSUNITS={insunits} ({unit_names.get(insunits,'?')}) → scale×{DRAWING_SCALE}")

# # For heat-load unit conversion (all geometry will be in mm after scaling)
# # unit_factor: mm² → m²  = 1,000,000
# # unit_div:    mm  → m   = 1,000
# unit_factor = 1_000_000.0
# unit_div    = 1_000.0

# # ──────────────────────────────────────────────────────
# #  GEOMETRY EXTRACTION  (reference-style path flattening)
# # ──────────────────────────────────────────────────────
# def process_entity(entity, raw_lines, scale, exclude_arcs=False):
#     """Flatten any DXF entity to line segments using ezdxf path engine."""
#     if entity.dxftype() == 'INSERT':
#         try:
#             for sub in entity.virtual_entities():
#                 process_entity(sub, raw_lines, scale, exclude_arcs)
#         except Exception:
#             pass
#         return
#     if exclude_arcs and entity.dxftype() == 'ARC':
#         return
#     try:
#         p     = dxf_path.make_path(entity)
#         pts   = list(p.flattening(distance=0.1))
#         for i in range(len(pts) - 1):
#             s = (round(pts[i].x   * scale, 1), round(pts[i].y   * scale, 1))
#             e = (round(pts[i+1].x * scale, 1), round(pts[i+1].y * scale, 1))
#             if s != e:
#                 raw_lines.append([s, e])
#     except Exception:
#         pass

# ALLOWED_LAYERS = WALL_LAYERS | GLASS_LAYERS | FURN_LAYERS

# with st.spinner("🔍 Extracting geometry…"):
#     raw_wall_lines  = []
#     raw_glass_lines = []
#     raw_furn_lines  = []

#     for ent in msp.query('LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE'):
#         layer_up = ent.dxf.get("layer","0").upper()
#         if layer_up in WALL_LAYERS:
#             process_entity(ent, raw_wall_lines,  DRAWING_SCALE,
#                            exclude_arcs=(not USE_GLASS_MODE))
#         elif layer_up in GLASS_LAYERS:
#             process_entity(ent, raw_glass_lines, DRAWING_SCALE, False)
#         elif layer_up in FURN_LAYERS:
#             process_entity(ent, raw_furn_lines,  DRAWING_SCALE, False)

# if not raw_wall_lines:
#     st.error(f"❌ No geometry on wall layers {sorted(WALL_LAYERS)}. "
#              f"File layers: {sorted(all_layers_in_file)}"); st.stop()

# st.success(f"✅ wall:{len(raw_wall_lines)}  glass:{len(raw_glass_lines)}  furn:{len(raw_furn_lines)}")

# # ──────────────────────────────────────────────────────
# #  FURNITURE → OBJECTS  (reference code)
# # ──────────────────────────────────────────────────────
# def process_furniture_to_objects(furn_lines, gap_tol=50):
#     if not furn_lines: return []
#     from shapely.strtree import STRtree
#     shapely_lines = [LineString(l) for l in furn_lines]
#     buffered      = [l.buffer(gap_tol) for l in shapely_lines]
#     tree = STRtree(buffered)
#     G = nx.Graph(); G.add_nodes_from(range(len(furn_lines)))
#     for i, poly in enumerate(buffered):
#         for j in tree.query(poly):
#             if i != j and poly.intersects(buffered[j]):
#                 G.add_edge(i, j)
#     objects_data = []
#     for comp in nx.connected_components(G):
#         lines = [furn_lines[idx] for idx in comp]
#         mls   = MultiLineString([LineString(l) for l in lines])
#         minx, miny, maxx, maxy = mls.bounds
#         objects_data.append({
#             "object_id": f"Obj {len(objects_data)+1}",
#             "length": round(maxx - minx, 2),
#             "width":  round(maxy - miny, 2),
#             "center_x": round(mls.centroid.x, 2),
#             "center_y": round(mls.centroid.y, 2),
#             "point":  mls.centroid,
#         })
#     return objects_data

# with st.spinner("🪑 Processing furniture/door objects…"):
#     extracted_objects = process_furniture_to_objects(raw_furn_lines)
# if show_debug:
#     st.info(f"Furniture objects detected: **{len(extracted_objects)}**")

# # ──────────────────────────────────────────────────────
# #  GLASS EDGE HELPERS  (Mode A)
# # ──────────────────────────────────────────────────────
# glass_segs   = [LineString(l) for l in raw_glass_lines]
# glass_union  = unary_union(glass_segs) if glass_segs else None

# def edge_glass_fraction(poly, glass_u, tol, mult):
#     if glass_u is None or glass_u.is_empty: return 0.0
#     coords = list(poly.exterior.coords)
#     total = gls = 0.0
#     cd = tol * mult
#     for i in range(len(coords)-1):
#         p1,p2   = coords[i], coords[i+1]
#         seg_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
#         mid     = ((p1[0]+p2[0])/2,   (p1[1]+p2[1])/2)
#         q1      = ((p1[0]*3+p2[0])/4, (p1[1]*3+p2[1])/4)
#         q2      = ((p1[0]+p2[0]*3)/4, (p1[1]+p2[1]*3)/4)
#         total  += seg_len
#         if any(glass_u.distance(Point(p)) <= cd for p in (mid,q1,q2)):
#             gls += seg_len
#     return 0.0 if total==0 else min(gls/total, 1.0)

# def has_any_glass_edge(poly, glass_u, tol, mult):
#     if glass_u is None or glass_u.is_empty: return False
#     coords = list(poly.exterior.coords)
#     cd = tol * mult
#     for i in range(len(coords)-1):
#         p1,p2 = coords[i], coords[i+1]
#         mid   = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
#         q1    = ((p1[0]*3+p2[0])/4, (p1[1]*3+p2[1])/4)
#         q2    = ((p1[0]+p2[0]*3)/4, (p1[1]+p2[1]*3)/4)
#         if any(glass_u.distance(Point(p)) <= cd for p in (mid,q1,q2)):
#             return True
#     return False

# # ──────────────────────────────────────────────────────
# #  MODE A — Glass-partition polygonize  (v8 engine)
# # ──────────────────────────────────────────────────────
# def detect_rooms_mode_a(raw_wall_lines, glass_union,
#                          snap_tol, bridge_tol,
#                          glass_edge_thresh, glass_proximity_mult,
#                          min_area_m2, max_area_m2, unit_factor):

#     def node_snap(segs, tol):
#         out = []
#         for ls in segs:
#             try:
#                 coords = [(round(x/tol)*tol, round(y/tol)*tol) for x,y in ls.coords]
#                 dedup  = [coords[0]]
#                 for c in coords[1:]:
#                     if c != dedup[-1]: dedup.append(c)
#                 if len(dedup) >= 2:
#                     out.append(LineString(dedup))
#             except Exception: pass
#         return out

#     def bridge_gaps(lines, tol):
#         from collections import defaultdict
#         ep = defaultdict(int)
#         for ls in lines:
#             cs = list(ls.coords)
#             ep[cs[0]] += 1; ep[cs[-1]] += 1
#         dangling = [pt for pt,cnt in ep.items() if cnt==1]
#         if not dangling: return []
#         bridges, used = [], set()
#         arr = np.array(dangling)
#         for i,pt in enumerate(dangling):
#             if i in used: continue
#             diffs = arr - np.array(pt)
#             dists = np.hypot(diffs[:,0],diffs[:,1])
#             dists[i] = np.inf
#             j = int(np.argmin(dists))
#             if dists[j] <= tol and j not in used:
#                 bridges.append(LineString([pt, dangling[j]]))
#                 used.add(i); used.add(j)
#         return bridges

#     wall_ls  = [LineString(l) for l in raw_wall_lines]
#     glass_ls = list(glass_union.geoms) if (glass_union and
#                 glass_union.geom_type=='MultiLineString') else \
#                ([glass_union] if glass_union else [])

#     wall_sn  = node_snap(wall_ls,  snap_tol)
#     glass_sn = node_snap(glass_ls, snap_tol)
#     boundary = wall_sn + glass_sn
#     bridges  = bridge_gaps(boundary, bridge_tol)
#     lines_poly = boundary + bridges

#     merged   = unary_union(lines_poly)
#     all_poly = list(polygonize(merged))
#     all_poly.sort(key=lambda p: p.area, reverse=True)

#     # Outer envelope
#     outer_ids = set()
#     for poly in all_poly[:5]:
#         others = [p for p in all_poly if p is not poly and p.area < poly.area*0.85]
#         if len(others) > 1:
#             n = sum(1 for p in others if poly.contains(p.centroid))
#             if n/len(others) >= 0.4:
#                 outer_ids.add(id(poly)); break

#     candidates = []
#     for poly in all_poly:
#         if id(poly) in outer_ids: continue
#         a = poly.area / unit_factor
#         if a < min_area_m2 or a > max_area_m2: continue
#         gf  = edge_glass_fraction(poly, glass_union, snap_tol, glass_proximity_mult)
#         hg  = has_any_glass_edge(poly,  glass_union, snap_tol, glass_proximity_mult)
#         candidates.append((poly, gf, hg))

#     candidates.sort(key=lambda x: x[0].area, reverse=True)
#     accepted = []
#     for (poly, gf, hg) in candidates:
#         parent = next(((ap,ag,ai) for ap,ag,ai in accepted
#                        if poly.intersection(ap).area/poly.area >= 0.9), None)
#         if parent is None:
#             accepted.append((poly, gf, gf >= glass_edge_thresh))
#         elif hg:
#             accepted.append((poly, gf, True))
#     return accepted, bridges

# # ──────────────────────────────────────────────────────
# #  MODE B — Endpoint-bridging polygonize  (reference engine)
# # ──────────────────────────────────────────────────────
# def detect_rooms_mode_b(raw_wall_lines, extracted_objects,
#                          gap_close_tol, max_door_width, min_wall_len,
#                          min_area_m2, max_area_m2, unit_factor,
#                          outer_area_pct=25.0,
#                          exclude_stairs=True,
#                          stair_parallel_min=4,
#                          stair_angle_tol=8.0,
#                          max_stair_area_m2=20.0,
#                          min_solidity=0.50,
#                          max_aspect_ratio=15.0,
#                          max_interior_walls=8,
#                          min_closet_area_m2=0.3):
#     """
#     Reference-based endpoint-bridging polygonize engine with:
#     - Staircase region exclusion (parallel line density check)
#     - Robust outer envelope removal (largest containing polygon)
#     - Nested duplicate removal
#     """
#     if not raw_wall_lines:
#         return [], []

#     # --- Step 1: filter short noise, build Shapely lines ---
#     shapely_walls = [LineString(l) for l in raw_wall_lines
#                      if LineString(l).length >= min_wall_len]
#     if not shapely_walls:
#         return [], []

#     merged_walls = unary_union(shapely_walls)
#     if merged_walls.geom_type == 'LineString':
#         lines_list = [merged_walls]
#     elif merged_walls.geom_type == 'MultiLineString':
#         lines_list = list(merged_walls.geoms)
#     else:
#         lines_list = [g for g in merged_walls.geoms if g.geom_type == 'LineString']

#     # --- Step 2: collect dangling endpoints ---
#     valid_endpoints = []
#     for line in lines_list:
#         if line.length > min_wall_len:
#             valid_endpoints.append(Point(line.coords[0]))
#             valid_endpoints.append(Point(line.coords[-1]))

#     if show_debug:
#         st.info(f"Valid endpoints for bridging: **{len(valid_endpoints)}**")

#     bridges = []

#     # --- Step 3a: snap tiny gaps ---
#     for i, ep1 in enumerate(valid_endpoints):
#         for j, ep2 in enumerate(valid_endpoints):
#             if i < j and ep1.distance(ep2) <= gap_close_tol:
#                 bridges.append(LineString([ep1, ep2]))

#     # --- Step 3b: orthogonal door/archway bridges ---
#     for i, ep1 in enumerate(valid_endpoints):
#         for j, ep2 in enumerate(valid_endpoints):
#             if i < j:
#                 dist = ep1.distance(ep2)
#                 if gap_close_tol < dist <= max_door_width:
#                     dx = abs(ep1.x - ep2.x)
#                     dy = abs(ep1.y - ep2.y)
#                     if dx < 150 or dy < 150:
#                         bridge = LineString([ep1, ep2])
#                         if not bridge.crosses(merged_walls):
#                             bridges.append(bridge)

#     if show_debug:
#         st.info(f"Bridges added: **{len(bridges)}**")

#     # --- Step 4: polygonize ---
#     all_geom  = lines_list + bridges
#     noded     = unary_union(all_geom)
#     raw_polys = list(polygonize(noded))

#     if show_debug:
#         st.info(f"Raw polygons: **{len(raw_polys)}**")

#     if not raw_polys:
#         return [], []

#     # Sort large → small
#     raw_polys.sort(key=lambda p: p.area, reverse=True)

#     # Bounding box for outer-envelope threshold
#     min_x, min_y, max_x, max_y = noded.bounds
#     total_bbox_area = (max_x - min_x) * (max_y - min_y)
#     outer_thresh_mm2 = total_bbox_area * (outer_area_pct / 100.0)

#     min_area_mm2   = min_area_m2       * unit_factor
#     max_area_mm2   = max_area_m2       * unit_factor
#     min_closet_mm2 = min_closet_area_m2 * unit_factor
#     max_stair_mm2  = max_stair_area_m2  * unit_factor

#     # ── STAIRCASE DETECTOR ──
#     # A staircase polygon contains many short parallel lines inside it.
#     # We check wall segments whose midpoint is inside the polygon.
#     def is_staircase(poly, wall_segs, min_parallel, angle_tol):
#         try:
#             angles = []
#             for seg in wall_segs:
#                 mid = Point((seg.coords[0][0]+seg.coords[-1][0])/2,
#                              (seg.coords[0][1]+seg.coords[-1][1])/2)
#                 if poly.contains(mid):
#                     dx = seg.coords[-1][0] - seg.coords[0][0]
#                     dy = seg.coords[-1][1] - seg.coords[0][1]
#                     angles.append(math.degrees(math.atan2(dy, dx)) % 180)
#             if len(angles) < min_parallel:
#                 return False
#             angles.sort()
#             for ref in angles:
#                 count = sum(1 for a in angles
#                             if abs(a - ref) <= angle_tol
#                             or abs(a - ref) >= (180 - angle_tol))
#                 if count >= min_parallel:
#                     return True
#         except Exception:
#             pass
#         return False

#     rooms_data    = []
#     wall_cavities = []

#     for poly in raw_polys:
#         area = poly.area

#         # ── 1. Outer envelope exclusion ──
#         if area >= outer_thresh_mm2:
#             if show_debug:
#                 st.info(f"🏛️ Outer envelope excluded: {area/unit_factor:.1f} m²")
#             continue

#         # ── 2. Area range filter ──
#         if area < min_area_mm2 or area > max_area_mm2:
#             # Small closet-check
#             if min_closet_mm2 <= area < min_area_mm2:
#                 buffered = poly.buffer(50)
#                 for obj in extracted_objects:
#                     if buffered.covers(obj['point']):
#                         rooms_data.append({
#                             "width": round(poly.bounds[2]-poly.bounds[0], 2),
#                             "height": round(poly.bounds[3]-poly.bounds[1], 2),
#                             "area": round(area, 2),
#                             "polygon": poly, "objects_inside": []
#                         })
#                         break
#             elif 10000 < area < min_closet_mm2:
#                 wall_cavities.append(poly)
#             continue

#         # ── 3. Shape validation — reject non-room artifacts ──

#         # 3a. Solidity: area / convex_hull_area
#         # Real rooms are solid (>0.55). Stair artifacts and corridor slivers are hollow/jagged.
#         try:
#             hull  = poly.convex_hull
#             solid = area / hull.area if hull.area > 0 else 0
#         except Exception:
#             solid = 1.0

#         if solid < min_solidity:
#             if show_debug:
#                 st.info(f"⛔ Low solidity ({solid:.2f}) excluded: {area/unit_factor:.1f} m²")
#             continue

#         # 3b. Aspect ratio: bounding box W/H must be reasonable
#         rmx, rmy, rMx, rMy = poly.bounds
#         width  = rMx - rmx
#         height = rMy - rmy
#         ar = max(width, height) / max(min(width, height), 1)
#         if ar > max_aspect_ratio:
#             if show_debug:
#                 st.info(f"⛔ High aspect ratio ({ar:.1f}) excluded: {area/unit_factor:.1f} m²")
#             continue

#         # 3c. Staircase exclusion — dense parallel interior lines
#         if exclude_stairs and area <= max_stair_mm2:
#             if is_staircase(poly, shapely_walls, stair_parallel_min, stair_angle_tol):
#                 if show_debug:
#                     st.info(f"🪜 Staircase excluded: {area/unit_factor:.1f} m²")
#                 continue

#         # 3d. Interior wall piercing — if many wall segments pass THROUGH this polygon
#         # (midpoint inside AND both endpoints inside), it's not a real enclosed room.
#         # Real rooms: wall lines form the BOUNDARY, not cross through.
#         interior_piercing = 0
#         for seg in shapely_walls:
#             try:
#                 s_pt = Point(seg.coords[0])
#                 e_pt = Point(seg.coords[-1])
#                 # Both endpoints inside = line is fully inside = pierces the room
#                 if poly.contains(s_pt) and poly.contains(e_pt):
#                     interior_piercing += 1
#             except Exception:
#                 pass
#         # A real room boundary has walls ON its edges, not inside it
#         # Allow a few (partition walls, columns) but many = artifact
#         if interior_piercing > max_interior_walls:
#             if show_debug:
#                 st.info(f"⛔ Wall-pierced polygon ({interior_piercing} internal segs) excluded: "
#                         f"{area/unit_factor:.1f} m²")
#             continue

#         rooms_data.append({
#             "width": round(width, 2), "height": round(height, 2),
#             "area": round(area, 2), "polygon": poly, "objects_inside": []
#         })

#     # --- Clean: remove nested duplicates ---
#     # A polygon whose centroid is inside another polygon = nested → remove smaller one
#     clean_rooms = []
#     for i, room in enumerate(rooms_data):
#         is_bad = False
#         for j, other in enumerate(rooms_data):
#             if i == j: continue
#             # If this room's representative point is inside another → it's nested
#             try:
#                 if other['polygon'].contains(room['polygon'].representative_point()):
#                     is_bad = True; break
#                 # Also catch heavy overlap (>80% overlap with a larger room)
#                 inter = room['polygon'].intersection(other['polygon'])
#                 if inter.area > 0.8 * room['area'] and room['area'] < other['area']:
#                     is_bad = True; break
#             except Exception:
#                 pass
#         if not is_bad:
#             clean_rooms.append(room)

#     clean_rooms.sort(key=lambda x: x['area'], reverse=True)
#     for idx, room in enumerate(clean_rooms):
#         room['name']    = f"Room {idx+1}"
#         room['room_id'] = f"R{idx+1}"

#     return clean_rooms, wall_cavities

# # ──────────────────────────────────────────────────────
# #  RUN DETECTION
# # ──────────────────────────────────────────────────────
# with st.spinner("🔲 Detecting rooms…"):
#     if USE_GLASS_MODE:
#         accepted, bridges_used = detect_rooms_mode_a(
#             raw_wall_lines, glass_union,
#             snap_tol_a, bridge_tol_a,
#             glass_edge_thresh, glass_proximity_mult,
#             min_area_m2, max_area_m2, unit_factor)
#         wall_cavities = []
#         # Convert to unified format
#         rooms_unified = []
#         for i, (poly, gf, is_glass) in enumerate(accepted):
#             minx,miny,maxx,maxy = poly.bounds
#             rooms_unified.append({
#                 "name": f"Room {i+1}", "room_id": f"R{i+1}",
#                 "polygon": poly, "gf": gf, "is_glass": is_glass,
#                 "width": maxx-minx, "height": maxy-miny,
#                 "area": poly.area, "objects_inside": []
#             })
#     else:
#         room_list, wall_cavities = detect_rooms_mode_b(
#             raw_wall_lines, extracted_objects,
#             gap_close_tol, max_door_width, min_wall_len,
#             min_area_m2, max_area_m2, unit_factor,
#             outer_area_pct=outer_area_pct,
#             exclude_stairs=exclude_stairs,
#             stair_parallel_min=int(stair_parallel_min),
#             stair_angle_tol=float(stair_angle_tol),
#             max_stair_area_m2=float(max_stair_area_m2),
#             min_solidity=float(min_solidity),
#             max_aspect_ratio=float(max_aspect),
#             max_interior_walls=int(max_interior_walls))
#         bridges_used = []
#         rooms_unified = []
#         for r in room_list:
#             rooms_unified.append({
#                 "name": r['name'], "room_id": r['room_id'],
#                 "polygon": r['polygon'], "gf": 0.0, "is_glass": False,
#                 "width": r['width'], "height": r['height'],
#                 "area": r['area'], "objects_inside": r['objects_inside']
#             })

# # Place objects in rooms
# for room in rooms_unified:
#     poly_buf = room['polygon'].buffer(10)
#     for obj in extracted_objects:
#         if poly_buf.covers(obj['point']):
#             room['objects_inside'].append(obj['object_id'])

# n_rooms = len(rooms_unified)
# n_glass = sum(1 for r in rooms_unified if r.get('is_glass'))
# n_wall  = n_rooms - n_glass

# if show_debug:
#     st.success(f"Final rooms: **{n_rooms}**  ({n_wall} wall + {n_glass} glass)")

# if not rooms_unified:
#     st.error(
#         "❌ No rooms found.\n\n"
#         "**Try:**\n"
#         "- ↑ Max door/archway width (currently " + str(max_door_width) + " mm)\n"
#         "- ↓ Min Room Area\n"
#         "- ↓ Min wall segment length\n"
#         "- Check correct layers are selected in Layer debug above"
#     )
#     # Show raw geometry for diagnosis
#     fig0, ax0 = plt.subplots(figsize=(14,7))
#     fig0.patch.set_facecolor("#0f1117"); ax0.set_facecolor("#0f1117")
#     for l in raw_wall_lines[:5000]:
#         ax0.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
#                  color="#00d4ff", lw=0.5, alpha=0.5)
#     ax0.set_aspect("equal")
#     ax0.set_title("Raw wall geometry — no rooms found", color="white")
#     st.pyplot(fig0); plt.close()
#     st.stop()

# st.success(f"✅ **{n_rooms} rooms** detected  ({n_glass} glass + {n_wall} wall-bounded)")

# # ──────────────────────────────────────────────────────
# #  HEAT LOAD
# # ──────────────────────────────────────────────────────
# COLORS = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
#           "#DDA0DD","#98D8C8","#F7DC6F","#82E0AA","#F1948A",
#           "#85C1E9","#F0B27A","#C39BD3","#76D7C4","#F9E79F",
#           "#AED6F1","#A9DFBF","#FAD7A0","#D2B4DE","#FFB3BA"]

# room_data = []
# for i, room in enumerate(rooms_unified):
#     poly       = room['polygon']
#     gf         = room.get('gf', 0.0)
#     is_glass   = room.get('is_glass', False)
#     area_m2    = poly.area   / unit_factor
#     perim_m    = poly.length / unit_div
#     minx,miny,maxx,maxy = poly.bounds
#     length_m   = (maxx - minx) / unit_div
#     breadth_m  = (maxy - miny) / unit_div

#     glass_p_m  = perim_m * gf
#     wall_p_m   = perim_m * (1 - gf)
#     wall_a_m2  = wall_p_m  * H
#     glass_a_m2 = glass_p_m * H

#     q_wall   = wall_a_m2  * U_wall  * DT
#     q_glass  = glass_a_m2 * U_glass * DT
#     q_people = people_per_room * Q_person
#     q_total  = q_wall + q_glass + q_people
#     tr       = q_total / 3517

#     room_data.append({
#         "Room":            room['name'],
#         "Type":            "🔵 Glass" if is_glass else "🟩 Wall",
#         "Area (m²)":       round(area_m2,     3),
#         "Perimeter (m)":   round(perim_m,     3),
#         "Length (m)":      round(length_m,    2),
#         "Breadth (m)":     round(breadth_m,   2),
#         "Glass % edge":    round(gf*100,      1),
#         "Wall Area (m²)":  round(wall_a_m2,   3),
#         "Glass Area (m²)": round(glass_a_m2,  3),
#         "Q_wall (W)":      round(q_wall,      1),
#         "Q_glass (W)":     round(q_glass,     1),
#         "Q_people (W)":    round(q_people,    1),
#         "Q_total (W)":     round(q_total,     1),
#         "TR":              round(tr,          3),
#         "Objects":         ", ".join(room.get('objects_inside', [])) or "—",
#         "_poly":           poly,
#         "_gf":             gf,
#         "_is_glass":       is_glass,
#         "_color":          "#00d4ff" if is_glass else COLORS[i % len(COLORS)],
#     })

# df           = pd.DataFrame(room_data)
# display_cols = [c for c in df.columns if not c.startswith("_")]

# # ──────────────────────────────────────────────────────
# #  FLOOR PLAN
# # ──────────────────────────────────────────────────────
# st.subheader("🗺️ Detected Floor Plan")
# fig, ax = plt.subplots(figsize=(20, 11))
# fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
# ax.tick_params(colors="#888")
# for sp in ax.spines.values(): sp.set_edgecolor("#333")

# if show_raw:
#     for l in raw_wall_lines[:8000]:
#         ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
#                 color="#1e3a50", lw=0.3, alpha=0.4)

# # Draw wall lines
# for l in raw_wall_lines[:8000]:
#     ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
#             color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

# # Draw glass lines
# for l in raw_glass_lines:
#     ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
#             color="#00d4ff", lw=1.4, alpha=0.75, zorder=2)

# # Draw bridges
# for br in bridges_used:
#     xs, ys = br.xy
#     ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.7, zorder=2)

# # Draw wall cavities
# for cav in wall_cavities:
#     if cav.geom_type == 'Polygon':
#         xs, ys = cav.exterior.xy
#         ax.fill(xs, ys, color="dimgray", alpha=0.8, zorder=2)

# # Draw rooms
# for row in room_data:
#     poly   = row["_poly"]
#     color  = row["_color"]
#     ig     = row["_is_glass"]
#     gf     = row["_gf"]
#     xs, ys = poly.exterior.xy
#     ax.fill(xs, ys, alpha=0.38 if ig else 0.22, color=color, zorder=3)
#     ax.plot(xs, ys, color="#00d4ff" if ig else color,
#             lw=2.5 if ig else 1.6, zorder=4)
#     cx, cy = poly.centroid.x, poly.centroid.y
#     label  = f"{'🔵' if ig else '🟩'} {row['Room']}\n{row['Area (m²)']} m²\nTR:{row['TR']}"
#     if gf > 0.05:
#         label += f"\n{row['Glass % edge']}% glass"
#     ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5,
#             color="white", fontfamily="monospace", zorder=5,
#             bbox=dict(boxstyle="round,pad=0.28", facecolor="#000000dd",
#                       edgecolor="#00d4ff" if ig else color,
#                       linewidth=1.3 if ig else 0.5))

# # Draw furniture objects
# for obj in extracted_objects:
#     ax.plot(obj['center_x'], obj['center_y'], 'ro', markersize=4, zorder=6)

# ax.set_aspect("equal")
# ax.set_title(
#     f"Floor Plan — {n_rooms} rooms ({n_wall} wall + {n_glass} glass) | {mode_label}",
#     color="#00d4ff", fontfamily="monospace", fontsize=10)
# ax.grid(True, color="#1a2a3a", lw=0.25)
# legend_items = [
#     mpatches.Patch(color="#3a6a8a", label="Wall lines"),
#     mpatches.Patch(color="#ff4444", label=f"Bridges"),
# ]
# if USE_GLASS_MODE:
#     legend_items.append(mpatches.Patch(color="#00d4ff", label="Glass lines"))
# ax.legend(handles=legend_items, loc="upper right",
#           facecolor="#0f1117", edgecolor="#444", labelcolor="white", fontsize=9)
# st.pyplot(fig); plt.close()

# # ──────────────────────────────────────────────────────
# #  TABLE
# # ──────────────────────────────────────────────────────
# st.subheader("📊 Room Measurements & Heat Load")
# st.dataframe(
#     df[display_cols].style
#       .format({"TR":"{:.3f}","Area (m²)":"{:.3f}",
#                "Perimeter (m)":"{:.3f}","Q_total (W)":"{:.1f}","Glass % edge":"{:.1f}"})
#       .background_gradient(subset=["TR"],         cmap="YlOrRd")
#       .background_gradient(subset=["Area (m²)"],  cmap="Greens"),
#     use_container_width=True)

# # ──────────────────────────────────────────────────────
# #  METRICS
# # ──────────────────────────────────────────────────────
# st.subheader("🌡️ Heat Load Summary")
# total_area = df["Area (m²)"].sum()
# total_kw   = df["Q_total (W)"].sum() / 1000
# total_tr   = df["TR"].sum()
# c1,c2,c3,c4,c5 = st.columns(5)
# c1.metric("🏠 Rooms",     n_rooms)
# c2.metric("🔵 Glass",     n_glass)
# c3.metric("📐 Total Area",f"{total_area:.2f} m²")
# c4.metric("⚡ Total Load",f"{total_kw:.2f} kW")
# c5.metric("❄️ Total TR",  f"{total_tr:.2f} TR")

# st.subheader("📈 TR per Room")
# fig2, ax2 = plt.subplots(figsize=(max(10, n_rooms), 4))
# fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
# ax2.tick_params(colors="#888")
# for sp in ax2.spines.values(): sp.set_edgecolor("#333")
# bars = ax2.bar(df["Room"], df["TR"],
#                color=[r["_color"] for r in room_data],
#                alpha=0.85, edgecolor="#ffffff11")
# for bar, row in zip(bars, room_data):
#     ax2.text(bar.get_x()+bar.get_width()/2,
#              bar.get_height()+total_tr*0.003,
#              f"{row['TR']:.3f}", ha="center", va="bottom", color="white", fontsize=8)
# ax2.set_xlabel("Room", color="#aaa"); ax2.set_ylabel("TR", color="#aaa")
# ax2.set_title("TR per Room", color="#00d4ff", fontfamily="monospace")
# ax2.grid(axis="y", color="#1a2a3a", lw=0.4)
# plt.xticks(rotation=45, ha="right", color="#aaa")
# st.pyplot(fig2); plt.close()

# st.download_button("⬇️ Download CSV",
#     df[display_cols].to_csv(index=False), "rooms_heat_load.csv", "text/csv")
# st.divider()
# st.caption(f"v11 | {mode_label} | {n_rooms} rooms | {total_tr:.2f} TR")































"""
CAD Room Extractor + Heat Load  v12 (UNIFIED)
==============================================
MERGED from v8 (Glass-partition engine) + v11 (Endpoint-bridging engine)

WHAT'S IN THIS FILE:
  ✅ Mode A  — Glass-partition (office/commercial):
       Full v8 geometry extraction (per-segment INSERT traversal, ARC/CIRCLE/SPLINE)
       v8 glass edge detection (edge_glass_fraction + has_any_glass_edge)
       v8 sub-room logic: ANY glass edge = keep as separate glass room
       v8 snap + bridge_gaps + polygonize + deduplication
  ✅ Mode B  — Base-layer only (residential/no-glass):
       v11 endpoint-bridging polygonize engine
       Staircase detection (parallel interior line density)
       Solidity + aspect ratio + interior-wall-piercing shape validation
       Nested duplicate removal
       Furniture object clustering (networkx)
  ✅ Common:
       ODA DWG→DXF conversion
       $INSUNITS scale detection
       Layer auto-discovery + interactive selector
       Heat load (Q_wall + Q_glass + Q_people → TR)
       Floor plan matplotlib visualisation
       CSV export

MODE SELECTION:
  - Auto-detect: glass layers present → Mode A, else Mode B
  - Manual override via sidebar selectbox
"""

import streamlit as st
import subprocess, os, math
import ezdxf
from ezdxf import path as dxf_path
from ezdxf.math import Matrix44
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiLineString, MultiPoint
from shapely.ops import polygonize, unary_union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── auto-install networkx if missing ──────────────────
import importlib, subprocess as _sp
if importlib.util.find_spec("networkx") is None:
    _sp.run(["pip","install","networkx","--break-system-packages","-q"], check=False)
import networkx as nx

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="CAD Room Extractor v12", layout="wide")
st.markdown("""
<style>
  .main{background:#0f1117}
  h1{color:#00d4ff;font-family:'Courier New',monospace}
  .block-container{padding-top:2rem}
  div[data-testid="metric-container"]{
    background:#1a1f2e;border-radius:10px;
    padding:10px;border:1px solid #00d4ff33}
</style>
""", unsafe_allow_html=True)
st.title("🏗️ CAD Room Extractor + Heat Load Calculator")
st.caption("v12 UNIFIED — Mode A (Glass-partition v8) + Mode B (Endpoint-bridging v11) | Nothing missed")

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    oda_path = st.text_input(
        "ODAFileConverter Path",
        r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe")

    st.divider()
    st.markdown("**Layers**")
    wall_input  = st.text_area("Wall / Base layers",
        "MAIN\nMAIN-4\nCEN-1\nA-WALL\nWALLS\nWall\nWalls\nBase\n0")
    glass_input = st.text_area("Glass layers (leave blank = Mode B forced)", "GLASS")
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
    st.markdown("**Heat Load**")
    H               = st.number_input("Room Height (m)",    value=3.0,  step=0.1)
    U_wall          = st.number_input("Wall U-Value",       value=1.8,  step=0.1)
    U_glass         = st.number_input("Glass U-Value",      value=5.8,  step=0.1)
    DT              = st.number_input("ΔT (°C)",            value=10,   step=1)
    people_per_room = st.number_input("People / Room",      value=2,    step=1)
    Q_person        = st.number_input("Heat/Person (W)",    value=75,   step=5)

    st.divider()
    st.markdown("**Room Filtering (both modes)**")
    min_area_m2    = st.number_input("Min Room Area (m²)",  value=2.0,   step=0.5)
    max_area_m2    = st.number_input("Max Room Area (m²)",  value=500.0, step=10.0)
    outer_area_pct = st.number_input(
        "Outer envelope threshold (%)", value=25.0, step=5.0,
        help="Polygons larger than this % of total bounding box area → excluded. Lower = more aggressive.")

    st.divider()
    st.markdown("**Mode A — Glass Detection**")
    glass_edge_thresh    = st.number_input(
        "Glass edge threshold (0–1)", value=0.15, step=0.05,
        help="Standalone rooms: fraction of edges that must be glass to label as glass room. "
             "Sub-rooms: ANY glass edge is enough (v8 fix).")
    glass_proximity_mult = st.number_input(
        "Glass proximity multiplier", value=3.0, step=0.5, min_value=1.0, max_value=10.0,
        help="snap_tol × this = search radius for glass edge detection. "
             "Increase if glass walls are slightly offset from polygon edges.")
    snap_tol_a   = st.number_input("Snap tolerance (Mode A)",   value=10.0, step=1.0)
    bridge_tol_a = st.number_input("Bridge tolerance (Mode A)", value=80.0, step=5.0)

    st.divider()
    st.markdown("**Mode A — Shape Quality**")
    min_compact_a = st.number_input("Min Compactness (Mode A)", value=0.04, step=0.01)
    max_aspect_a  = st.number_input("Max Aspect Ratio (Mode A)", value=10.0, step=0.5)

    st.divider()
    st.markdown("**Mode B — Bridging (gap sealing)**")
    gap_close_tol = st.number_input(
        "Snap/merge tolerance (mm)", value=15.0, step=5.0,
        help="Lines within this distance are merged into one wall cluster.")
    max_door_width = st.number_input(
        "Max door/archway width (mm)", value=1500.0, step=100.0,
        help="Endpoints closer than this and axis-aligned → bridged to seal the opening.")
    min_wall_len = st.number_input(
        "Min wall segment length (mm)", value=200.0, step=50.0,
        help="Segments shorter than this are ignored (noise/hatching).")

    st.divider()
    st.markdown("**Mode B — Shape Validation**")
    min_solidity = st.number_input(
        "Min solidity (0–1)", value=0.50, step=0.05, min_value=0.0, max_value=1.0,
        help="Area ÷ convex hull area. Real rooms ≥ 0.5.")
    max_aspect_b = st.number_input(
        "Max aspect ratio (Mode B)", value=15.0, step=1.0,
        help="Extreme thin strips = border artifacts → excluded.")
    max_interior_walls = st.number_input(
        "Max interior wall segments", value=8, step=1,
        help="Wall segments fully INSIDE a polygon. High count = artifact polygon.")
    exclude_stairs     = st.checkbox("Exclude staircase regions", value=True)
    stair_parallel_min = st.number_input("Min parallel lines to flag as stair", value=4, step=1)
    stair_angle_tol    = st.number_input("Stair angle tolerance (°)", value=8.0, step=1.0)
    max_stair_area_m2  = st.number_input("Max staircase area (m²)", value=20.0, step=1.0)

    st.divider()
    show_debug = st.checkbox("Show debug info",   value=True)
    show_raw   = st.checkbox("Show raw geometry", value=False)

# ══════════════════════════════════════════════════════
#  FILE UPLOAD
# ══════════════════════════════════════════════════════
uploaded_file = st.file_uploader("📂 Upload DWG File", type=["dwg"])
if not uploaded_file:
    st.info("👆 Upload a DWG file to begin.")
    st.stop()

dwg_filename = uploaded_file.name
dwg_path_file = os.path.join(os.getcwd(), dwg_filename)
with open(dwg_path_file, "wb") as f:
    f.write(uploaded_file.getbuffer())
st.success(f"Uploaded: **{dwg_filename}**")

# ══════════════════════════════════════════════════════
#  ODA DWG → DXF CONVERSION
# ══════════════════════════════════════════════════════
dxf_folder = os.path.join(os.getcwd(), "converted_dxf")
os.makedirs(dxf_folder, exist_ok=True)

with st.spinner("🔄 Converting DWG → DXF…"):
    try:
        subprocess.run(
            [oda_path, os.getcwd(), dxf_folder, "ACAD2013", "DXF", "0", "1"],
            check=True, capture_output=True, text=True, timeout=120)
        st.success("✅ ODA conversion OK")
    except subprocess.CalledProcessError as e:
        st.error(f"ODA failed: {e.stderr}"); st.stop()
    except FileNotFoundError:
        st.error("ODAFileConverter.exe not found — check path in sidebar."); st.stop()

dxf_name     = dwg_filename.rsplit(".", 1)[0] + ".dxf"
dxf_path_conv = os.path.join(dxf_folder, dxf_name)
if not os.path.exists(dxf_path_conv):
    hits = [f for f in os.listdir(dxf_folder) if f.endswith(".dxf")]
    if not hits:
        st.error("No DXF found after conversion."); st.stop()
    dxf_path_conv = os.path.join(dxf_folder, hits[0])

# ══════════════════════════════════════════════════════
#  LOAD DXF + LAYER DISCOVERY
# ══════════════════════════════════════════════════════
try:
    doc = ezdxf.readfile(dxf_path_conv)
except Exception as e:
    st.error(f"DXF load error: {e}"); st.stop()

msp = doc.modelspace()
all_layers_in_file = {layer.dxf.name for layer in doc.layers}

layer_entity_count = {}
for ent in msp:
    l = ent.dxf.get("layer", "0")
    layer_entity_count[l] = layer_entity_count.get(l, 0) + 1

sorted_layers = sorted(all_layers_in_file,
    key=lambda l: layer_entity_count.get(l, 0), reverse=True)
active_layers = [l for l in sorted_layers if layer_entity_count.get(l, 0) > 0]

# Parse sidebar layer inputs
WALL_LAYERS  = {l.strip().upper() for l in wall_input.strip().splitlines()  if l.strip()}
GLASS_LAYERS = {l.strip().upper() for l in glass_input.strip().splitlines() if l.strip()}
FURN_LAYERS  = {l.strip().upper() for l in furn_input.strip().splitlines()  if l.strip()}

matched_wall  = WALL_LAYERS  & {x.upper() for x in all_layers_in_file}
matched_glass = GLASS_LAYERS & {x.upper() for x in all_layers_in_file}
matched_furn  = FURN_LAYERS  & {x.upper() for x in all_layers_in_file}

matched_wall_ents = sum(layer_entity_count.get(l, 0)
    for l in all_layers_in_file if l.upper() in matched_wall)
total_ents = sum(layer_entity_count.values()) or 1
poor_match = matched_wall_ents < total_ents * 0.20

NON_WALL_KW = {"furniture","plant","planter","text","vp","defpoint",
               "dimension","dim","hatch","annotation","title","border",
               "viewport","pplne","electrical","elec","plumbing","mech","door"}

# ── Interactive layer selector ──────────────────────
with st.expander("🔍 Layer debug & selector", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Layers in DXF (by entity count):**")
        for l in sorted_layers:
            cnt = layer_entity_count.get(l, 0)
            tag = ("🟩 WALL"  if l.upper() in WALL_LAYERS
              else ("🟦 GLASS" if l.upper() in GLASS_LAYERS
              else ("🪑 FURN"  if l.upper() in FURN_LAYERS else "⚪")))
            st.markdown(f"{tag} `{l}` — {cnt} entities")
    with col2:
        if matched_wall and not poor_match:
            st.success(f"Wall layers matched: {sorted(matched_wall)} ✅")
        else:
            st.warning("⚠️ Poor or no wall layer match — select below")

        smart_def = [l for l in active_layers
                     if not any(kw in l.lower() for kw in NON_WALL_KW)]
        if not smart_def and active_layers:
            smart_def = [active_layers[0]]

        sel_wall = st.multiselect("Wall / Base layers", options=active_layers,
            default=smart_def if poor_match else
                    [l for l in active_layers if l.upper() in matched_wall],
            key="sel_wall")
        sel_glass = st.multiselect("Glass layers (optional)", options=active_layers,
            default=[l for l in active_layers if l.upper() in matched_glass],
            key="sel_glass")
        sel_furn = st.multiselect("Furniture/Door layers (optional)", options=active_layers,
            default=[l for l in active_layers if l.upper() in matched_furn],
            key="sel_furn")

        if sel_wall:
            WALL_LAYERS  = {l.upper() for l in sel_wall}
            GLASS_LAYERS = {l.upper() for l in sel_glass}
            FURN_LAYERS  = {l.upper() for l in sel_furn}
            matched_wall  = WALL_LAYERS
            matched_glass = GLASS_LAYERS
            matched_furn  = FURN_LAYERS
            st.success(f"✅ Using wall layers: {sorted(WALL_LAYERS)}")
        elif active_layers:
            WALL_LAYERS  = {l.upper() for l in active_layers}
            GLASS_LAYERS = set()
            FURN_LAYERS  = set()
            matched_wall = WALL_LAYERS

ALLOWED_LAYERS = WALL_LAYERS | GLASS_LAYERS | FURN_LAYERS

# ══════════════════════════════════════════════════════
#  MODE SELECTION
# ══════════════════════════════════════════════════════
glass_found = bool(matched_glass)
if mode_override == "Force Glass-partition (Mode A)":
    USE_GLASS_MODE = True
elif mode_override == "Force Base-layer only (Mode B)":
    USE_GLASS_MODE = False
else:
    USE_GLASS_MODE = glass_found and bool(GLASS_LAYERS)

mode_label = ("🔵 Mode A: Glass-partition (v8 engine)"
              if USE_GLASS_MODE else
              "🟩 Mode B: Base-layer endpoint-bridging (v11 engine)")
st.info(f"**Detection Mode:** {mode_label}")

# ══════════════════════════════════════════════════════
#  SCALE DETECTION  ($INSUNITS)
# ══════════════════════════════════════════════════════
insunits  = doc.header.get("$INSUNITS", 0)
scale_map = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}
unit_names = {0:"Unitless",1:"Inches",2:"Feet",4:"mm",5:"cm",6:"m"}
DRAWING_SCALE = scale_map.get(insunits, 1.0)
st.info(f"📏 $INSUNITS={insunits} ({unit_names.get(insunits,'?')}) → scale×{DRAWING_SCALE}")

unit_factor = 1_000_000.0   # mm² → m²
unit_div    = 1_000.0       # mm  → m

# ══════════════════════════════════════════════════════
#  GEOMETRY EXTRACTION
#  Mode A uses full v8 per-segment INSERT traversal (preserves arc/spline/circle).
#  Mode B uses v11 ezdxf path flattening (simpler, handles all entity types).
#  Both paths produce raw_wall_lines, raw_glass_lines, raw_furn_lines.
# ══════════════════════════════════════════════════════

# ── Shared helper ─────────────────────────────────────
def apply_m44(mat, x, y):
    from ezdxf.math import Vec3
    v = mat.transform(Vec3(x, y, 0))
    return (v.x, v.y)

def build_matrix(ent):
    try:
        return ent.matrix44()
    except Exception:
        ix  = ent.dxf.insert.x; iy = ent.dxf.insert.y
        rot = math.radians(ent.dxf.get("rotation", 0))
        sx  = ent.dxf.get("xscale", 1); sy = ent.dxf.get("yscale", 1)
        c, s = math.cos(rot), math.sin(rot)
        return Matrix44([[sx*c,-sy*s,0,ix],[sx*s,sy*c,0,iy],[0,0,1,0],[0,0,0,1]])

# ── Mode A extraction (v8 full per-segment) ───────────
GEOM_TYPES_V8 = {"LINE","LWPOLYLINE","POLYLINE","ARC","CIRCLE","ELLIPSE","SPLINE"}

def ent_to_segments_v8(entity, mat, layer):
    """v8: returns list of (LineString, is_glass, is_arc). Breaks everything into segments."""
    t        = entity.dxftype()
    is_glass = layer.upper() in GLASS_LAYERS
    res      = []
    try:
        if t == "LINE":
            p1 = apply_m44(mat, entity.dxf.start.x, entity.dxf.start.y)
            p2 = apply_m44(mat, entity.dxf.end.x,   entity.dxf.end.y)
            if p1 != p2:
                res.append((LineString([p1, p2]), is_glass, False))

        elif t == "LWPOLYLINE":
            pts = [apply_m44(mat, p[0], p[1]) for p in entity.get_points()]
            closed = entity.closed or (len(pts) > 2 and pts[0] == pts[-1])
            if closed and pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))

        elif t == "POLYLINE":
            pts = [apply_m44(mat, v.dxf.location.x, v.dxf.location.y)
                   for v in entity.vertices if hasattr(v.dxf, "location")]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))

        elif t == "ARC":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r      = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea <= sa: ea += 2 * math.pi
            angles = np.linspace(sa, ea, 32)
            pts = [apply_m44(mat, cx + r*math.cos(a), cy + r*math.sin(a)) for a in angles]
            if len(pts) >= 2:
                for a, b in zip(pts[:-1], pts[1:]):
                    if a != b:
                        res.append((LineString([a, b]), is_glass, True))
                res.append((LineString([pts[0], pts[-1]]), False, True))

        elif t == "CIRCLE":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r      = entity.dxf.radius
            angles = np.linspace(0, 2*math.pi, 64)
            pts = [apply_m44(mat, cx + r*math.cos(a), cy + r*math.sin(a)) for a in angles]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))

        elif t == "SPLINE":
            try:    raw_pts = [(p[0],p[1]) for p in entity.control_points]
            except: raw_pts = []
            if len(raw_pts) < 2:
                try:    raw_pts = [(p[0],p[1]) for p in entity.fit_points]
                except: raw_pts = []
            pts = [apply_m44(mat, x, y) for x, y in raw_pts]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))
    except Exception:
        pass
    return res

def extract_all_v8(layout, doc, allowed_up, parent_mat=None, parent_layer="0", depth=0):
    """v8: recursive INSERT traversal."""
    if depth > 30: return []
    if parent_mat is None: parent_mat = Matrix44()
    out = []
    for ent in layout:
        et = ent.dxftype()
        if et == "INSERT":
            bname = ent.dxf.name
            if bname not in doc.blocks: continue
            ins_layer = ent.dxf.get("layer", parent_layer)
            combined  = parent_mat @ build_matrix(ent)
            out.extend(extract_all_v8(doc.blocks[bname], doc, allowed_up,
                                      combined, ins_layer, depth+1))
        elif et in GEOM_TYPES_V8:
            eff = ent.dxf.get("layer", parent_layer)
            eff = parent_layer if eff == "0" else eff
            if eff.upper() not in allowed_up: continue
            out.extend(ent_to_segments_v8(ent, parent_mat, eff.upper()))
    return out

# ── Mode B extraction (v11 path flattening) ───────────
def process_entity_v11(entity, raw_lines, scale, exclude_arcs=False):
    """v11: flatten any DXF entity to line segments via ezdxf path engine."""
    if entity.dxftype() == "INSERT":
        try:
            for sub in entity.virtual_entities():
                process_entity_v11(sub, raw_lines, scale, exclude_arcs)
        except Exception:
            pass
        return
    if exclude_arcs and entity.dxftype() == "ARC":
        return
    try:
        p   = dxf_path.make_path(entity)
        pts = list(p.flattening(distance=0.1))
        for i in range(len(pts) - 1):
            s = (round(pts[i].x   * scale, 1), round(pts[i].y   * scale, 1))
            e = (round(pts[i+1].x * scale, 1), round(pts[i+1].y * scale, 1))
            if s != e:
                raw_lines.append([s, e])
    except Exception:
        pass

# ── Run extraction ────────────────────────────────────
with st.spinner("🔍 Extracting geometry…"):
    raw_wall_lines  = []
    raw_glass_lines = []
    raw_furn_lines  = []

    if USE_GLASS_MODE:
        # Mode A: full v8 extraction
        raw_v8 = extract_all_v8(msp, doc, ALLOWED_LAYERS)
        for (ls, is_glass, is_arc) in raw_v8:
            seg = list(ls.coords)
            if is_glass:
                raw_glass_lines.append(seg)
            else:
                # furn check: if layer is in FURN_LAYERS we can't tell from v8 easily
                # → put everything non-glass into wall for geometry; furniture handled separately
                raw_wall_lines.append(seg)
        # Furniture via v11 path (simpler, no glass distinction needed)
        for ent in msp.query("LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE"):
            if ent.dxf.get("layer","0").upper() in FURN_LAYERS:
                process_entity_v11(ent, raw_furn_lines, DRAWING_SCALE, False)
    else:
        # Mode B: v11 path flattening
        for ent in msp.query("LINE LWPOLYLINE POLYLINE INSERT ARC CIRCLE ELLIPSE SPLINE"):
            layer_up = ent.dxf.get("layer","0").upper()
            if layer_up in WALL_LAYERS:
                process_entity_v11(ent, raw_wall_lines,  DRAWING_SCALE, exclude_arcs=True)
            elif layer_up in GLASS_LAYERS:
                process_entity_v11(ent, raw_glass_lines, DRAWING_SCALE, False)
            elif layer_up in FURN_LAYERS:
                process_entity_v11(ent, raw_furn_lines,  DRAWING_SCALE, False)

if not raw_wall_lines:
    st.error(f"❌ No geometry on wall layers {sorted(WALL_LAYERS)}. "
             f"File has: {sorted(all_layers_in_file)}")
    st.stop()

st.success(f"✅ wall:{len(raw_wall_lines)}  glass:{len(raw_glass_lines)}  furn:{len(raw_furn_lines)}")

# ══════════════════════════════════════════════════════
#  FURNITURE OBJECT CLUSTERING  (v11 networkx approach)
# ══════════════════════════════════════════════════════
def process_furniture_to_objects(furn_lines, gap_tol=50):
    if not furn_lines: return []
    from shapely.strtree import STRtree
    shapely_lines = [LineString(l) for l in furn_lines]
    buffered      = [l.buffer(gap_tol) for l in shapely_lines]
    tree = STRtree(buffered)
    G = nx.Graph(); G.add_nodes_from(range(len(furn_lines)))
    for i, poly in enumerate(buffered):
        for j in tree.query(poly):
            if i != j and poly.intersects(buffered[j]):
                G.add_edge(i, j)
    objects_data = []
    for comp in nx.connected_components(G):
        lines = [furn_lines[idx] for idx in comp]
        mls   = MultiLineString([LineString(l) for l in lines])
        minx, miny, maxx, maxy = mls.bounds
        objects_data.append({
            "object_id": f"Obj {len(objects_data)+1}",
            "length":    round(maxx - minx, 2),
            "width":     round(maxy - miny, 2),
            "center_x":  round(mls.centroid.x, 2),
            "center_y":  round(mls.centroid.y, 2),
            "point":     mls.centroid,
        })
    return objects_data

with st.spinner("🪑 Processing furniture/door objects…"):
    extracted_objects = process_furniture_to_objects(raw_furn_lines)
if show_debug:
    st.info(f"Furniture objects detected: **{len(extracted_objects)}**")

# ══════════════════════════════════════════════════════
#  GLASS HELPERS  (v8 — used in Mode A)
# ══════════════════════════════════════════════════════
glass_segs_shapely = [LineString(l) for l in raw_glass_lines]
glass_union        = unary_union(glass_segs_shapely) if glass_segs_shapely else None

def edge_glass_fraction(poly, glass_u, tol, mult):
    """
    v8: fraction of polygon perimeter near a glass line.
    Uses mid + quarter-points on each edge.
    mult = glass_proximity_mult (configurable; v8 default 3.0, was 1.5 in v7).
    """
    if glass_u is None or glass_u.is_empty: return 0.0
    coords    = list(poly.exterior.coords)
    total_len = 0.0; glass_len = 0.0
    cd = tol * mult
    for i in range(len(coords) - 1):
        p1, p2  = coords[i], coords[i+1]
        seg_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        mid  = ((p1[0]+p2[0])/2,     (p1[1]+p2[1])/2)
        qtr1 = ((p1[0]*3+p2[0])/4,   (p1[1]*3+p2[1])/4)
        qtr2 = ((p1[0]+p2[0]*3)/4,   (p1[1]+p2[1]*3)/4)
        total_len += seg_len
        if any(glass_u.distance(Point(p)) <= cd for p in (mid, qtr1, qtr2)):
            glass_len += seg_len
    return 0.0 if total_len == 0 else min(glass_len / total_len, 1.0)

def has_any_glass_edge(poly, glass_u, tol, mult):
    """
    v8: binary — does the polygon have AT LEAST ONE edge near a glass line?
    Used for sub-room detection: any glass contact → keep as separate glass room.
    """
    if glass_u is None or glass_u.is_empty: return False
    coords = list(poly.exterior.coords)
    cd = tol * mult
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        mid  = ((p1[0]+p2[0])/2,     (p1[1]+p2[1])/2)
        qtr1 = ((p1[0]*3+p2[0])/4,   (p1[1]*3+p2[1])/4)
        qtr2 = ((p1[0]+p2[0]*3)/4,   (p1[1]+p2[1]*3)/4)
        if any(glass_u.distance(Point(p)) <= cd for p in (mid, qtr1, qtr2)):
            return True
    return False

# ══════════════════════════════════════════════════════
#  SHAPE QUALITY HELPERS  (v8 — used in Mode A)
# ══════════════════════════════════════════════════════
def compactness(poly):
    if poly.length == 0: return 0
    return (4 * math.pi * poly.area) / (poly.length ** 2)

def aspect_ratio(poly):
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx; h = maxy - miny
    if min(w, h) == 0: return 999
    return max(w, h) / min(w, h)

def is_outer_envelope_v8(poly, all_p, threshold=0.35):
    others = [p for p in all_p if p is not poly]
    if not others: return False
    n = sum(1 for p in others if poly.contains(p.centroid))
    return (n / len(others)) >= threshold

# ══════════════════════════════════════════════════════
#  MODE A — Glass-partition detection  (v8 engine)
# ══════════════════════════════════════════════════════
def detect_rooms_mode_a(raw_wall_lines, glass_union,
                         snap_tol, bridge_tol,
                         glass_edge_thresh, glass_proximity_mult,
                         min_area_m2, max_area_m2, unit_factor,
                         min_compact, max_aspect):

    def node_snap(segs, tol):
        out = []
        for ls in segs:
            try:
                coords = [(round(x/tol)*tol, round(y/tol)*tol) for x, y in ls.coords]
                dedup  = [coords[0]]
                for c in coords[1:]:
                    if c != dedup[-1]: dedup.append(c)
                if len(dedup) >= 2:
                    out.append(LineString(dedup))
            except Exception: pass
        return out

    def bridge_gaps(lines, tol):
        from collections import defaultdict
        ep = defaultdict(int)
        for ls in lines:
            cs = list(ls.coords)
            ep[cs[0]] += 1; ep[cs[-1]] += 1
        dangling = [pt for pt, cnt in ep.items() if cnt == 1]
        if not dangling: return []
        bridges, used = [], set()
        arr = np.array(dangling)
        for i, pt in enumerate(dangling):
            if i in used: continue
            diffs = arr - np.array(pt)
            dists = np.hypot(diffs[:,0], diffs[:,1])
            dists[i] = np.inf
            j = int(np.argmin(dists))
            if dists[j] <= tol and j not in used:
                bridges.append(LineString([pt, dangling[j]]))
                used.add(i); used.add(j)
        return bridges

    # Build Shapely lines from raw
    wall_ls  = [LineString(l) for l in raw_wall_lines]
    glass_ls = (list(glass_union.geoms)
                if glass_union and glass_union.geom_type == "MultiLineString"
                else ([glass_union] if glass_union else []))

    wall_sn   = node_snap(wall_ls,  snap_tol)
    glass_sn  = node_snap(glass_ls, snap_tol)
    boundary  = wall_sn + glass_sn
    bridges   = bridge_gaps(boundary, bridge_tol)
    lines_poly = boundary + bridges

    if show_debug:
        st.info(f"[Mode A] Snapped lines: {len(boundary)}  Bridges: {len(bridges)}")

    merged   = unary_union(lines_poly)
    all_poly = list(polygonize(merged))
    all_poly.sort(key=lambda p: p.area, reverse=True)

    if show_debug:
        st.info(f"[Mode A] Raw polygons: {len(all_poly)}")

    # Identify outer envelope
    outer_ids = set()
    for poly in all_poly[:5]:
        if is_outer_envelope_v8(poly, all_poly):
            outer_ids.add(id(poly))
            if show_debug:
                st.info(f"🏛️ [Mode A] Outer envelope ({poly.area/unit_factor:.1f} m²) → excluded")
            break

    # Pass 1: filter by area + shape
    candidates = []
    for poly in all_poly:
        if id(poly) in outer_ids: continue
        a = poly.area / unit_factor
        if a < min_area_m2 or a > max_area_m2: continue
        if compactness(poly) < min_compact:    continue
        if aspect_ratio(poly) > max_aspect:    continue
        gf  = edge_glass_fraction(poly, glass_union, snap_tol, glass_proximity_mult)
        hg  = has_any_glass_edge(poly,  glass_union, snap_tol, glass_proximity_mult)
        candidates.append((poly, gf, hg))

    if show_debug:
        st.info(f"[Mode A] Candidates after shape filter: {len(candidates)}")

    candidates.sort(key=lambda x: x[0].area, reverse=True)

    def is_mostly_inside(small, large, tol=0.90):
        try:   return small.intersection(large).area / small.area >= tol
        except: return False

    accepted = []
    for (poly, gf, hg) in candidates:
        parent = next(
            ((ap, ag, ai) for ap, ag, ai in accepted if is_mostly_inside(poly, ap)),
            None)
        if parent is None:
            # Standalone → always a room; label as glass if fraction meets threshold
            accepted.append((poly, gf, gf >= glass_edge_thresh))
        else:
            # Sub-room: v8 fix — keep if ANY glass edge (binary), not fraction threshold
            if hg:
                accepted.append((poly, gf, True))
            # else: no glass at all → furniture/column/artifact → skip

    # Deduplicate near-identical polygons
    keep  = list(accepted)
    flags = [False] * len(keep)
    for i in range(len(keep)):
        if flags[i]: continue
        for j in range(i+1, len(keep)):
            if flags[j]: continue
            a, b = keep[i][0], keep[j][0]
            if abs(a.area - b.area) / max(a.area, b.area) > 0.06: continue
            try:
                inter = a.intersection(b).area
                if inter / min(a.area, b.area) >= 0.90:
                    flags[j] = True
            except Exception: pass
    accepted = [r for r, f in zip(keep, flags) if not f]

    # Sort top-left → bottom-right
    accepted.sort(key=lambda r: (-r[0].centroid.y, r[0].centroid.x))
    return accepted, bridges

# ══════════════════════════════════════════════════════
#  MODE B — Endpoint-bridging polygonize  (v11 engine)
# ══════════════════════════════════════════════════════
def detect_rooms_mode_b(raw_wall_lines, extracted_objects,
                         gap_close_tol, max_door_width, min_wall_len,
                         min_area_m2, max_area_m2, unit_factor,
                         outer_area_pct=25.0,
                         exclude_stairs=True,
                         stair_parallel_min=4,
                         stair_angle_tol=8.0,
                         max_stair_area_m2=20.0,
                         min_solidity=0.50,
                         max_aspect_ratio=15.0,
                         max_interior_walls=8,
                         min_closet_area_m2=0.3):

    if not raw_wall_lines:
        return [], []

    # Step 1: filter short noise
    shapely_walls = [LineString(l) for l in raw_wall_lines
                     if LineString(l).length >= min_wall_len]
    if not shapely_walls:
        return [], []

    merged_walls = unary_union(shapely_walls)
    if merged_walls.geom_type == "LineString":
        lines_list = [merged_walls]
    elif merged_walls.geom_type == "MultiLineString":
        lines_list = list(merged_walls.geoms)
    else:
        lines_list = [g for g in merged_walls.geoms if g.geom_type == "LineString"]

    # Step 2: collect dangling endpoints
    valid_endpoints = []
    for line in lines_list:
        if line.length > min_wall_len:
            valid_endpoints.append(Point(line.coords[0]))
            valid_endpoints.append(Point(line.coords[-1]))

    if show_debug:
        st.info(f"[Mode B] Valid endpoints: {len(valid_endpoints)}")

    bridges = []

    # Step 3a: snap tiny gaps
    for i, ep1 in enumerate(valid_endpoints):
        for j, ep2 in enumerate(valid_endpoints):
            if i < j and ep1.distance(ep2) <= gap_close_tol:
                bridges.append(LineString([ep1, ep2]))

    # Step 3b: orthogonal door/archway bridges
    for i, ep1 in enumerate(valid_endpoints):
        for j, ep2 in enumerate(valid_endpoints):
            if i < j:
                dist = ep1.distance(ep2)
                if gap_close_tol < dist <= max_door_width:
                    dx = abs(ep1.x - ep2.x)
                    dy = abs(ep1.y - ep2.y)
                    if dx < 150 or dy < 150:
                        bridge = LineString([ep1, ep2])
                        if not bridge.crosses(merged_walls):
                            bridges.append(bridge)

    if show_debug:
        st.info(f"[Mode B] Bridges added: {len(bridges)}")

    # Step 4: polygonize
    noded     = unary_union(lines_list + bridges)
    raw_polys = list(polygonize(noded))
    raw_polys.sort(key=lambda p: p.area, reverse=True)

    if show_debug:
        st.info(f"[Mode B] Raw polygons: {len(raw_polys)}")

    if not raw_polys:
        return [], []

    # Bounding box for outer-envelope threshold
    minx_n, miny_n, maxx_n, maxy_n = noded.bounds
    total_bbox_area  = (maxx_n - minx_n) * (maxy_n - miny_n)
    outer_thresh_mm2 = total_bbox_area * (outer_area_pct / 100.0)
    min_area_mm2     = min_area_m2        * unit_factor
    max_area_mm2     = max_area_m2        * unit_factor
    min_closet_mm2   = min_closet_area_m2 * unit_factor
    max_stair_mm2    = max_stair_area_m2  * unit_factor

    # Staircase detector
    def is_staircase(poly, wall_segs, min_parallel, angle_tol):
        try:
            angles = []
            for seg in wall_segs:
                mid = Point((seg.coords[0][0]+seg.coords[-1][0])/2,
                             (seg.coords[0][1]+seg.coords[-1][1])/2)
                if poly.contains(mid):
                    dx = seg.coords[-1][0] - seg.coords[0][0]
                    dy = seg.coords[-1][1] - seg.coords[0][1]
                    angles.append(math.degrees(math.atan2(dy, dx)) % 180)
            if len(angles) < min_parallel:
                return False
            angles.sort()
            for ref in angles:
                count = sum(1 for a in angles
                            if abs(a - ref) <= angle_tol
                            or abs(a - ref) >= (180 - angle_tol))
                if count >= min_parallel:
                    return True
        except Exception:
            pass
        return False

    rooms_data    = []
    wall_cavities = []

    for poly in raw_polys:
        area = poly.area

        # 1. Outer envelope
        if area >= outer_thresh_mm2:
            if show_debug:
                st.info(f"🏛️ [Mode B] Outer envelope excluded: {area/unit_factor:.1f} m²")
            continue

        # 2. Area range
        if area < min_area_mm2 or area > max_area_mm2:
            if min_closet_mm2 <= area < min_area_mm2:
                buffered = poly.buffer(50)
                for obj in extracted_objects:
                    if buffered.covers(obj["point"]):
                        rooms_data.append({
                            "width": round(poly.bounds[2]-poly.bounds[0], 2),
                            "height": round(poly.bounds[3]-poly.bounds[1], 2),
                            "area": round(area, 2),
                            "polygon": poly, "objects_inside": []
                        })
                        break
            elif 10000 < area < min_closet_mm2:
                wall_cavities.append(poly)
            continue

        # 3a. Solidity
        try:
            hull  = poly.convex_hull
            solid = area / hull.area if hull.area > 0 else 0
        except Exception:
            solid = 1.0
        if solid < min_solidity:
            if show_debug:
                st.info(f"⛔ [Mode B] Low solidity ({solid:.2f}): {area/unit_factor:.1f} m²")
            continue

        # 3b. Aspect ratio
        rmx, rmy, rMx, rMy = poly.bounds
        width = rMx - rmx; height = rMy - rmy
        ar = max(width, height) / max(min(width, height), 1)
        if ar > max_aspect_ratio:
            if show_debug:
                st.info(f"⛔ [Mode B] High aspect ({ar:.1f}): {area/unit_factor:.1f} m²")
            continue

        # 3c. Staircase
        if exclude_stairs and area <= max_stair_mm2:
            if is_staircase(poly, shapely_walls, stair_parallel_min, stair_angle_tol):
                if show_debug:
                    st.info(f"🪜 [Mode B] Staircase excluded: {area/unit_factor:.1f} m²")
                continue

        # 3d. Interior wall piercing
        interior_piercing = 0
        for seg in shapely_walls:
            try:
                if poly.contains(Point(seg.coords[0])) and poly.contains(Point(seg.coords[-1])):
                    interior_piercing += 1
            except Exception:
                pass
        if interior_piercing > max_interior_walls:
            if show_debug:
                st.info(f"⛔ [Mode B] Wall-pierced ({interior_piercing} segs): {area/unit_factor:.1f} m²")
            continue

        rooms_data.append({
            "width": round(width, 2), "height": round(height, 2),
            "area": round(area, 2), "polygon": poly, "objects_inside": []
        })

    # Remove nested duplicates
    clean_rooms = []
    for i, room in enumerate(rooms_data):
        is_bad = False
        for j, other in enumerate(rooms_data):
            if i == j: continue
            try:
                if other["polygon"].contains(room["polygon"].representative_point()):
                    is_bad = True; break
                inter = room["polygon"].intersection(other["polygon"])
                if inter.area > 0.8 * room["area"] and room["area"] < other["area"]:
                    is_bad = True; break
            except Exception:
                pass
        if not is_bad:
            clean_rooms.append(room)

    clean_rooms.sort(key=lambda x: x["area"], reverse=True)
    for idx, room in enumerate(clean_rooms):
        room["name"]    = f"Room {idx+1}"
        room["room_id"] = f"R{idx+1}"

    return clean_rooms, wall_cavities

# ══════════════════════════════════════════════════════
#  RUN DETECTION
# ══════════════════════════════════════════════════════
with st.spinner("🔲 Detecting rooms…"):
    if USE_GLASS_MODE:
        accepted, bridges_used = detect_rooms_mode_a(
            raw_wall_lines, glass_union,
            snap_tol_a, bridge_tol_a,
            glass_edge_thresh, glass_proximity_mult,
            min_area_m2, max_area_m2, unit_factor,
            min_compact_a, max_aspect_a)
        wall_cavities = []

        # Convert Mode A output to unified room format
        rooms_unified = []
        for i, (poly, gf, is_glass) in enumerate(accepted):
            minx, miny, maxx, maxy = poly.bounds
            rooms_unified.append({
                "name":         f"Room {i+1}",
                "room_id":      f"R{i+1}",
                "polygon":      poly,
                "gf":           gf,
                "is_glass":     is_glass,
                "width":        maxx - minx,
                "height":       maxy - miny,
                "area":         poly.area,
                "objects_inside": [],
            })
    else:
        room_list, wall_cavities = detect_rooms_mode_b(
            raw_wall_lines, extracted_objects,
            gap_close_tol, max_door_width, min_wall_len,
            min_area_m2, max_area_m2, unit_factor,
            outer_area_pct       = outer_area_pct,
            exclude_stairs       = exclude_stairs,
            stair_parallel_min   = int(stair_parallel_min),
            stair_angle_tol      = float(stair_angle_tol),
            max_stair_area_m2    = float(max_stair_area_m2),
            min_solidity         = float(min_solidity),
            max_aspect_ratio     = float(max_aspect_b),
            max_interior_walls   = int(max_interior_walls))
        bridges_used = []

        rooms_unified = []
        for r in room_list:
            rooms_unified.append({
                "name":         r["name"],
                "room_id":      r["room_id"],
                "polygon":      r["polygon"],
                "gf":           0.0,
                "is_glass":     False,
                "width":        r["width"],
                "height":       r["height"],
                "area":         r["area"],
                "objects_inside": r["objects_inside"],
            })

# Place furniture objects inside rooms
for room in rooms_unified:
    poly_buf = room["polygon"].buffer(10)
    for obj in extracted_objects:
        if poly_buf.covers(obj["point"]):
            room["objects_inside"].append(obj["object_id"])

n_rooms = len(rooms_unified)
n_glass = sum(1 for r in rooms_unified if r.get("is_glass"))
n_wall  = n_rooms - n_glass

if show_debug:
    st.success(f"Final rooms: **{n_rooms}**  ({n_wall} wall + {n_glass} glass)")

if not rooms_unified:
    st.error(
        "❌ No rooms found.\n\n"
        f"**Try:**\n"
        f"- ↑ Max door/archway width (currently {max_door_width} mm)\n"
        f"- ↓ Min Room Area\n"
        f"- ↓ Min wall segment length\n"
        f"- Check correct layers are selected above"
    )
    fig0, ax0 = plt.subplots(figsize=(14, 7))
    fig0.patch.set_facecolor("#0f1117"); ax0.set_facecolor("#0f1117")
    for l in raw_wall_lines[:5000]:
        ax0.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
                 color="#00d4ff", lw=0.5, alpha=0.5)
    ax0.set_aspect("equal")
    ax0.set_title("Raw wall geometry — no rooms found", color="white")
    st.pyplot(fig0); plt.close()
    st.stop()

st.success(f"✅ **{n_rooms} rooms** detected  ({n_glass} glass + {n_wall} wall-bounded)")

# ══════════════════════════════════════════════════════
#  HEAT LOAD CALCULATION
# ══════════════════════════════════════════════════════
COLORS = [
    "#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
    "#DDA0DD","#98D8C8","#F7DC6F","#82E0AA","#F1948A",
    "#85C1E9","#F0B27A","#C39BD3","#76D7C4","#F9E79F",
    "#AED6F1","#A9DFBF","#FAD7A0","#D2B4DE","#FFB3BA",
]

room_data = []
for i, room in enumerate(rooms_unified):
    poly      = room["polygon"]
    gf        = room.get("gf", 0.0)
    is_glass  = room.get("is_glass", False)
    area_m2   = poly.area   / unit_factor
    perim_m   = poly.length / unit_div
    minx, miny, maxx, maxy = poly.bounds
    length_m  = (maxx - minx) / unit_div
    breadth_m = (maxy - miny) / unit_div

    glass_p_m  = perim_m * gf
    wall_p_m   = perim_m * (1 - gf)
    wall_a_m2  = wall_p_m  * H
    glass_a_m2 = glass_p_m * H

    q_wall   = wall_a_m2  * U_wall  * DT
    q_glass  = glass_a_m2 * U_glass * DT
    q_people = people_per_room * Q_person
    q_total  = q_wall + q_glass + q_people
    tr       = q_total / 3517

    room_data.append({
        "Room":            room["name"],
        "Type":            "🔵 Glass" if is_glass else "🟩 Wall",
        "Area (m²)":       round(area_m2,    3),
        "Perimeter (m)":   round(perim_m,    3),
        "Length (m)":      round(length_m,   2),
        "Breadth (m)":     round(breadth_m,  2),
        "Glass % edge":    round(gf * 100,   1),
        "Wall Area (m²)":  round(wall_a_m2,  3),
        "Glass Area (m²)": round(glass_a_m2, 3),
        "Q_wall (W)":      round(q_wall,     1),
        "Q_glass (W)":     round(q_glass,    1),
        "Q_people (W)":    round(q_people,   1),
        "Q_total (W)":     round(q_total,    1),
        "TR":              round(tr,         3),
        "Objects":         ", ".join(room.get("objects_inside", [])) or "—",
        "_poly":           poly,
        "_gf":             gf,
        "_is_glass":       is_glass,
        "_color":          "#00d4ff" if is_glass else COLORS[i % len(COLORS)],
    })

df           = pd.DataFrame(room_data)
display_cols = [c for c in df.columns if not c.startswith("_")]

# ══════════════════════════════════════════════════════
#  FLOOR PLAN VISUALISATION
# ══════════════════════════════════════════════════════
st.subheader("🗺️ Detected Floor Plan")
fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333")

if show_raw:
    for l in raw_wall_lines[:8000]:
        ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
                color="#1e3a50", lw=0.3, alpha=0.4)

# Wall lines
for l in raw_wall_lines[:8000]:
    ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
            color="#3a6a8a", lw=0.6, alpha=0.5, zorder=1)

# Glass lines
for l in raw_glass_lines:
    ax.plot([l[0][0],l[1][0]], [l[0][1],l[1][1]],
            color="#00d4ff", lw=1.4, alpha=0.75, zorder=2)

# Bridges
for br in bridges_used:
    xs, ys = br.xy
    ax.plot(xs, ys, color="#ff4444", lw=1.0, ls="--", alpha=0.7, zorder=2)

# Wall cavities (Mode B)
for cav in wall_cavities:
    if cav.geom_type == "Polygon":
        xs, ys = cav.exterior.xy
        ax.fill(xs, ys, color="dimgray", alpha=0.8, zorder=2)

# Rooms
for row in room_data:
    poly   = row["_poly"]
    color  = row["_color"]
    ig     = row["_is_glass"]
    gf     = row["_gf"]
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, alpha=0.38 if ig else 0.22, color=color, zorder=3)
    ax.plot(xs, ys, color="#00d4ff" if ig else color,
            lw=2.5 if ig else 1.6, zorder=4)
    cx, cy = poly.centroid.x, poly.centroid.y
    label  = f"{'🔵' if ig else '🟩'} {row['Room']}\n{row['Area (m²)']} m²\nTR:{row['TR']}"
    if gf > 0.05:
        label += f"\n{row['Glass % edge']}% glass"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.5,
            color="white", fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#000000dd",
                      edgecolor="#00d4ff" if ig else color,
                      linewidth=1.3 if ig else 0.5))

# Furniture objects
for obj in extracted_objects:
    ax.plot(obj["center_x"], obj["center_y"], "ro", markersize=4, zorder=6)

ax.set_aspect("equal")
ax.set_title(
    f"Floor Plan — {n_rooms} rooms ({n_wall} wall + {n_glass} glass) | {mode_label}",
    color="#00d4ff", fontfamily="monospace", fontsize=10)
ax.grid(True, color="#1a2a3a", lw=0.25)

legend_items = [mpatches.Patch(color="#3a6a8a", label="Wall lines")]
if USE_GLASS_MODE:
    legend_items.append(mpatches.Patch(color="#00d4ff", label="Glass lines"))
legend_items.append(mpatches.Patch(color="#ff4444", label=f"Bridges ({len(bridges_used)})"))
ax.legend(handles=legend_items, loc="upper right",
          facecolor="#0f1117", edgecolor="#444", labelcolor="white", fontsize=9)
st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════
st.subheader("📊 Room Measurements & Heat Load")
st.caption("✅ Area and Perimeter = actual polygon geometry (not bounding box)")
st.dataframe(
    df[display_cols].style
      .format({
          "TR":            "{:.3f}",
          "Area (m²)":     "{:.3f}",
          "Perimeter (m)": "{:.3f}",
          "Q_total (W)":   "{:.1f}",
          "Glass % edge":  "{:.1f}",
      })
      .background_gradient(subset=["TR"],           cmap="YlOrRd")
      .background_gradient(subset=["Area (m²)"],    cmap="Greens")
      .background_gradient(subset=["Glass % edge"], cmap="Blues"),
    use_container_width=True)

# ══════════════════════════════════════════════════════
#  HEAT LOAD SUMMARY METRICS
# ══════════════════════════════════════════════════════
st.subheader("🌡️ Heat Load Summary")
total_area = df["Area (m²)"].sum()
total_kw   = df["Q_total (W)"].sum() / 1000
total_tr   = df["TR"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏠 Total Rooms",  n_rooms)
c2.metric("🔵 Glass Rooms",  n_glass)
c3.metric("📐 Total Area",   f"{total_area:.2f} m²")
c4.metric("⚡ Total Load",   f"{total_kw:.2f} kW")
c5.metric("❄️ Total TR",     f"{total_tr:.2f} TR")

# ══════════════════════════════════════════════════════
#  TR BAR CHART
# ══════════════════════════════════════════════════════
st.subheader("📈 TR per Room")
fig2, ax2 = plt.subplots(figsize=(max(10, n_rooms), 4))
fig2.patch.set_facecolor("#0f1117"); ax2.set_facecolor("#0f1117")
ax2.tick_params(colors="#888")
for sp in ax2.spines.values(): sp.set_edgecolor("#333")
bars = ax2.bar(df["Room"], df["TR"],
               color=[r["_color"] for r in room_data],
               alpha=0.85, edgecolor="#ffffff11")
for bar, row in zip(bars, room_data):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + total_tr * 0.003,
             f"{row['TR']:.3f}",
             ha="center", va="bottom", color="white", fontsize=8)
ax2.set_xlabel("Room", color="#aaa")
ax2.set_ylabel("TR", color="#aaa")
ax2.set_title("TR per Room  (🔵 = glass-bounded)", color="#00d4ff", fontfamily="monospace")
ax2.grid(axis="y", color="#1a2a3a", lw=0.4)
plt.xticks(rotation=45, ha="right", color="#aaa")
st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════
#  DOWNLOAD
# ══════════════════════════════════════════════════════
st.download_button(
    "⬇️ Download CSV",
    df[display_cols].to_csv(index=False),
    "rooms_heat_load.csv",
    "text/csv")

st.divider()
st.caption(
    f"v12 UNIFIED | {mode_label} | {n_rooms} rooms ({n_glass} glass + {n_wall} wall) | "
    f"{total_tr:.2f} TR | "
    f"Wall:{','.join(sorted(WALL_LAYERS))} | "
    f"Glass:{','.join(sorted(GLASS_LAYERS)) or 'none'}")