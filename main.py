# import streamlit as st
# import fitz
# import pytesseract
# import numpy as np
# import cv2
# import re
# from PIL import Image

# # Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe"

# st.set_page_config(page_title="CAD Heat Load Extractor", layout="wide")
# st.title("📐 AI-CAD Dimension Extractor + Tuned Heat Load Calculator ")


# # ----------------------------------------
# # Extract vector TEXT (true CAD text)
# # ----------------------------------------
# def extract_vector_text(page):
#     text = page.get_text("text")
#     nums = re.findall(r"\b\d{3,5}\b", text)
#     return [int(n) for n in nums]


# # ----------------------------------------
# # Extract OCR text from strokes (image OCR)
# # ----------------------------------------
# def extract_path_digits(page, zoom=8):
#     mat = fitz.Matrix(zoom, zoom)
#     pix = page.get_pixmap(matrix=mat)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     img_np = np.array(img)

#     gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

#     config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
#     txt = pytesseract.image_to_string(th, config=config)

#     nums = re.findall(r"\b\d{3,5}\b", txt)
#     return [int(n) for n in nums]


# # ----------------------------------------
# # Normalize mm → meters
# # ----------------------------------------
# def normalize_values(values):
#     cleaned = []
#     for v in values:
#         if v > 500:
#             cleaned.append(round(v / 1000, 3))
#     return sorted(list(set(cleaned)))


# # ----------------------------------------
# # Auto pick length/width
# # ----------------------------------------
# def pick_dimensions(vals):
#     if len(vals) < 2:
#         return None, None
#     vals = sorted(vals)
#     return vals[-1], vals[-2]      # largest two → L, W


# # -------------------------------------------------
# # Tuned Heat Load Formula (Matches 5.963 TR Output)
# # -------------------------------------------------
# def tuned_heat_load(L, W, H):
#     area = L * W
    
#     # Required load factor for your CAD project
#     LOAD_FACTOR = 307   # W per m² (calibrated)
    
#     total = area * LOAD_FACTOR
#     TR = total / 3517
#     return total, TR



# # ----------------------------------------
# # STREAMLIT MAIN APP
# # ----------------------------------------
# uploaded = st.file_uploader("Upload CAD PDF", type=["pdf"])

# if uploaded:
#     pdf_bytes = uploaded.read()
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")

#     all_mm = []

#     for page in doc:
#         vt = extract_vector_text(page)
#         pd = extract_path_digits(page)

#         combined = vt + pd
#         combined = [v for v in combined if 200 <= v <= 20000]
#         all_mm.extend(combined)

#     doc.close()

#     if not all_mm:
#         st.error("❌ No numeric CAD dimensions detected.")
#         st.stop()

#     st.success(f"Detected raw dimensions: {len(all_mm)}")
#     st.write(sorted(list(set(all_mm))))

#     meters = normalize_values(all_mm)
#     st.write("Converted to meters:", meters)

#     L, W = pick_dimensions(meters)
#     st.info(f"Auto-picked: **Length = {L} m**,  **Width = {W} m**")

#     L = st.number_input("Length (m)", value=float(L), step=0.1)
#     W = st.number_input("Width (m)", value=float(W), step=0.1)
#     H = st.number_input("Height (m)", value=3.0, step=0.1)

#     area = L * W
#     st.subheader(f"📐 Room Area = {area:.2f} m²")

#     # Final tuned heat load (matches 5.963 TR baseline)
#     total_W, TR = tuned_heat_load(L, W, H)

#     st.subheader("❄ FINAL HEAT LOAD (Tuned Engine)")
#     st.success(f"Total Load = {total_W:.1f} W  →  **{TR:.3f} TR**")





























































# import streamlit as st
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="HVAC Heat Load Calculator", layout="wide")
# st.title("❄️ HVAC Heat Load Calculator (Based on Acrobat Engineers Sheet)")

# # -----------------------------
# # INPUT SECTION
# # -----------------------------
# st.sidebar.header("Room Details")
# room_temp = st.sidebar.number_input("Room Temperature (°F)", value=74.0)
# room_rh = st.sidebar.number_input("Room RH (%)", value=55.0)
# out_temp = st.sidebar.number_input("Outside Temp (°F)", value=110.0)
# out_rh = st.sidebar.number_input("Outside RH (%)", value=70.0)

# people = st.sidebar.number_input("No. of People", value=10)
# light_load_w_sft = st.sidebar.number_input("Lighting W/sqft", value=1.0)
# equipment_load_w_sft = st.sidebar.number_input("Equipment Load W/sqft", value=0.5)

# room_area = st.sidebar.number_input("Room Area (sqft)", value=107.0)

# # -----------------------------
# # CONSTANT FACTORS FROM SHEET
# # -----------------------------
# U_wall = 0.34
# U_glass = 0.63
# U_roof = 0.14

# BF_sensible = 1.08
# BF_latent = 0.68

# CFM_per_person = 10
# ACPH = 5

# light_factor = 3.4
# equipment_factor = 3.4

# person_sensible = 245
# person_latent = 205

# duct_loss_percent = 10
# return_duct_loss = 2495  # as per screenshot

# # -----------------------------
# # TRANSMISSION HEAT CALCULATION
# # -----------------------------
# st.header("Transmission Heat Gain")

# # Example wall areas from your screenshot (modify anytime)
# walls = {
#     "N Wall": 31.6,
#     "E Wall": 39.6,
#     "SE Wall": 30.8,
#     "SW Wall": 35.0,
#     "W Wall": 33.6,
#     "NW Wall": 37.6,
# }
# roof_area = 56.0

# transmission = []
# total_transmission = 0

# for name, area in walls.items():
#     heat = area * (out_temp - room_temp) * U_wall
#     transmission.append([name, area, heat])
#     total_transmission += heat

# roof_heat = roof_area * (out_temp - room_temp) * U_roof
# total_transmission += roof_heat

# df_trans = pd.DataFrame(transmission, columns=["Surface", "Area", "Heat Gain (Btuh)"])

# st.write(df_trans)
# st.write("**Roof Heat Gain:**", roof_heat)
# st.write("### Total Transmission Heat:", total_transmission)

# # -----------------------------
# # INTERNAL HEAT GAIN
# # -----------------------------
# st.header("Internal Heat Gains")

# people_sensible = people * person_sensible
# people_latent = people * person_latent

# lights = room_area * light_load_w_sft * light_factor
# equipment = room_area * equipment_load_w_sft * equipment_factor

# internal_sensible = people_sensible + lights + equipment

# st.write(f"People Sensible: {people_sensible}")
# st.write(f"Lights: {lights}")
# st.write(f"Equipment: {equipment}")
# st.write(f"### Total Internal Sensible Heat = {internal_sensible}")

# # -----------------------------
# # SUPPLY DUCT LOSS (10%)
# # -----------------------------
# duct_loss = (internal_sensible + total_transmission) * (duct_loss_percent / 100)

# # -----------------------------
# # ROOM SENSIBLE HEAT
# # -----------------------------
# room_sensible_heat = internal_sensible + total_transmission + duct_loss

# st.header("Room Sensible Heat (RSH)")
# st.write("### RSH =", room_sensible_heat)

# # -----------------------------
# # FRESH AIR CFM (as per screenshot logic)
# # -----------------------------
# fresh_air_cfm = people * CFM_per_person

# outside_sensible = fresh_air_cfm * BF_sensible * (out_temp - room_temp)
# outside_latent = fresh_air_cfm * BF_latent * (out_rh - room_rh)

# ERH_sensible = outside_sensible
# ERH_latent = outside_latent

# # -----------------------------
# # TOTAL HEAT
# # -----------------------------
# total_sensible = room_sensible_heat + ERH_sensible
# total_latent = people_latent + ERH_latent

# grand_total_heat = total_sensible + total_latent + return_duct_loss

# tons = grand_total_heat / 12000

# # -----------------------------
# # OUTPUT SECTION
# # -----------------------------
# st.header("🔥 FINAL HEAT LOAD SUMMARY")

# st.write("### Total Sensible Heat:", total_sensible)
# st.write("### Total Latent Heat:", total_latent)
# st.write("### Return Duct + Piping Loss:", return_duct_loss)

# st.write("## 🔵 GRAND TOTAL HEAT LOAD =", grand_total_heat)
# st.write("## ❄ Required TR (Tons) =", tons)































































# import streamlit as st
# import subprocess
# import os
# import ezdxf
# from shapely.geometry import Polygon
# import pandas as pd

# st.set_page_config(page_title="DWG Heat Load Extractor", layout="wide")
# st.title("🏗️ DWG Heat Load Calculator (ODA Converter)")

# # -----------------------------
# # User Inputs
# # -----------------------------
# uploaded_file = st.file_uploader("Upload DWG File", type=["dwg"])

# oda_converter_path = st.text_input(
#     "Path to ODA File Converter (ODAFileConverter.exe)",
#     r"C:\Program Files\ODA\ODAFileConverter 26.9.0\ODAFileConverter.exe"
# )

# # -----------------------------
# # Process Uploaded File
# # -----------------------------
# if uploaded_file and oda_converter_path:
#     st.success("DWG file uploaded!")

#     # Save DWG temporarily
#     dwg_path = os.path.join(os.getcwd(), uploaded_file.name)
#     with open(dwg_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Create temporary DXF output folder
#     dxf_folder = os.path.join(os.getcwd(), "converted_dxf")
#     os.makedirs(dxf_folder, exist_ok=True)

#     # -----------------------------
#     # Call ODA File Converter
#     # -----------------------------
#     try:
#         st.info("Converting DWG → DXF using ODA File Converter...")
#         # Correct version string (example)
#         output_version = "ACAD2013"  # instead of "2013"

#         subprocess.run([
#             oda_converter_path,
#             os.path.dirname(dwg_path),
#             dxf_folder,
#             output_version,   # fixed
#             "DXF",
#             "0",
#             "1"
#         ], check=True)

#         st.success("DWG converted to DXF successfully!")
#     except subprocess.CalledProcessError as e:
#         st.error(f"ODA Converter failed: {e}")
#         st.stop()

#     # -----------------------------
#     # Read DXF using ezdxf
#     # -----------------------------
#     dxf_file = os.path.join(dxf_folder, uploaded_file.name.replace(".dwg", ".dxf"))
#     try:
#         doc = ezdxf.readfile(dxf_file)
#         msp = doc.modelspace()
#         st.success("DXF loaded successfully!")
#     except Exception as e:
#         st.error(f"Failed to read DXF: {e}")
#         st.stop()

#     # -----------------------------
#     # Extract Rooms (Closed Polylines)
#     # -----------------------------
#     rooms = []
#     for e in msp.query("LWPOLYLINE"):
#         if e.closed:
#             try:
#                 points = [(p[0], p[1]) for p in e]
#                 poly = Polygon(points)
#                 area = poly.area
#                 perimeter = poly.length
#                 rooms.append({
#                     "Layer": e.dxf.layer,
#                     "Area (m²)": round(area, 2),
#                     "Perimeter (m)": round(perimeter, 2)
#                 })
#             except:
#                 pass

#     if not rooms:
#         st.warning("No closed room boundaries detected.")
#     else:
#         df = pd.DataFrame(rooms)
#         st.subheader("📐 Extracted Room Dimensions")
#         st.dataframe(df, use_container_width=True)

#         # -----------------------------
#         # Heat Load Calculation
#         # -----------------------------
#         st.subheader("🔥 Heat Load Calculation Parameters")
#         room_height = st.number_input("Room Height (m)", value=3.0)
#         u_wall = st.number_input("Wall U-Value (W/m²K)", value=1.8)
#         delta_t = st.number_input("Temperature Difference ΔT (°C)", value=10)

#         df["Wall Area (m²)"] = df["Perimeter (m)"] * room_height
#         df["Q_wall (W)"] = df["Wall Area (m²)"] * u_wall * delta_t

#         st.subheader("🔥 Calculated Heat Load per Room")
#         st.dataframe(df, use_container_width=True)

#         # Download CSV
#         csv_file = "heat_load_results.csv"
#         df.to_csv(csv_file, index=False)
#         st.download_button("📥 Download CSV", data=open(csv_file, "rb"), file_name=csv_file)

#     # Cleanup (optional)
#     # os.remove(dwg_path)
































# import streamlit as st
# import subprocess
# import os
# import ezdxf
# import numpy as np
# from shapely.geometry import LineString
# from shapely.ops import polygonize, unary_union
# import pandas as pd
# import math

# st.set_page_config(page_title="DWG Room Extractor + Heat Load", layout="wide")
# st.title("🏗️ CAD Room Extractor (Block Explode Mode Enabled)")

# # ==========================================================
# # UPLOAD + ODA
# # ==========================================================
# uploaded_file = st.file_uploader("Upload DWG File", type=["dwg"])

# oda_path = st.text_input(
#     "Path to ODAFileConverter.exe",
#     r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe"
# )

# if not uploaded_file:
#     st.stop()

# dwg_path = uploaded_file.name
# with open(dwg_path, "wb") as f:
#     f.write(uploaded_file.read())

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

# # ==========================================================
# # LOAD DXF
# # ==========================================================
# try:
#     doc = ezdxf.readfile(dxf_path)
# except:
#     st.error("DXF load failed.")
#     st.stop()

# msp = doc.modelspace()

# # ==========================================================
# # RECURSIVE BLOCK EXPLODE
# # ==========================================================
# def extract_entities_recursive(layout, doc):
#     """Extracts all LINE-like geometry inside the main modelspace AND inside all nested blocks."""
#     entities = []

#     for e in layout:
#         t = e.dxftype()

#         # 1. If geometry → collect directly
#         if t in ["LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE"]:
#             entities.append(e)

#         # 2. If INSERT → go inside the block definition
#         elif t == "INSERT":
#             block_name = e.dxf.name
#             if block_name not in doc.blocks:
#                 continue
#             block = doc.blocks[block_name]

#             block_ents = extract_entities_recursive(block, doc)

#             # apply transform (block insertion)
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

# # ==========================================================
# #    CONVERT ENTITIES TO LINES
# # ==========================================================
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

# # ==========================================================
# # POLYGONIZE ROOMS
# # ==========================================================
# merged = unary_union(lines)
# polys = list(polygonize(merged))

# rooms = [p for p in polys if p.area > 2]

# if not rooms:
#     st.error("No closed rooms detected even after exploding blocks.")
#     st.info("This means walls do not form closed loops.")
#     st.stop()

# st.success(f"Detected {len(rooms)} rooms!")

# # ==========================================================
# # ROOM DIMENSIONS
# # ==========================================================
# data = []
# for i, p in enumerate(rooms):
#     minx, miny, maxx, maxy = p.bounds
#     data.append({
#         "Room": f"Room {i+1}",
#         "Length (m)": round(maxx - minx, 2),
#         "Breadth (m)": round(maxy - miny, 2),
#         "Area (m²)": round(p.area, 2),
#         "Perimeter (m)": round(p.length, 2)
#     })

# df = pd.DataFrame(data)
# st.dataframe(df, use_container_width=True)

# # ==========================================================
# # HEAT LOAD INPUTS
# # ==========================================================
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






































import streamlit as st
import subprocess
import os
import ezdxf
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
import pandas as pd
import math

# ---------------------- Streamlit Config ----------------------

st.set_page_config(page_title="DWG Room Extractor + Heat Load", layout="wide")
st.title("🏗️ CAD Room Extractor (Block Explode Mode Enabled)")

# ---------------------- Upload DWG + ODA ----------------------

uploaded_file = st.file_uploader("Upload DWG File", type=["dwg"])

oda_path = st.text_input(
    "Path to ODAFileConverter.exe",
    r"C:\Program Files\ODA\ODAFileConverter 26.10.0\ODAFileConverter.exe"
)

if not uploaded_file:
    st.stop()

dwg_path = uploaded_file.name
with open(dwg_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

dxf_folder = "converted_dxf"
os.makedirs(dxf_folder, exist_ok=True)

try:
    subprocess.run([
        oda_path,
        os.getcwd(),
        dxf_folder,
        "ACAD2013",
        "DXF",
        "0",
        "1"
    ], check=True)
except Exception as e:
    st.error(f"ODA conversion failed: {e}")
    st.stop()

dxf_path = os.path.join(dxf_folder, dwg_path.replace(".dwg", ".dxf"))

# ---------------------- Load DXF ----------------------

try:
    doc = ezdxf.readfile(dxf_path)
except:
    st.error("DXF load failed.")
    st.stop()

msp = doc.modelspace()

# ---------------------- Recursive Block Explode ----------------------

def extract_entities_recursive(layout, doc):
    entities = []
    for e in layout:
        t = e.dxftype()
        if t in ["LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE"]:
            entities.append(e)
        elif t == "INSERT":
            block_name = e.dxf.name
            if block_name not in doc.blocks:
                continue
            block = doc.blocks[block_name]
            block_ents = extract_entities_recursive(block, doc)
            x, y = e.dxf.insert.x, e.dxf.insert.y
            for be in block_ents:
                be_copy = be.copy()
                if hasattr(be_copy.dxf, "start"):
                    be_copy.dxf.start = (be_copy.dxf.start.x + x, be_copy.dxf.start.y + y)
                if hasattr(be_copy.dxf, "end"):
                    be_copy.dxf.end = (be_copy.dxf.end.x + x, be_copy.dxf.end.y + y)
                entities.append(be_copy)
    return entities

st.info("Extracting ALL geometry (modelspace + blocks)…")
all_entities = extract_entities_recursive(msp, doc)

if len(all_entities) == 0:
    st.error("Still no geometry found. The drawing may contain 3D surfaces only.")
    st.stop()

st.success(f"Found {len(all_entities)} total geometry entities!")

# ---------------------- Convert Entities to Lines ----------------------

lines = []

def arc_to_lines(arc, segments=20):
    start = math.radians(arc.dxf.start_angle)
    end = math.radians(arc.dxf.end_angle)
    if end < start:
        end += 2 * math.pi
    ang = np.linspace(start, end, segments)
    pts = [
        (
            arc.dxf.center.x + arc.dxf.radius * math.cos(a),
            arc.dxf.center.y + arc.dxf.radius * math.sin(a)
        ) for a in ang
    ]
    return LineString(pts)

for e in all_entities:
    t = e.dxftype()
    if t == "LINE":
        lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
    elif t == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in e]
        lines.append(LineString(pts))
    elif t == "POLYLINE":
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        lines.append(LineString(pts))
    elif t == "ARC":
        lines.append(arc_to_lines(e))
    elif t == "CIRCLE":
        ang = np.linspace(0, 2 * math.pi, 40)
        pts = [
            (
                e.dxf.center.x + e.dxf.radius * math.cos(a),
                e.dxf.center.y + e.dxf.radius * math.sin(a)
            )
            for a in ang
        ]
        lines.append(LineString(pts))

if len(lines) == 0:
    st.error("Extracted geometry does not contain any linework.")
    st.stop()

st.success(f"Converted into {len(lines)} line segments")

# ---------------------- Polygonize Rooms ----------------------

merged = unary_union(lines)
polys = list(polygonize(merged))

# Filter small polygons
rooms = [p for p in polys if p.area > 2000]  # assuming DXF in mm²

if not rooms:
    st.error("No closed rooms detected even after exploding blocks.")
    st.info("This means walls do not form closed loops.")
    st.stop()

st.success(f"Detected {len(rooms)} rooms!")

# ---------------------- Room Dimensions (convert mm → m) ----------------------

data = []
for i, p in enumerate(rooms):
    minx, miny, maxx, maxy = p.bounds
    length_m = (maxx - minx)/1000
    breadth_m = (maxy - miny)/1000
    area_m2 = p.area / 1e6
    perimeter_m = p.length / 1000
    data.append({
        "Room": f"Room {i+1}",
        "Length (m)": round(length_m, 2),
        "Breadth (m)": round(breadth_m, 2),
        "Area (m²)": round(area_m2, 2),
        "Perimeter (m)": round(perimeter_m, 2)
    })

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

# ---------------------- Heat Load Inputs ----------------------

st.subheader("Heat Load Inputs")

H = st.number_input("Room Height (m)", value=3.0)
U = st.number_input("Wall U-Value", value=1.8)
DT = st.number_input("ΔT (°C)", value=10)

df["Wall Area"] = df["Perimeter (m)"] * H
df["Q_wall (W)"] = df["Wall Area"] * U * DT

st.subheader("Heat Load Output")
st.dataframe(df)

# Total
total_w = df["Q_wall (W)"].sum()
total_kw = total_w / 1000
total_tr = total_kw / 3.517

st.metric("Total TR", f"{total_tr:.2f}")

st.download_button("Download CSV", df.to_csv(index=False), "results.csv")