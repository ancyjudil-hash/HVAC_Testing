"""
object_detection_v22.py  — Drop-in replacement for the object detection
section inside app.py (v21 → v22).

WHAT CHANGED vs v21
════════════════════════════════════════════════════════════════════
Layer 1 — DXF geometry hint
  • Before calling GPT, count furniture clusters from the CAD engine
    (extracted_objects) that fall inside the room polygon.
  • Pass this count to GPT as a hard constraint:
      "The CAD engine detected N furniture clusters inside this room.
       Make sure your total object count is at least N."

Layer 2 — Full-room vision (unchanged crop, but better prompt)
  • System prompt now includes the CAD cluster count.
  • Temperature dropped to 0.02 for maximum determinism.
  • Response validated: if GPT returns fewer total items than the
    CAD count, a RETRY is triggered with an explicit nudge.

Layer 3 — Tile sub-crops for large rooms
  • Rooms whose bounding box > TILE_THRESHOLD_M2 (default 15 m²) are
    split into a 2×2 grid of overlapping sub-crops (50 % overlap).
  • Each tile gets its own GPT call with a sub-region coordinate hint.
  • Results are merged back, deduplicating by name + proximity.

Layer 4 — Merge & reconcile
  • Results from Layer 2 and Layer 3 are merged.
  • Duplicate object types (same name in both passes) are merged:
      count = max(L2_count, L3_count)
      instances deduplicated by centroid proximity (< 200 mm).
  • Confidence gate REMOVED — no object is silently dropped.
  • Final result has a "source" field: "full" | "tile" | "both".

COORDINATE CORRECTION
  • _render_room_crop() now embeds the room's minx/miny offset into
    the returned metadata so GPT coordinates can be converted to
    absolute drawing units correctly.

HOW TO USE IN app.py
════════════════════════════════════════════════════════════════════
1. Replace the "AI OBJECT DETECTION v21" section entirely with this
   file's contents (or import from here).
2. The public API is identical:
      detect_objects_in_rooms_ai(
          rooms_unified, raw_wall_lines, use_glass_mode,
          wall_sn, glass_sn, extracted_objects,
          openai_key, progress_callback=None
      ) -> dict[room_id, list[structured_obj_dict]]

3. _objects_display_str() and _build_inventory_df() are unchanged.
"""

import io, base64, json, re, math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shapely.geometry import Point
from shapely.geometry import box as shapely_box

# ─────────────────────────────────────────────────────────────────────────────
#  TUNABLES
# ─────────────────────────────────────────────────────────────────────────────
TILE_THRESHOLD_M2    = 15.0   # rooms larger than this get tile sub-crops
TILE_OVERLAP_FRAC    = 0.50   # 50 % overlap between adjacent tiles
TILE_DPI             = 220    # higher DPI for tiles (smaller area → more detail)
FULL_DPI             = 200    # DPI for full-room crop
FULL_PAD_FRAC        = 0.30   # padding fraction around room
MAX_RETRIES          = 1      # how many times to retry if count < CAD hint
INSTANCE_MERGE_MM    = 200.0  # proximity threshold for deduplicating instances
GPT_TEMPERATURE      = 0.02   # near-deterministic output

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (v22 — confidence gate removed, CAD-hint slot added)
# ─────────────────────────────────────────────────────────────────────────────
_OBJ_SYSTEM_PROMPT_V22 = """You are a senior CAD/architectural floor-plan analyst.

You are shown a ZOOMED-IN, cropped image of ONE room (or a sub-tile of a room)
from a 2-D floor plan.
  • The room boundary = LIGHT YELLOW fill + BOLD ORANGE border.
  • Pink/red rectangles or red dots inside = furniture geometry from the CAD engine.
  • Faint grey grid lines = spatial reference.
  • A scale bar appears at the bottom-left of the image.

════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════
Identify EVERY DISTINCT object drawn inside the highlighted room boundary.
For EACH unique object TYPE:
  • Count exactly how many instances are present.
  • Estimate the centroid position of EACH instance in mm,
    measured from the crop image's BOTTOM-LEFT corner.
  • Estimate the bounding-box size (width × height) in mm from the drawing scale.
    Use 0 when genuinely indeterminate.
  • Assign confidence: "high" (clear CAD symbol), "medium" (probable from shape),
    "low" (uncertain but plausible — STILL INCLUDE IT, do not suppress it).

════════════════════════════════════════════════════════════
OBJECT CATALOGUE  — use EXACTLY these names
════════════════════════════════════════════════════════════
Residential:
  Single Bed, Double Bed, Queen Bed, King Bed,
  Sofa (2-seater), Sofa (3-seater), Armchair, Recliner,
  Dining Table, Dining Chair, Coffee Table, Side Table,
  TV Unit, Wardrobe, Study Desk, Office Chair, Bookshelf,
  Kitchen Counter, Kitchen Island, Stove / Hob, Sink (Kitchen),
  Refrigerator, Washing Machine, Dryer,
  Toilet / WC, Washbasin / Vanity, Bathtub, Shower Tray, Shower Enclosure,
  Geyser / Water Heater, Balcony Railing,
  Staircase (straight), Staircase (L-shaped), Staircase (U-shaped),
  Lift / Elevator

Office / Commercial:
  Office Desk (single), Office Desk (L-shaped),
  Office Chair, Conference Table, Meeting Chair,
  Reception Desk, Counter,
  Server Rack, UPS Unit, Patch Panel,
  Printer / Copier, Filing Cabinet,
  Sofa (2-seater), Coffee Table, Pantry Counter,
  Toilet / WC, Urinal, Washbasin / Vanity,
  Lift / Elevator, Staircase (straight), Staircase (L-shaped)

Outdoor / Parking:
  Car Parking Space, Motorbike Space, Planter / Plant,
  Fire Extinguisher, Electrical Panel

════════════════════════════════════════════════════════════
STAIRCASE DETECTION — CRITICAL RULES
════════════════════════════════════════════════════════════
A staircase in a 2-D floor plan looks like:
  • A series of PARALLEL LINES (step risers) evenly spaced across a rectangular
    or L/U-shaped area — like a ladder or comb shape.
  • Often has a diagonal "nosing" or "going" line cutting across the steps.
  • May contain UP / DN / ↑ / ↓ text annotation.
  • Shape variants: straight (rectangle of parallel lines), L-shaped (quarter-turn),
    U-shaped (half-turn with a central landing gap).
DO NOT confuse with: hatching patterns, louvres, grilles, or window glazing bars.
KEY TEST: if you count 4 or more evenly-spaced parallel lines filling a zone → Staircase.

════════════════════════════════════════════════════════════
TOILET / WASHBASIN / URINAL COUNTING RULES  ← CRITICAL
════════════════════════════════════════════════════════════
These fixtures are small and there are often multiple in one room (toilet block).
  • Toilet / WC symbol = a rounded D-shape (pan) with a smaller oval (seat lid).
  • Washbasin / Vanity symbol = a rounded rectangle or oval with a circle (drain).
  • Urinal = narrow rectangular/arch shape, usually in a row along a wall.
COUNT EVERY INDIVIDUAL FIXTURE SYMBOL YOU SEE.
If you see 3 toilet symbols → count=3. Report as ONE entry with count=3.
Do not merge different fixture types into one entry.
SCAN the entire image systematically (left column, right column) before answering.

════════════════════════════════════════════════════════════
COUNTING DISCIPLINE — MANDATORY
════════════════════════════════════════════════════════════
Before finalising your response:
  1. Count every red dot / pink rectangle visible inside the orange border.
     Each distinct cluster = at least one object. Do not ignore any.
  2. The CAD engine count (given in the user message) is a MINIMUM BASELINE.
     Your total object count MUST be ≥ that number.
  3. Do NOT suppress "low" confidence objects. Include ALL objects you can see,
     even if uncertain.

════════════════════════════════════════════════════════════
STRICT RULES
════════════════════════════════════════════════════════════
1. Only report objects CLEARLY VISIBLE inside the ORANGE-bordered room.
2. Do NOT list: doors, windows, wall openings, dimension lines, text annotations.
3. DEDUPLICATE: one JSON entry per unique object type.
   Use count field for multiples. NEVER repeat the same object name.
4. Empty room → return empty array [].
5. Coordinates are from the CROP IMAGE bottom-left corner (not room corner).
6. NEVER suppress low-confidence items — include them all.

════════════════════════════════════════════════════════════
RESPONSE FORMAT — ONLY valid JSON, no markdown, no prose
════════════════════════════════════════════════════════════
[
  {
    "name":       "<name from catalogue above>",
    "count":      <integer ≥ 1>,
    "confidence": "high" | "medium" | "low",
    "instances":  [
      { "x_crop_mm": <number>, "y_crop_mm": <number>,
        "width_mm":  <number>, "height_mm": <number> }
    ],
    "notes": "<optional brief note>"
  }
]
Each object type appears EXACTLY ONCE in the array.
instances array length MUST equal count.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_base(ax, room_polygon, raw_wall_lines, use_glass_mode,
               wall_sn, glass_sn, extracted_objects,
               x0, y0, x1, y1):
    """Shared drawing: walls, room fill, furniture rectangles/dots."""
    from shapely.geometry import box as _box
    crop_box = _box(x0, y0, x1, y1)

    # Grid
    for gx in [x0 + (x1-x0)*t for t in (0.2, 0.4, 0.6, 0.8)]:
        ax.axvline(gx, color="#cccccc", lw=0.4, alpha=0.5, zorder=0)
    for gy in [y0 + (y1-y0)*t for t in (0.2, 0.4, 0.6, 0.8)]:
        ax.axhline(gy, color="#cccccc", lw=0.4, alpha=0.5, zorder=0)

    # Walls
    if use_glass_mode:
        for ls in wall_sn:
            if ls.intersects(crop_box):
                xs, ys = ls.xy
                ax.plot(xs, ys, color="#111111", lw=1.4, alpha=0.90, zorder=1)
        for ls in glass_sn:
            if ls.intersects(crop_box):
                xs, ys = ls.xy
                ax.plot(xs, ys, color="#1560BD", lw=1.8, alpha=0.95, zorder=1)
    else:
        for seg in raw_wall_lines:
            sx0, sy0 = seg[0]; sx1, sy1 = seg[1]
            if (x0 <= max(sx0,sx1) and x1 >= min(sx0,sx1) and
                    y0 <= max(sy0,sy1) and y1 >= min(sy0,sy1)):
                ax.plot([sx0,sx1],[sy0,sy1],
                        color="#111111", lw=1.2, alpha=0.85, zorder=1)

    # Room highlight
    rxs, rys = room_polygon.exterior.xy
    ax.fill(rxs, rys, color="#FFF9C4", alpha=0.35, zorder=2)
    ax.plot(rxs, rys, color="#E65100", lw=3.0, zorder=3)

    # Furniture
    for obj in extracted_objects:
        cx_obj = obj.get("center_x", 0)
        cy_obj = obj.get("center_y", 0)
        if not (x0 <= cx_obj <= x1 and y0 <= cy_obj <= y1):
            continue
        if not room_polygon.buffer(20).contains(Point(cx_obj, cy_obj)):
            continue
        ol = obj.get("length", 0); ow = obj.get("width", 0)
        if ol > 10 and ow > 10:
            rect = plt.Rectangle(
                (cx_obj - ol/2, cy_obj - ow/2), ol, ow,
                linewidth=1.2, edgecolor="#B71C1C", facecolor="#FFCDD2",
                alpha=0.70, zorder=5)
            ax.add_patch(rect)
        else:
            ax.plot(cx_obj, cy_obj, "o", color="#D32F2F", markersize=8,
                    markeredgecolor="#7B0000", markeredgewidth=0.9,
                    alpha=0.95, zorder=5)


def _add_scalebar(ax, x0, y0, x1, y1):
    span    = x1 - x0
    bar_len = span * 0.20
    bx0_s   = x0 + span * 0.05
    bx1_s   = bx0_s + bar_len
    by_s    = y0 + (y1 - y0) * 0.04
    ax.plot([bx0_s, bx1_s], [by_s, by_s], color="#333333", lw=2.5, zorder=7)
    ax.text((bx0_s+bx1_s)/2, by_s + (y1-y0)*0.02,
            f"{bar_len/1000:.1f} m" if bar_len > 500 else f"{bar_len:.0f} mm",
            ha="center", va="bottom", fontsize=7, color="#333333", zorder=7)


def _fig_to_b64(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_room_crop_v22(
    room_polygon, raw_wall_lines, use_glass_mode,
    wall_sn, glass_sn, extracted_objects,
    room_id, room_type,
    pad_frac=FULL_PAD_FRAC, dpi=FULL_DPI,
):
    """
    Full-room crop. Returns (b64_str, crop_meta) where crop_meta contains
    the image's coordinate offsets so GPT coordinates can be corrected.
    """
    minx, miny, maxx, maxy = room_polygon.bounds
    w = maxx - minx; h = maxy - miny
    pad_x = max(w * pad_frac, 80); pad_y = max(h * pad_frac, 80)
    x0, y0 = minx - pad_x, miny - pad_y
    x1, y1 = maxx + pad_x, maxy + pad_y

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f6fa")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.axis("off")

    _draw_base(ax, room_polygon, raw_wall_lines, use_glass_mode,
               wall_sn, glass_sn, extracted_objects, x0, y0, x1, y1)

    cx, cy = room_polygon.centroid.x, room_polygon.centroid.y
    ax.text(cx, cy + h*0.38, f"{room_id}  |  {room_type}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#1a237e", zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#1a237e", alpha=0.95, linewidth=1.4))
    _add_scalebar(ax, x0, y0, x1, y1)

    plt.tight_layout(pad=0.2)
    b64 = _fig_to_b64(fig, dpi)

    # crop_meta: pixel origin in drawing units, for coordinate correction
    crop_meta = {
        "img_x0_draw": x0, "img_y0_draw": y0,
        "img_x1_draw": x1, "img_y1_draw": y1,
        "room_minx":   minx, "room_miny": miny,
    }
    return b64, crop_meta


def render_tile_crop_v22(
    room_polygon, raw_wall_lines, use_glass_mode,
    wall_sn, glass_sn, extracted_objects,
    tile_x0, tile_y0, tile_x1, tile_y1,
    tile_label, room_type, dpi=TILE_DPI,
):
    """
    Sub-tile crop (quadrant of a large room).
    Returns (b64_str, crop_meta).
    """
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f6fa")
    ax.set_xlim(tile_x0, tile_x1); ax.set_ylim(tile_y0, tile_y1)
    ax.set_aspect("equal"); ax.axis("off")

    _draw_base(ax, room_polygon, raw_wall_lines, use_glass_mode,
               wall_sn, glass_sn, extracted_objects,
               tile_x0, tile_y0, tile_x1, tile_y1)

    # Tile boundary marker (dashed blue rect)
    ax.add_patch(plt.Rectangle(
        (tile_x0 + 10, tile_y0 + 10),
        (tile_x1-tile_x0) - 20, (tile_y1-tile_y0) - 20,
        linewidth=1.5, edgecolor="#1565C0", facecolor="none",
        linestyle="--", alpha=0.6, zorder=6))
    ax.text(tile_x0 + (tile_x1-tile_x0)*0.5,
            tile_y1 - (tile_y1-tile_y0)*0.06,
            f"Tile {tile_label}  ({room_type})",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#0D47A1", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#0D47A1", alpha=0.9, linewidth=1.2))
    _add_scalebar(ax, tile_x0, tile_y0, tile_x1, tile_y1)

    plt.tight_layout(pad=0.2)
    b64 = _fig_to_b64(fig, dpi)

    crop_meta = {
        "img_x0_draw": tile_x0, "img_y0_draw": tile_y0,
        "img_x1_draw": tile_x1, "img_y1_draw": tile_y1,
        "room_minx": tile_x0,   "room_miny":   tile_y0,
    }
    return b64, crop_meta


# ─────────────────────────────────────────────────────────────────────────────
#  GPT CALL — single image, returns raw parsed list
# ─────────────────────────────────────────────────────────────────────────────

def _call_gpt(client, b64_img, user_text, max_tokens=2000):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _OBJ_SYSTEM_PROMPT_V22},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_img}",
                               "detail": "high"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        temperature=GPT_TEMPERATURE,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$",       "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        try:    parsed = json.loads(m.group()) if m else []
        except: return []
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list): parsed = v; break
        else: return []
    return parsed if isinstance(parsed, list) else []


def _normalise_items(raw_list):
    """
    Normalise + deduplicate a raw GPT item list.
    NO confidence gate — all items kept.
    """
    cleaned    = []
    seen_names = set()
    for item in raw_list:
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen_names: continue
        seen_names.add(name)

        count      = max(1, int(item.get("count", 1)))
        confidence = str(item.get("confidence", "medium"))
        notes      = str(item.get("notes", "")).strip()
        raw_inst   = item.get("instances", [])

        norm_inst = []
        for inst in raw_inst[:count]:
            if not isinstance(inst, dict): continue
            norm_inst.append({
                # Support both old (x_room_mm) and new (x_crop_mm) key names
                "x_crop_mm":  float(inst.get("x_crop_mm",  inst.get("x_room_mm", 0))),
                "y_crop_mm":  float(inst.get("y_crop_mm",  inst.get("y_room_mm", 0))),
                "width_mm":   float(inst.get("width_mm",  0)),
                "height_mm":  float(inst.get("height_mm", 0)),
            })
        while len(norm_inst) < count:
            norm_inst.append({"x_crop_mm": 0, "y_crop_mm": 0,
                              "width_mm": 0, "height_mm": 0})

        cleaned.append({
            "name":       name,
            "count":      count,
            "confidence": confidence,
            "instances":  norm_inst,
            "notes":      notes,
            "source":     "full",    # overwritten by tile pass
        })
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 2 — full-room vision pass (with retry)
# ─────────────────────────────────────────────────────────────────────────────

def _layer2_full_room(client, b64, crop_meta, room_id, room_type,
                      area_m2, cad_cluster_count, room_bbox):
    minx, miny, maxx, maxy = room_bbox
    room_w = round(maxx - minx)
    room_h = round(maxy - miny)
    unit_hint = "mm" if room_w > 500 else "drawing units"

    base_text = (
        f"Room ID   : {room_id}\n"
        f"Room Type : {room_type}\n"
        f"Floor Area: {area_m2:.2f} m²\n"
        f"Room dims : {room_w} × {room_h} {unit_hint}\n"
        f"CAD engine detected {cad_cluster_count} furniture/fixture clusters "
        f"inside this room. Your total object count MUST be ≥ {cad_cluster_count}.\n\n"
        f"Identify every object drawn inside the ORANGE-bordered room.\n"
        f"Coordinates (x_crop_mm, y_crop_mm) = from the CROP IMAGE bottom-left corner.\n"
        f"Return ONLY a valid JSON array."
    )

    raw = _call_gpt(client, b64, base_text)
    items = _normalise_items(raw)

    # ── Retry if total count < CAD hint ──────────────────────────────────
    total_count = sum(it.get("count", 1) for it in items)
    if total_count < cad_cluster_count:
        retry_text = (
            base_text +
            f"\n\nFIRST ATTEMPT returned {total_count} total objects, "
            f"but the CAD engine found {cad_cluster_count} clusters. "
            f"Look more carefully at small fixtures (toilets, washbasins, "
            f"sinks, chairs, electrical panels). Count every red dot and "
            f"pink rectangle you see. Return the corrected JSON array."
        )
        raw2  = _call_gpt(client, b64, retry_text)
        items2 = _normalise_items(raw2)
        total2 = sum(it.get("count", 1) for it in items2)
        if total2 >= total_count:
            items = items2

    for it in items:
        it["source"] = "full"
    return items


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 3 — tile sub-crops for large rooms
# ─────────────────────────────────────────────────────────────────────────────

def _layer3_tiles(client, room_polygon, raw_wall_lines, use_glass_mode,
                  wall_sn, glass_sn, extracted_objects,
                  room_id, room_type, area_m2, unit_factor):
    """
    Split the room bounding box into a 2×2 grid with 50 % overlap.
    Analyse each tile separately; return merged list.
    """
    minx, miny, maxx, maxy = room_polygon.bounds
    cx_mid = (minx + maxx) / 2
    cy_mid = (miny + maxy) / 2
    ox = (maxx - minx) * TILE_OVERLAP_FRAC / 2   # overlap in x
    oy = (maxy - miny) * TILE_OVERLAP_FRAC / 2   # overlap in y

    tiles = [
        ("TL", minx - ox/2, cy_mid - oy, maxx/2 + cx_mid/2 + ox/2, maxy + oy/2),
        ("TR", cx_mid - ox/2, cy_mid - oy, maxx + ox/2,              maxy + oy/2),
        ("BL", minx - ox/2, miny - oy/2,   cx_mid + ox/2,            cy_mid + oy),
        ("BR", cx_mid - ox/2, miny - oy/2, maxx + ox/2,              cy_mid + oy),
    ]
    # Clamp tile coords
    tiles = [
        (lbl, max(tx0, minx-50), max(ty0, miny-50),
               min(tx1, maxx+50), min(ty1, maxy+50))
        for lbl, tx0, ty0, tx1, ty1 in tiles
    ]

    tile_items = []
    for lbl, tx0, ty0, tx1, ty1 in tiles:
        # Skip tiles fully outside the room
        tile_poly = shapely_box(tx0, ty0, tx1, ty1)
        if room_polygon.intersection(tile_poly).area < 0.05 * room_polygon.area:
            continue

        try:
            b64t, meta_t = render_tile_crop_v22(
                room_polygon, raw_wall_lines, use_glass_mode,
                wall_sn, glass_sn, extracted_objects,
                tx0, ty0, tx1, ty1, lbl, room_type)
        except Exception:
            continue

        tile_w = round(tx1 - tx0); tile_h = round(ty1 - ty0)
        unit_hint = "mm" if tile_w > 500 else "drawing units"
        user_text = (
            f"Room ID   : {room_id} — Tile {lbl}\n"
            f"Room Type : {room_type}\n"
            f"Tile area : {round((tile_w/1000)*(tile_h/1000),1)} m²\n"
            f"Tile dims : {tile_w} × {tile_h} {unit_hint}\n"
            f"This is a sub-crop (quadrant {lbl}) of a larger room.\n"
            f"Report ONLY objects visible inside the ORANGE border in THIS tile.\n"
            f"Coordinates (x_crop_mm, y_crop_mm) = from this TILE's bottom-left corner.\n"
            f"Return ONLY a valid JSON array."
        )
        raw_tile = _call_gpt(client, b64t, user_text)
        norm     = _normalise_items(raw_tile)
        for it in norm:
            it["source"]     = f"tile_{lbl}"
            it["_tile_x0"]   = tx0
            it["_tile_y0"]   = ty0
            # Convert tile-local coords → room-bbox-local coords
            for inst in it["instances"]:
                inst["x_crop_mm"] = inst["x_crop_mm"] + (tx0 - minx)
                inst["y_crop_mm"] = inst["y_crop_mm"] + (ty0 - miny)
        tile_items.extend(norm)

    return tile_items


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 4 — merge + reconcile
# ─────────────────────────────────────────────────────────────────────────────

def _merge_items(full_items, tile_items):
    """
    Merge full-room results with tile results.
    For matching object names:
      count  = max(full_count, tile_count)
      instances deduplicated by centroid proximity < INSTANCE_MERGE_MM
    For names only in tiles: add them in.
    """
    full_by_name = {it["name"]: it for it in full_items}
    tile_by_name = defaultdict(list)
    for it in tile_items:
        tile_by_name[it["name"]].append(it)

    merged = {}
    # Start with full results
    for name, f_item in full_by_name.items():
        merged[name] = dict(f_item)
        merged[name]["source"] = "full"

    # Integrate tile results
    for name, t_list in tile_by_name.items():
        # Combine all tile instances for this name
        all_tile_instances = []
        tile_total_count   = 0
        for t in t_list:
            all_tile_instances.extend(t["instances"])
            tile_total_count = max(tile_total_count, t["count"])

        if name in merged:
            f = merged[name]
            # Pick highest count
            best_count = max(f["count"], sum(t["count"] for t in t_list))

            # Merge instances, deduplicating by proximity
            combined_inst = list(f["instances"])
            for ti in all_tile_instances:
                too_close = any(
                    math.hypot(ti["x_crop_mm"] - ci["x_crop_mm"],
                               ti["y_crop_mm"] - ci["y_crop_mm"])
                    < INSTANCE_MERGE_MM
                    for ci in combined_inst)
                if not too_close:
                    combined_inst.append(ti)

            # Pad/trim to best_count
            while len(combined_inst) < best_count:
                combined_inst.append({"x_crop_mm": 0, "y_crop_mm": 0,
                                      "width_mm": 0, "height_mm": 0})
            combined_inst = combined_inst[:best_count]

            merged[name]["count"]     = best_count
            merged[name]["instances"] = combined_inst
            merged[name]["source"]    = "both"
            # Upgrade confidence if any tile says higher
            for t in t_list:
                conf_rank = {"high": 2, "medium": 1, "low": 0}
                if conf_rank.get(t["confidence"], 0) > conf_rank.get(merged[name]["confidence"], 0):
                    merged[name]["confidence"] = t["confidence"]
        else:
            # New from tile — add it
            best_count = sum(t["count"] for t in t_list)
            merged[name] = {
                "name":       name,
                "count":      best_count,
                "confidence": max(t_list, key=lambda x: {"high":2,"medium":1,"low":0}.get(x["confidence"],0))["confidence"],
                "instances":  all_tile_instances[:best_count],
                "notes":      t_list[0].get("notes", ""),
                "source":     "tile",
            }
            while len(merged[name]["instances"]) < best_count:
                merged[name]["instances"].append({"x_crop_mm":0,"y_crop_mm":0,"width_mm":0,"height_mm":0})

    return list(merged.values())


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT  (drop-in replacement for detect_objects_in_rooms_ai)
# ─────────────────────────────────────────────────────────────────────────────

def detect_objects_in_rooms_ai(
    rooms_unified,
    raw_wall_lines,
    use_glass_mode,
    wall_sn,
    glass_sn,
    extracted_objects,
    openai_key,
    progress_callback=None,
):
    """
    v22 — 4-layer object detection.
    Returns {room_id: [structured_obj_dict, ...], ...}
    Each dict: {"name","count","confidence","instances","notes","source"}
    """
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("openai package not installed.")

    client  = _openai.OpenAI(api_key=openai_key)
    results = {}

    for idx, room in enumerate(rooms_unified):
        rid     = room["room_id"]
        rtype   = room.get("ai_type", "Unknown")
        polygon = room["polygon"]
        uf      = room.get("_unit_factor", 1_000_000)
        area_m2 = round(polygon.area / uf, 2)
        bbox    = polygon.bounds

        if progress_callback:
            progress_callback(idx, len(rooms_unified), rid, rtype)

        # ── Layer 1: DXF cluster count (no API call) ─────────────────────
        cad_clusters = [
            o for o in extracted_objects
            if polygon.buffer(20).covers(Point(o.get("center_x", o.get("pt_xy", (0,0))[0]),
                                               o.get("center_y", o.get("pt_xy", (0,0))[1])))
        ]
        cad_count = len(cad_clusters)

        # ── Layer 2: Full-room vision pass ───────────────────────────────
        try:
            b64_full, meta_full = render_room_crop_v22(
                room_polygon      = polygon,
                raw_wall_lines    = raw_wall_lines,
                use_glass_mode    = use_glass_mode,
                wall_sn           = wall_sn,
                glass_sn          = glass_sn,
                extracted_objects = extracted_objects,
                room_id           = rid,
                room_type         = rtype,
            )
        except Exception:
            results[rid] = []
            continue

        try:
            full_items = _layer2_full_room(
                client, b64_full, meta_full, rid, rtype,
                area_m2, cad_count, bbox)
        except Exception:
            full_items = []

        # ── Layer 3: Tile sub-crops (only for large rooms) ───────────────
        tile_items = []
        if area_m2 > TILE_THRESHOLD_M2:
            try:
                tile_items = _layer3_tiles(
                    client, polygon, raw_wall_lines, use_glass_mode,
                    wall_sn, glass_sn, extracted_objects,
                    rid, rtype, area_m2, uf)
            except Exception:
                tile_items = []

        # ── Layer 4: Merge + reconcile ───────────────────────────────────
        if tile_items:
            final = _merge_items(full_items, tile_items)
        else:
            final = full_items

        results[rid] = final

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS  (unchanged API, works with v22 "source" field)
# ─────────────────────────────────────────────────────────────────────────────

def _objects_display_str(room_id, ai_objects_by_room):
    items = ai_objects_by_room.get(room_id, [])
    if not items:
        return "—"
    parts = []
    for item in items:
        if isinstance(item, dict):
            n    = item.get("count", 1)
            name = item.get("name", "?")
            parts.append(f"{n}× {name}" if n > 1 else name)
        else:
            parts.append(str(item))
    return ", ".join(parts)


def _build_inventory_df(rooms_unified, ai_objects_by_room):
    import pandas as pd
    rows = []
    for room in rooms_unified:
        rid   = room["room_id"]
        rname = room.get("ai_name") or room.get("name") or rid
        rtype = room.get("ai_type", "Other")
        items = ai_objects_by_room.get(rid, [])
        for item in items:
            if not isinstance(item, dict):
                rows.append({
                    "Room": rid, "Room Name": rname, "Room Type": rtype,
                    "Object": str(item), "Count": 1, "Instance #": 1,
                    "Confidence": "—", "Source": "—",
                    "Centroid X (mm)": 0, "Centroid Y (mm)": 0,
                    "Width (mm)": 0, "Height (mm)": 0, "Notes": "",
                })
                continue

            name       = item.get("name", "?")
            count      = item.get("count", 1)
            confidence = item.get("confidence", "—")
            source     = item.get("source", "full")
            notes      = item.get("notes", "")
            instances  = item.get("instances", [])

            for i, inst in enumerate(instances):
                rows.append({
                    "Room":            rid,
                    "Room Name":       rname,
                    "Room Type":       rtype,
                    "Object":          name,
                    "Count":           count,
                    "Instance #":      i + 1,
                    "Confidence":      confidence,
                    "Source":          source,
                    "Centroid X (mm)": round(inst.get("x_crop_mm", inst.get("x_room_mm", 0))),
                    "Centroid Y (mm)": round(inst.get("y_crop_mm", inst.get("y_room_mm", 0))),
                    "Width (mm)":      round(inst.get("width_mm",  0)),
                    "Height (mm)":     round(inst.get("height_mm", 0)),
                    "Notes":           notes,
                })

    if not rows:
        return pd.DataFrame()

    col_order = [
        "Room", "Room Name", "Room Type",
        "Object", "Count", "Instance #", "Confidence", "Source",
        "Centroid X (mm)", "Centroid Y (mm)",
        "Width (mm)", "Height (mm)", "Notes",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in col_order if c in df.columns]]