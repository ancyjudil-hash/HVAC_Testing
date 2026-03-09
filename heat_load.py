"""
heat_load.py
============
Heat load (TR) calculation logic — no Streamlit, no geometry.

Given a list of room polygons with glass-fraction data, computes:
  - Wall surface area
  - Glass surface area
  - Heat gain from walls, glass, and occupants
  - Total cooling load in Watts and TR (Tons of Refrigeration)

Usage
-----
from heat_load import compute_room_heat_loads, summarise_heat_loads

room_data = compute_room_heat_loads(
    rooms_unified,       # list of room dicts from geometry_engine
    unit_factor,         # mm² → m²  (1_000_000 for mm drawings)
    unit_div,            # mm → m    (1000 for mm drawings)
    H          = 3.0,    # room height in m
    U_wall     = 1.8,    # wall U-value W/(m²·K)
    U_glass    = 5.8,    # glass U-value W/(m²·K)
    DT         = 10,     # temperature difference °C
    people_per_room = 2,
    Q_person   = 75,     # sensible heat per person W
)

summary = summarise_heat_loads(room_data)
# summary['total_kw'], summary['total_tr'], summary['total_area_m2']
"""

from __future__ import annotations
from typing import List, Dict, Any

# Conversion: 1 TR = 3517 W
TR_PER_WATT = 1.0 / 3517.0

# Room colour palette for plotting (20 colours, cycles)
ROOM_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#82E0AA", "#F1948A",
    "#85C1E9", "#F0B27A", "#C39BD3", "#76D7C4", "#F9E79F",
    "#AED6F1", "#A9DFBF", "#FAD7A0", "#D2B4DE", "#FFB3BA",
]


def compute_room_heat_loads(
    rooms_unified: List[Dict[str, Any]],
    unit_factor: float,
    unit_div: float,
    H: float = 3.0,
    U_wall: float = 1.8,
    U_glass: float = 5.8,
    DT: float = 10.0,
    people_per_room: int = 2,
    Q_person: float = 75.0,
) -> List[Dict[str, Any]]:
    """
    Compute heat load for each room and return an enriched list of dicts.

    Parameters
    ----------
    rooms_unified : list of room dicts
        Each dict must have:
            polygon   : shapely Polygon
            gf        : float  — glass-edge fraction 0–1
            is_glass  : bool   — room type label
            name      : str
            room_id   : str
            objects_inside : list[str]

    unit_factor : float
        Divide polygon.area by this to get m².   E.g. 1_000_000 for mm drawings.
    unit_div : float
        Divide polygon.length by this to get m.  E.g. 1000 for mm drawings.
    H : float
        Room height in metres.
    U_wall : float
        Wall U-value in W/(m²·K).
    U_glass : float
        Glass U-value in W/(m²·K).
    DT : float
        Design temperature difference in °C (inside–outside).
    people_per_room : int
        Number of occupants assumed per room.
    Q_person : float
        Sensible heat gain per person in Watts.

    Returns
    -------
    list of dicts, one per room, with keys:
        Room, Type, Area (m²), Perimeter (m), Length ref (m), Breadth ref (m),
        Glass % edge,
        Wall Area (m²), Glass Area (m²),
        Q_wall (W), Q_glass (W), Q_people (W), Q_total (W), TR,
        Objects,
        _poly, _gf, _is_glass, _color       ← private keys for plotting
    """
    result = []
    for i, room in enumerate(rooms_unified):
        poly     = room["polygon"]
        gf       = room.get("gf", 0.0)
        is_glass = room.get("is_glass", False)

        # Geometry
        area_m2   = poly.area   / unit_factor
        perim_m   = poly.length / unit_div
        minx, miny, maxx, maxy = poly.bounds
        length_m  = (maxx - minx) / unit_div
        breadth_m = (maxy - miny) / unit_div

        # Surface areas
        glass_p_m = perim_m * gf
        wall_p_m  = perim_m * (1.0 - gf)
        wall_a_m2 = wall_p_m  * H
        glass_a_m2= glass_p_m * H

        # Heat gains
        q_wall   = wall_a_m2  * U_wall  * DT
        q_glass  = glass_a_m2 * U_glass * DT
        q_people = people_per_room * Q_person
        q_total  = q_wall + q_glass + q_people
        tr       = q_total * TR_PER_WATT

        color = "#00d4ff" if is_glass else ROOM_COLORS[i % len(ROOM_COLORS)]

        result.append({
            # Display columns
            "Room":            room.get("name", f"Room {i+1}"),
            "Type":            "🔵 Glass" if is_glass else "🟩 Wall",
            "Area (m2)":       round(area_m2,   3),
            "Perimeter (m)":   round(perim_m,   3),
            "Length ref (m)":  round(length_m,  2),
            "Breadth ref (m)": round(breadth_m, 2),
            "Glass % edge":    round(gf * 100,  1),
            "Wall Area (m2)":  round(wall_a_m2, 3),
            "Glass Area (m2)": round(glass_a_m2,3),
            "Q_wall (W)":      round(q_wall,    1),
            "Q_glass (W)":     round(q_glass,   1),
            "Q_people (W)":    round(q_people,  1),
            "Q_total (W)":     round(q_total,   1),
            "TR":              round(tr,         3),
            "Objects":         ", ".join(room.get("objects_inside", [])) or "—",
            # Private (for plotting — not shown in table)
            "_poly":     poly,
            "_gf":       gf,
            "_is_glass": is_glass,
            "_color":    color,
        })

    return result


def summarise_heat_loads(room_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate totals across all rooms.

    Returns
    -------
    dict with:
        total_area_m2   : float
        total_q_w       : float   (total heat gain in Watts)
        total_kw        : float
        total_tr        : float
        n_rooms         : int
        n_glass         : int
        n_wall          : int
    """
    total_area = sum(r["Area (m2)"]  for r in room_data)
    total_q    = sum(r["Q_total (W)"] for r in room_data)
    n_glass    = sum(1 for r in room_data if r["_is_glass"])

    return {
        "total_area_m2": round(total_area, 3),
        "total_q_w":     round(total_q,    1),
        "total_kw":      round(total_q / 1000.0, 3),
        "total_tr":      round(total_q * TR_PER_WATT, 3),
        "n_rooms":       len(room_data),
        "n_glass":       n_glass,
        "n_wall":        len(room_data) - n_glass,
    }


def display_columns() -> List[str]:
    """Return the list of column names suitable for a table (excludes _ private keys)."""
    return [
        "Room", "Type", "Area (m2)", "Perimeter (m)",
        "Length ref (m)", "Breadth ref (m)", "Glass % edge",
        "Wall Area (m2)", "Glass Area (m2)",
        "Q_wall (W)", "Q_glass (W)", "Q_people (W)", "Q_total (W)",
        "TR", "Objects",
    ]