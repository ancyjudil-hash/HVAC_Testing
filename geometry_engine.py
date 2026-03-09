"""
geometry_engine.py
==================
CAD Room Extractor — Pure geometry logic (no Streamlit UI).

Contains:
  - DXF entity extraction   : extract_all_v8()
  - Segment helpers         : ent_to_segments(), apply_m44(), effective_layer()
  - Mode A detection        : detect_rooms_mode_a()   ← v8 exact engine
  - Mode B detection        : detect_rooms_mode_b()   ← v11 endpoint-bridging
  - Furniture clustering    : process_furniture_to_objects()
  - Shape helpers           : compactness(), aspect_ratio(), is_outer_envelope()
  - Glass helpers           : edge_glass_fraction(), has_any_glass_edge()
  - Mode B entity extractor : process_entity_v11()

Import from app.py:
  from geometry_engine import (
      extract_all_v8, process_entity_v11,
      detect_rooms_mode_a, detect_rooms_mode_b,
      process_furniture_to_objects,
  )
"""

import math
import numpy as np
import networkx as nx

import ezdxf
from ezdxf import path as dxf_path
from ezdxf.math import Matrix44, Vec3

from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import polygonize, unary_union

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GEOM_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}


# ─────────────────────────────────────────────────────────────────────────────
#  MATRIX / LAYER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_m44(mat, x, y):
    """Apply a Matrix44 transform to a 2-D point."""
    v = mat.transform(Vec3(x, y, 0))
    return (v.x, v.y)


def effective_layer(entity, parent_layer):
    """
    v8 exact: entities whose own layer is '0' inherit the parent INSERT layer.
    All other entities keep their own layer.
    """
    own = entity.dxf.get("layer", "0")
    return parent_layer if own == "0" else own


def build_matrix(ent):
    """Build a Matrix44 from an INSERT entity (fallback if matrix44() fails)."""
    try:
        return ent.matrix44()
    except Exception:
        ix  = ent.dxf.insert.x
        iy  = ent.dxf.insert.y
        rot = math.radians(ent.dxf.get("rotation", 0))
        sx  = ent.dxf.get("xscale", 1)
        sy  = ent.dxf.get("yscale", 1)
        c, s = math.cos(rot), math.sin(rot)
        return Matrix44([
            [sx*c, -sy*s, 0, ix],
            [sx*s,  sy*c, 0, iy],
            [0,    0,     1, 0 ],
            [0,    0,     0, 1 ],
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  ENTITY → SEGMENTS  (v8 exact)
# ─────────────────────────────────────────────────────────────────────────────
def ent_to_segments(entity, mat, layer, glass_layers_up):
    """
    Convert a single DXF entity into a list of (LineString, is_glass, is_arc).

    v8 EXACT rules:
      LINE / LWPOLYLINE / POLYLINE / CIRCLE / SPLINE:
        is_arc = False

      ARC tessellation segments (31 of them):
        (LineString, is_glass, is_arc=True)    ← is_glass follows the layer
      ARC closing chord (1 extra):
        (LineString, is_glass=False, is_arc=True)  ← always False

      LWPOLYLINE: all vertices preserved — NO chord stripping.
    """
    t        = entity.dxftype()
    is_glass = layer.upper() in glass_layers_up
    res      = []
    try:
        if t == "LINE":
            p1 = apply_m44(mat, entity.dxf.start.x, entity.dxf.start.y)
            p2 = apply_m44(mat, entity.dxf.end.x,   entity.dxf.end.y)
            if p1 != p2:
                res.append((LineString([p1, p2]), is_glass, False))

        elif t == "LWPOLYLINE":
            pts    = [apply_m44(mat, p[0], p[1]) for p in entity.get_points()]
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
            cx = entity.dxf.center.x
            cy = entity.dxf.center.y
            r  = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea <= sa:
                ea += 2 * math.pi
            angles = np.linspace(sa, ea, 32)
            pts    = [apply_m44(mat, cx + r * math.cos(a), cy + r * math.sin(a))
                      for a in angles]
            if len(pts) >= 2:
                for a, b in zip(pts[:-1], pts[1:]):
                    if a != b:
                        res.append((LineString([a, b]), is_glass, True))   # tessellation
                res.append((LineString([pts[0], pts[-1]]), False, True))   # closing chord

        elif t == "CIRCLE":
            cx = entity.dxf.center.x
            cy = entity.dxf.center.y
            r  = entity.dxf.radius
            angles = np.linspace(0, 2 * math.pi, 64)
            pts    = [apply_m44(mat, cx + r * math.cos(a), cy + r * math.sin(a))
                      for a in angles]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))

        elif t == "SPLINE":
            try:    raw_pts = [(p[0], p[1]) for p in entity.control_points]
            except: raw_pts = []
            if len(raw_pts) < 2:
                try:    raw_pts = [(p[0], p[1]) for p in entity.fit_points]
                except: raw_pts = []
            pts = [apply_m44(mat, x, y) for x, y in raw_pts]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    res.append((LineString([a, b]), is_glass, False))

    except Exception:
        pass
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  MODE A EXTRACTOR  (v8 exact — recursive INSERT traversal)
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_v8(layout, doc, allowed_up, glass_layers_up,
                   parent_mat=None, parent_layer="0", depth=0):
    """
    v8 exact recursive extraction.

    Returns list of (LineString, is_glass, is_arc) for all geometry
    on allowed layers, resolving INSERT block transforms up to depth=30.

    Parameters
    ----------
    layout        : ezdxf layout or block definition
    doc           : ezdxf document (for block lookup)
    allowed_up    : set of uppercase layer names to include (WALL | GLASS only)
    glass_layers_up : set of uppercase glass layer names
    parent_mat    : accumulated Matrix44 transform (identity at top level)
    parent_layer  : inherited layer name for entities on layer '0'
    depth         : recursion depth (capped at 30)
    """
    if depth > 30:
        return []
    if parent_mat is None:
        parent_mat = Matrix44()

    out = []
    for ent in layout:
        et = ent.dxftype()
        if et == "INSERT":
            bname = ent.dxf.name
            if bname not in doc.blocks:
                continue
            ins_layer = ent.dxf.get("layer", parent_layer)
            combined  = parent_mat @ build_matrix(ent)
            out.extend(extract_all_v8(
                doc.blocks[bname], doc, allowed_up, glass_layers_up,
                combined, ins_layer, depth + 1))
        elif et in GEOM_TYPES:
            eff = effective_layer(ent, parent_layer).upper()
            if eff not in allowed_up:
                continue
            out.extend(ent_to_segments(ent, parent_mat, eff, glass_layers_up))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MODE B EXTRACTOR  (v11 path-flattening)
# ─────────────────────────────────────────────────────────────────────────────
def process_entity_v11(entity, raw_lines, scale, exclude_arcs=False):
    """
    v11 path-flattening extractor for Mode B.
    Appends [start, end] pairs to raw_lines (list of [[x,y],[x,y]]).
    Uses ezdxf path.flattening() — handles INSERT virtual_entities recursively.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE QUALITY HELPERS  (v8 exact)
# ─────────────────────────────────────────────────────────────────────────────
def compactness(poly):
    """4π·area / perimeter². Range 0–1. Circle = 1. Very low = jagged."""
    if poly.length == 0:
        return 0.0
    return (4 * math.pi * poly.area) / (poly.length ** 2)


def aspect_ratio(poly):
    """BoundingBox max_side / min_side. High = thin strip."""
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if min(w, h) == 0:
        return 999.0
    return max(w, h) / min(w, h)


def is_outer_envelope(poly, all_p, threshold=0.35):
    """
    v8 exact: returns True if this polygon contains the centroids of
    ≥ threshold fraction of all other polygons.
    Used to exclude the building outer boundary.
    """
    others = [p for p in all_p if p is not poly]
    if not others:
        return False
    n = sum(1 for p in others if poly.contains(p.centroid))
    return (n / len(others)) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
#  GLASS EDGE HELPERS  (v8 exact — 3-point sampling per edge)
# ─────────────────────────────────────────────────────────────────────────────
def edge_glass_fraction(poly, glass_u, tol, mult):
    """
    Fraction of the polygon perimeter that runs within (tol × mult) of glass geometry.
    Sampled at mid-point and two quarter-points of each edge (v8 exact).
    Returns 0.0 if no glass union exists.
    """
    if glass_u is None or glass_u.is_empty:
        return 0.0
    coords    = list(poly.exterior.coords)
    total_len = 0.0
    glass_len = 0.0
    cd        = tol * mult
    for i in range(len(coords) - 1):
        p1, p2  = coords[i], coords[i + 1]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        mid     = ((p1[0] + p2[0]) / 2,       (p1[1] + p2[1]) / 2)
        qtr1    = ((p1[0] * 3 + p2[0]) / 4,   (p1[1] * 3 + p2[1]) / 4)
        qtr2    = ((p1[0] + p2[0] * 3) / 4,   (p1[1] + p2[1] * 3) / 4)
        total_len += seg_len
        if any(glass_u.distance(Point(p)) <= cd for p in (mid, qtr1, qtr2)):
            glass_len += seg_len
    return 0.0 if total_len == 0 else min(glass_len / total_len, 1.0)


def has_any_glass_edge(poly, glass_u, tol, mult):
    """
    Binary check: does any edge of the polygon run near glass?
    Same 3-point sampling as edge_glass_fraction (v8 exact).
    """
    if glass_u is None or glass_u.is_empty:
        return False
    coords = list(poly.exterior.coords)
    cd     = tol * mult
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        mid    = ((p1[0] + p2[0]) / 2,       (p1[1] + p2[1]) / 2)
        qtr1   = ((p1[0] * 3 + p2[0]) / 4,   (p1[1] * 3 + p2[1]) / 4)
        qtr2   = ((p1[0] + p2[0] * 3) / 4,   (p1[1] + p2[1] * 3) / 4)
        if any(glass_u.distance(Point(p)) <= cd for p in (mid, qtr1, qtr2)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  MODE A — v8 EXACT DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_a(wall_segs, glass_segs, arc_segs,
                         snap_tol, bridge_tol,
                         glass_edge_thresh, glass_proximity_mult,
                         min_area_m2, max_area_m2,
                         min_compact, max_aspect,
                         unit_factor,
                         log=None):
    """
    v8 EXACT room detection pipeline.

    Parameters
    ----------
    wall_segs / glass_segs / arc_segs : list of LineString
        Segments classified by extract_all_v8() with v8-exact rules:
          wall_segs  = [g for (g,ig,ia) in raw if not ig and not ia]
          glass_segs = [g for (g,ig,ia) in raw if ig  and not ia]   ← NOT ia
          arc_segs   = [g for (g,ig,ia) in raw if ia]               ← ALL arcs
    snap_tol, bridge_tol : float
        Coordinate snapping and gap-bridging tolerances.
    glass_edge_thresh : float
        Minimum glass-edge fraction for a standalone room to be labelled "glass".
    glass_proximity_mult : float
        snap_tol × this = glass search radius.
    min/max_area_m2 : float
        Room area bounds in m².
    min_compact : float
        Minimum compactness score (4π·A/P²). Rejects jagged shapes.
    max_aspect : float
        Maximum aspect ratio (long/short side). Rejects corridors.
    unit_factor : float
        Raw-coordinate area → m². 1_000_000 for mm drawings, 1.0 for m drawings.
    log : callable or None
        Optional logging callback f(message: str). Pass st.info for Streamlit.

    Returns
    -------
    accepted : list of (Polygon, glass_fraction, is_glass_room)
    bridges  : list of LineString  (gap-bridging lines added)
    wall_sn  : list of LineString  (snapped wall segments)
    glass_sn : list of LineString  (snapped glass segments)
    """

    def _log(msg):
        if log:
            log(msg)

    # ── Inner helpers ────────────────────────────────────────────────────────

    def node_snap_segs(segs, tol):
        """Snap all coordinates to nearest multiple of tol, remove duplicates."""
        out = []
        for ls in segs:
            try:
                coords = [(round(x / tol) * tol, round(y / tol) * tol)
                          for x, y in ls.coords]
                dedup = [coords[0]]
                for c in coords[1:]:
                    if c != dedup[-1]:
                        dedup.append(c)
                if len(dedup) >= 2:
                    out.append(LineString(dedup))
            except Exception:
                pass
        return out

    def bridge_gaps(lines, tol):
        """
        Find all dangling endpoints (degree-1 nodes) and pair each with
        its nearest neighbour within tol. Returns bridging LineStrings.
        """
        from collections import defaultdict
        ep = defaultdict(int)
        for ls in lines:
            cs = list(ls.coords)
            ep[cs[0]]  += 1
            ep[cs[-1]] += 1
        dangling = [pt for pt, cnt in ep.items() if cnt == 1]
        if not dangling:
            return []
        bridges, used = [], set()
        arr = np.array(dangling)
        for i, pt in enumerate(dangling):
            if i in used:
                continue
            diffs = arr - np.array(pt)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            dists[i] = np.inf
            j = int(np.argmin(dists))
            if dists[j] <= tol and j not in used:
                bridges.append(LineString([pt, dangling[j]]))
                used.add(i)
                used.add(j)
        return bridges

    # ── Step 1: Snap each type separately, then merge ────────────────────────
    wall_sn  = node_snap_segs(wall_segs,  snap_tol)
    glass_sn = node_snap_segs(glass_segs, snap_tol)
    arc_sn   = node_snap_segs(arc_segs,   snap_tol)
    boundary_snapped = wall_sn + glass_sn + arc_sn

    # ── Step 2: Bridge dangling endpoints ────────────────────────────────────
    bridges        = bridge_gaps(boundary_snapped, bridge_tol)
    lines_for_poly = boundary_snapped + bridges
    _log(f"[Mode A] Gap bridges: {len(bridges)}")

    # ── Step 3: Polygonize ───────────────────────────────────────────────────
    merged   = unary_union(lines_for_poly)
    all_poly = list(polygonize(merged))
    all_poly.sort(key=lambda p: p.area, reverse=True)
    _log(f"[Mode A] Raw polygons: {len(all_poly)}")

    # ── Step 4: Glass union (snapped glass ONLY — no arc geometry) ──────────
    glass_union = unary_union(glass_sn) if glass_sn else None

    # ── Step 5: Outer envelope exclusion — first polygon that contains ≥35% ─
    outer_ids = set()
    for poly in all_poly[:5]:
        if is_outer_envelope(poly, all_poly, threshold=0.35):
            outer_ids.add(id(poly))
            _log(f"  Outer envelope excluded: {poly.area / unit_factor:.1f} m²")
            break

    # ── Step 6: Pass 1 — area + shape + glass fraction filter ───────────────
    candidates = []
    for poly in all_poly:
        if id(poly) in outer_ids:
            continue
        area_m2 = poly.area / unit_factor
        if area_m2 < min_area_m2 or area_m2 > max_area_m2:
            continue
        if compactness(poly)  < min_compact:
            continue
        if aspect_ratio(poly) > max_aspect:
            continue
        gf      = edge_glass_fraction(poly, glass_union, snap_tol, glass_proximity_mult)
        has_gls = has_any_glass_edge(poly,    glass_union, snap_tol, glass_proximity_mult)
        candidates.append((poly, gf, has_gls))

    _log(f"[Mode A] Candidates after shape filter: {len(candidates)}")
    candidates.sort(key=lambda x: x[0].area, reverse=True)

    # ── Step 7: Pass 2 — parent / sub-room classification ───────────────────
    def is_mostly_inside(small, large, tol=0.90):
        try:
            return small.intersection(large).area / small.area >= tol
        except Exception:
            return False

    accepted = []
    for (poly, gf, has_gls) in candidates:
        parent = next(
            ((ap, ag, ai) for ap, ag, ai in accepted
             if is_mostly_inside(poly, ap)),
            None)
        if parent is None:
            # Standalone: always accept. Glass label by fraction threshold.
            accepted.append((poly, gf, gf >= glass_edge_thresh))
        else:
            # Sub-polygon inside parent:
            #   has any glass edge → separate glass-partition room
            #   no glass            → furniture / column stub → ignored
            if has_gls:
                accepted.append((poly, gf, True))

    # ── Step 8: Deduplicate (6 % area diff + 90 % overlap) ──────────────────
    flags = [False] * len(accepted)
    for i in range(len(accepted)):
        if flags[i]:
            continue
        for j in range(i + 1, len(accepted)):
            if flags[j]:
                continue
            ap, bp = accepted[i][0], accepted[j][0]
            if abs(ap.area - bp.area) / max(ap.area, bp.area) > 0.06:
                continue
            try:
                if ap.intersection(bp).area / min(ap.area, bp.area) >= 0.90:
                    flags[j] = True
            except Exception:
                pass
    accepted = [r for r, f in zip(accepted, flags) if not f]

    # ── Step 9: Sort top-left → bottom-right ────────────────────────────────
    accepted.sort(key=lambda r: (-r[0].centroid.y, r[0].centroid.x))

    n_gl = sum(1 for _, _, ig in accepted if ig)
    _log(f"[Mode A] Final: {len(accepted)} rooms  ({len(accepted)-n_gl} wall + {n_gl} glass)")

    return accepted, bridges, wall_sn, glass_sn


# ─────────────────────────────────────────────────────────────────────────────
#  MODE B — v11 ENDPOINT-BRIDGING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_b(raw_wall_lines, extracted_objects,
                         gap_close_tol, max_door_width, min_wall_len,
                         min_area_m2, max_area_m2, unit_factor,
                         outer_area_pct=25.0, exclude_stairs=True,
                         stair_parallel_min=4, stair_angle_tol=8.0,
                         max_stair_area_m2=20.0, min_solidity=0.50,
                         max_aspect_ratio=15.0, max_interior_walls=8,
                         min_closet_area_m2=0.3, log=None):
    """
    v11 Mode B room detection using endpoint bridging.

    Parameters
    ----------
    raw_wall_lines : list of [[x,y],[x,y]]  (from process_entity_v11)
    extracted_objects : list of dicts with 'point' key (from process_furniture_to_objects)
    log : callable or None  (pass st.info for Streamlit debug output)

    Returns
    -------
    rooms : list of room dicts with keys:
        name, room_id, polygon, width, height, area, objects_inside
    wall_cavities : list of Polygon (small non-room closed areas)
    """
    def _log(msg):
        if log:
            log(msg)

    if not raw_wall_lines:
        return [], []

    shapely_walls = [LineString(l) for l in raw_wall_lines
                     if LineString(l).length >= min_wall_len]
    if not shapely_walls:
        return [], []

    merged_walls = unary_union(shapely_walls)
    if merged_walls.geom_type == "MultiLineString":
        lines_list = list(merged_walls.geoms)
    elif merged_walls.geom_type == "LineString":
        lines_list = [merged_walls]
    else:
        lines_list = [g for g in merged_walls.geoms if g.geom_type == "LineString"]

    valid_endpoints = []
    for line in lines_list:
        if line.length > min_wall_len:
            valid_endpoints.append(Point(line.coords[0]))
            valid_endpoints.append(Point(line.coords[-1]))

    _log(f"[Mode B] Valid endpoints: {len(valid_endpoints)}")

    # Gap-closing bridges
    bridges = []
    for i, ep1 in enumerate(valid_endpoints):
        for j, ep2 in enumerate(valid_endpoints):
            if i < j and ep1.distance(ep2) <= gap_close_tol:
                bridges.append(LineString([ep1, ep2]))

    # Door/archway bridges (directional — must be roughly axis-aligned)
    for i, ep1 in enumerate(valid_endpoints):
        for j, ep2 in enumerate(valid_endpoints):
            if i < j:
                dist = ep1.distance(ep2)
                if gap_close_tol < dist <= max_door_width:
                    dx = abs(ep1.x - ep2.x)
                    dy = abs(ep1.y - ep2.y)
                    if dx < 150 or dy < 150:
                        br = LineString([ep1, ep2])
                        if not br.crosses(merged_walls):
                            bridges.append(br)

    _log(f"[Mode B] Bridges: {len(bridges)}")

    noded     = unary_union(lines_list + bridges)
    raw_polys = list(polygonize(noded))
    raw_polys.sort(key=lambda p: p.area, reverse=True)
    _log(f"[Mode B] Raw polygons: {len(raw_polys)}")
    if not raw_polys:
        return [], []

    mnx, mny, mxx, mxy = noded.bounds
    total_bbox     = (mxx - mnx) * (mxy - mny)
    outer_thresh   = total_bbox * (outer_area_pct / 100.0)
    min_area_mm2   = min_area_m2   * unit_factor
    max_area_mm2   = max_area_m2   * unit_factor
    min_closet_mm2 = min_closet_area_m2 * unit_factor
    max_stair_mm2  = max_stair_area_m2  * unit_factor

    def is_staircase(poly, wsegs, min_p, atol):
        try:
            angles = []
            for seg in wsegs:
                mid = Point(
                    (seg.coords[0][0] + seg.coords[-1][0]) / 2,
                    (seg.coords[0][1] + seg.coords[-1][1]) / 2)
                if poly.contains(mid):
                    dx = seg.coords[-1][0] - seg.coords[0][0]
                    dy = seg.coords[-1][1] - seg.coords[0][1]
                    angles.append(math.degrees(math.atan2(dy, dx)) % 180)
            if len(angles) < min_p:
                return False
            angles.sort()
            for ref in angles:
                count = sum(1 for a in angles
                            if abs(a - ref) <= atol or abs(a - ref) >= (180 - atol))
                if count >= min_p:
                    return True
        except Exception:
            pass
        return False

    rooms_data    = []
    wall_cavities = []

    for poly in raw_polys:
        area = poly.area

        if area >= outer_thresh:
            _log(f"  Outer excluded: {area/unit_factor:.1f} m²")
            continue

        if area < min_area_mm2 or area > max_area_mm2:
            if min_closet_mm2 <= area < min_area_mm2:
                buf = poly.buffer(50)
                for obj in extracted_objects:
                    if buf.covers(obj["point"]):
                        rooms_data.append({
                            "width":  round(poly.bounds[2] - poly.bounds[0], 2),
                            "height": round(poly.bounds[3] - poly.bounds[1], 2),
                            "area":   round(area, 2),
                            "polygon": poly,
                            "objects_inside": [],
                        })
                        break
            elif 10_000 < area < min_closet_mm2:
                wall_cavities.append(poly)
            continue

        try:
            hull   = poly.convex_hull
            solid  = area / hull.area if hull.area > 0 else 0
        except Exception:
            solid  = 1.0

        if solid < min_solidity:
            _log(f"  Low solidity ({solid:.2f}): {area/unit_factor:.1f} m²")
            continue

        rmx, rmy, rMx, rMy = poly.bounds
        w  = rMx - rmx
        h  = rMy - rmy
        ar = max(w, h) / max(min(w, h), 1)
        if ar > max_aspect_ratio:
            _log(f"  High aspect ({ar:.1f}): {area/unit_factor:.1f} m²")
            continue

        if exclude_stairs and area <= max_stair_mm2:
            if is_staircase(poly, shapely_walls, stair_parallel_min, stair_angle_tol):
                _log(f"  Stair excluded: {area/unit_factor:.1f} m²")
                continue

        piercing = sum(
            1 for seg in shapely_walls
            if poly.contains(Point(seg.coords[0])) and
               poly.contains(Point(seg.coords[-1])))
        if piercing > max_interior_walls:
            _log(f"  Wall-pierced ({piercing}): {area/unit_factor:.1f} m²")
            continue

        rooms_data.append({
            "width":  round(w, 2),
            "height": round(h, 2),
            "area":   round(area, 2),
            "polygon": poly,
            "objects_inside": [],
        })

    # Remove rooms fully inside another room
    clean = []
    for i, room in enumerate(rooms_data):
        bad = False
        for j, other in enumerate(rooms_data):
            if i == j:
                continue
            try:
                if other["polygon"].contains(room["polygon"].representative_point()):
                    bad = True; break
                inter = room["polygon"].intersection(other["polygon"])
                if inter.area > 0.80 * room["area"] and room["area"] < other["area"]:
                    bad = True; break
            except Exception:
                pass
        if not bad:
            clean.append(room)

    clean.sort(key=lambda x: x["area"], reverse=True)
    for idx, room in enumerate(clean):
        room["name"]    = f"Room {idx+1}"
        room["room_id"] = f"R{idx+1}"

    return clean, wall_cavities


# ─────────────────────────────────────────────────────────────────────────────
#  FURNITURE → OBJECT CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
def process_furniture_to_objects(furn_lines, gap_tol=50):
    """
    Group furniture line segments into objects by proximity (within gap_tol).
    Returns list of dicts: object_id, length, width, center_x, center_y, point.
    """
    if not furn_lines:
        return []
    from shapely.strtree import STRtree

    shapely_lines = [LineString(l) for l in furn_lines]
    buffered      = [ls.buffer(gap_tol) for ls in shapely_lines]
    tree          = STRtree(buffered)

    G = nx.Graph()
    G.add_nodes_from(range(len(furn_lines)))
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