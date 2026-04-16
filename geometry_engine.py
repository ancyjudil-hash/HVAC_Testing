"""
geometry_engine.py  — v22
=========================
CAD Room Extractor — Pure geometry logic (no Streamlit UI).

v22 Changes (on top of v21):
  - NEW Mode D: Closed-polyline fast-path detector.
      * Scans ALL layers for LWPOLYLINE/POLYLINE with closed=True or
        first_pt ≈ last_pt. Each closed poly is a direct room candidate.
      * Works even when wall layers are unknown / not configured.
      * Returns immediately if rooms are found — fastest path.
  - FIXED outer envelope detection: uses BOTH centroid-containment AND
    area-dominance (poly area > 60% of bounding-box area of all rooms).
    Prevents the building outline from being kept as a "room".
  - FIXED unit_factor auto-detection: smarter span-based heuristic:
        span > 50000 → micrometres  → factor 1e12, div 1e6
        span > 5000  → millimetres → factor 1e6,  div 1e3
        span > 500   → centimetres → factor 1e4,  div 1e2
        else         → metres      → factor 1,    div 1
  - FIXED Mode C: skip_layers now also skips layers whose entities are
    almost all very-short segments (< 5 units) — catches hatching layers
    that bloat the segment count and create false tiny polygons.
  - detect_rooms_mode_a / mode_b: outer-envelope fix applied consistently.
  - All previous functionality preserved.
"""

import math
import numpy as np
import networkx as nx

import ezdxf
from ezdxf import path as dxf_path
from ezdxf.math import Matrix44, Vec3

from shapely.geometry import LineString, Point, MultiLineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GEOM_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}

_WALL_KW  = {"wall","walls","base","main","boundary","outline","struct",
             "arch","build","a-wall","cen","ext","partition"}
_GLASS_KW = {"glass","glaz","window","curtain","facade","glazing","win","gl"}
_SKIP_KW  = {"text","dim","hatch","title","vp","defpoint","furniture",
             "plant","planter","door","electrical","elec","plumbing",
             "mech","annotation","border","viewport","symbol","tag",
             "room","space","area","no","note"}


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT FACTOR AUTO-DETECTION  (v22 — smarter heuristic)
# ─────────────────────────────────────────────────────────────────────────────
def auto_unit_factor(span: float):
    """
    Given the X-span of the drawing, return (unit_factor, unit_div).

    unit_factor : divide poly.area  by this to get m²
    unit_div    : divide poly.length by this to get m
    """
    if span > 50_000:          # micrometres
        return 1_000_000_000_000, 1_000_000
    elif span > 5_000:         # millimetres
        return 1_000_000, 1_000
    elif span > 500:           # centimetres
        return 10_000, 100
    else:                      # metres (span ≤ 500)
        return 1, 1


# ─────────────────────────────────────────────────────────────────────────────
#  MATRIX / LAYER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_m44(mat, x, y):
    v = mat.transform(Vec3(x, y, 0))
    return (v.x, v.y)


def effective_layer(entity, parent_layer):
    own = entity.dxf.get("layer", "0")
    return parent_layer if own == "0" else own


def build_matrix(ent):
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
#  HEURISTIC LAYER GUESSER
# ─────────────────────────────────────────────────────────────────────────────
def guess_wall_glass_layers(doc):
    msp    = doc.modelspace()
    counts = {}
    for ent in msp:
        if ent.dxftype() in GEOM_TYPES:
            lyr = ent.dxf.get("layer", "0")
            counts[lyr] = counts.get(lyr, 0) + 1
    for ent in msp:
        if ent.dxftype() == "INSERT":
            bname = ent.dxf.name
            if bname in doc.blocks:
                for sub in doc.blocks[bname]:
                    if sub.dxftype() in GEOM_TYPES:
                        lyr = sub.dxf.get("layer", "0")
                        counts[lyr] = counts.get(lyr, 0) + 1

    sorted_layers = sorted(counts, key=lambda l: counts[l], reverse=True)
    total_ents    = max(sum(counts.values()), 1)
    wall_layers   = set()
    glass_layers  = set()

    for lyr in sorted_layers:
        lo  = lyr.lower()
        pct = counts[lyr] / total_ents
        if any(kw in lo for kw in _SKIP_KW) and pct < 0.30:
            continue
        if any(kw in lo for kw in _GLASS_KW):
            glass_layers.add(lyr.upper()); continue
        if any(kw in lo for kw in _WALL_KW):
            wall_layers.add(lyr.upper());  continue
        if pct >= 0.10:
            wall_layers.add(lyr.upper())

    if not wall_layers:
        for lyr in sorted_layers[:5]:
            if not any(kw in lyr.lower() for kw in _SKIP_KW):
                wall_layers.add(lyr.upper())

    return wall_layers, glass_layers, sorted_layers, counts


# ─────────────────────────────────────────────────────────────────────────────
#  ENTITY → SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def ent_to_segments(entity, mat, layer, glass_layers_up):
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
            cx = entity.dxf.center.x; cy = entity.dxf.center.y
            r  = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea <= sa: ea += 2 * math.pi
            span = ea - sa
            if span < 1e-6: pass
            else:
                angles = np.linspace(sa, ea, 32)
                pts    = [apply_m44(mat, cx + r*math.cos(a), cy + r*math.sin(a)) for a in angles]
                if len(pts) >= 2:
                    for a, b in zip(pts[:-1], pts[1:]):
                        if a != b: res.append((LineString([a, b]), is_glass, True))
                    res.append((LineString([pts[0], pts[-1]]), False, True))

        elif t == "CIRCLE":
            cx = entity.dxf.center.x; cy = entity.dxf.center.y; r = entity.dxf.radius
            angles = np.linspace(0, 2*math.pi, 64)
            pts    = [apply_m44(mat, cx+r*math.cos(a), cy+r*math.sin(a)) for a in angles]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b: res.append((LineString([a, b]), is_glass, False))

        elif t == "ELLIPSE":
            try:
                cx=entity.dxf.center.x; cy=entity.dxf.center.y
                mx=entity.dxf.major_axis.x; my=entity.dxf.major_axis.y
                rat=entity.dxf.ratio; sa=entity.dxf.start_param; ea=entity.dxf.end_param
                if ea<=sa: ea+=2*math.pi
                a_len=math.hypot(mx,my); b_len=a_len*rat; angle_r=math.atan2(my,mx)
                angles=np.linspace(sa,ea,32); pts=[]
                for ang in angles:
                    ex=a_len*math.cos(ang); ey=b_len*math.sin(ang)
                    rx=ex*math.cos(angle_r)-ey*math.sin(angle_r)+cx
                    ry=ex*math.sin(angle_r)+ey*math.cos(angle_r)+cy
                    pts.append(apply_m44(mat,rx,ry))
                for a,b in zip(pts[:-1],pts[1:]):
                    if a!=b: res.append((LineString([a,b]),is_glass,False))
            except Exception: pass

        elif t == "SPLINE":
            raw_pts = []
            try:    raw_pts=[(p[0],p[1]) for p in entity.control_points]
            except: pass
            if len(raw_pts)<2:
                try:    raw_pts=[(p[0],p[1]) for p in entity.fit_points]
                except: pass
            pts=[apply_m44(mat,x,y) for x,y in raw_pts]
            for a,b in zip(pts[:-1],pts[1:]):
                if a!=b: res.append((LineString([a,b]),is_glass,False))

    except Exception:
        pass
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  MODE A EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_v8(layout, doc, allowed_up, glass_layers_up,
                   parent_mat=None, parent_layer="0", depth=0):
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
                                      glass_layers_up, combined, ins_layer, depth+1))
        elif et in GEOM_TYPES:
            eff = effective_layer(ent, parent_layer).upper()
            if eff not in allowed_up: continue
            out.extend(ent_to_segments(ent, parent_mat, eff, glass_layers_up))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ALL-LAYER EXTRACTOR (Mode C)
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_layers(layout, doc, skip_layers_up=None,
                       parent_mat=None, parent_layer="0", depth=0):
    if depth > 20: return []
    if parent_mat is None: parent_mat = Matrix44()
    if skip_layers_up is None: skip_layers_up = set()
    out = []
    for ent in layout:
        et = ent.dxftype()
        if et == "INSERT":
            bname = ent.dxf.name
            if bname not in doc.blocks: continue
            ins_layer = ent.dxf.get("layer", parent_layer)
            combined  = parent_mat @ build_matrix(ent)
            out.extend(extract_all_layers(doc.blocks[bname], doc, skip_layers_up,
                                          combined, ins_layer, depth+1))
        elif et in GEOM_TYPES:
            eff = effective_layer(ent, parent_layer).upper()
            if eff in skip_layers_up: continue
            segs = ent_to_segments(ent, parent_mat, eff, set())
            for (ls, _ig, ia) in segs:
                out.append((ls, eff, ia))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MODE B EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def process_entity_v11(entity, raw_lines, scale, exclude_arcs=False):
    if entity.dxftype() == "INSERT":
        try:
            for sub in entity.virtual_entities():
                process_entity_v11(sub, raw_lines, scale, exclude_arcs)
        except Exception: pass
        return
    if exclude_arcs and entity.dxftype() == "ARC": return
    try:
        p   = dxf_path.make_path(entity)
        pts = list(p.flattening(distance=0.1))
        for i in range(len(pts)-1):
            s=(round(pts[i].x*scale,1), round(pts[i].y*scale,1))
            e=(round(pts[i+1].x*scale,1), round(pts[i+1].y*scale,1))
            if s!=e: raw_lines.append([s,e])
    except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE QUALITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compactness(poly):
    if poly.length == 0: return 0.0
    return (4 * math.pi * poly.area) / (poly.length ** 2)

def aspect_ratio(poly):
    minx,miny,maxx,maxy = poly.bounds
    w=maxx-minx; h=maxy-miny
    if min(w,h)==0: return 999.0
    return max(w,h)/min(w,h)


def is_outer_envelope(poly, all_polys, centroid_threshold=0.35, area_fraction=0.55):
    """
    v22: A polygon is the outer envelope if EITHER:
    (a) it contains the centroid of >= centroid_threshold fraction of all others, OR
    (b) its area is >= area_fraction of the total bounding-box area of all polygons.

    This catches the building outline even when it doesn't fully "contain"
    the sub-room polygons (they share the same edges).
    """
    others = [p for p in all_polys if p is not poly]
    if not others:
        return False

    # (a) centroid containment
    n_inside = sum(1 for p in others
                   if poly.buffer(1e-3).contains(p.centroid))
    if n_inside / len(others) >= centroid_threshold:
        return True

    # (b) area dominance — compute overall bounding box of all polys
    all_xs = [c for p in all_polys for c in (p.bounds[0], p.bounds[2])]
    all_ys = [c for p in all_polys for c in (p.bounds[1], p.bounds[3])]
    if not all_xs: return False
    bbox_area = (max(all_xs)-min(all_xs)) * (max(all_ys)-min(all_ys))
    if bbox_area > 0 and poly.area / bbox_area >= area_fraction:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  GLASS EDGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def edge_glass_fraction(poly, glass_u, tol, mult):
    if glass_u is None or glass_u.is_empty: return 0.0
    coords=list(poly.exterior.coords); total_len=0.0; glass_len=0.0; cd=tol*mult
    for i in range(len(coords)-1):
        p1,p2=coords[i],coords[i+1]
        seg_len=math.hypot(p2[0]-p1[0],p2[1]-p1[1])
        mid=((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)
        qtr1=((p1[0]*3+p2[0])/4,(p1[1]*3+p2[1])/4)
        qtr2=((p1[0]+p2[0]*3)/4,(p1[1]+p2[1]*3)/4)
        total_len+=seg_len
        if any(glass_u.distance(Point(p))<=cd for p in (mid,qtr1,qtr2)):
            glass_len+=seg_len
    return 0.0 if total_len==0 else min(glass_len/total_len,1.0)

def has_any_glass_edge(poly, glass_u, tol, mult):
    if glass_u is None or glass_u.is_empty: return False
    coords=list(poly.exterior.coords); cd=tol*mult
    for i in range(len(coords)-1):
        p1,p2=coords[i],coords[i+1]
        mid=((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)
        qtr1=((p1[0]*3+p2[0])/4,(p1[1]*3+p2[1])/4)
        qtr2=((p1[0]+p2[0]*3)/4,(p1[1]+p2[1]*3)/4)
        if any(glass_u.distance(Point(p))<=cd for p in (mid,qtr1,qtr2)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  MODE D — CLOSED POLYLINE DETECTOR  ★ NEW in v22 ★
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_d(msp, doc, allowed_layers_up=None,
                         min_area_m2=1.0, max_area_m2=500.0,
                         unit_factor=1_000_000, unit_div=1000,
                         log=None):
    """
    Mode D: Extract rooms directly from closed LWPOLYLINE / POLYLINE entities.

    Strategy
    --------
    1. Walk every LWPOLYLINE / POLYLINE (including inside INSERTs).
    2. Accept if: entity.closed == True  OR  first_pt ≈ last_pt.
    3. Convert to Shapely Polygons, fix self-intersections with buffer(0).
    4. Filter by area (min_area_m2 .. max_area_m2).
    5. Remove outer-envelope polygon.
    6. Deduplicate near-identical polygons (> 90% overlap).

    Parameters
    ----------
    allowed_layers_up : set of uppercase layer names, or None = accept all layers
    unit_factor       : divide poly.area by this to get m²
    unit_div          : divide lengths by this to get metres

    Returns
    -------
    rooms  : list of dicts (same schema as Mode B)
    info   : dict {"total_closed", "accepted", "layers_seen"}
    """
    def _log(msg):
        if log: log(msg)

    def _pts_lwpoly(ent):
        return [(p[0], p[1]) for p in ent.get_points()]

    def _pts_poly(ent):
        pts = []
        for v in ent.vertices:
            try:    pts.append((v.dxf.location.x, v.dxf.location.y))
            except: pass
        return pts

    def _is_closed(ent, pts):
        try:
            if ent.dxftype() == "LWPOLYLINE" and ent.closed:
                return True
            if ent.dxftype() == "POLYLINE" and (ent.dxf.flags & 1):
                return True
        except Exception:
            pass
        if len(pts) >= 3:
            if math.hypot(pts[0][0]-pts[-1][0], pts[0][1]-pts[-1][1]) < 1e-3:
                return True
        return False

    def _walk(layout, depth=0):
        if depth > 10: return []
        found = []
        for ent in layout:
            et = ent.dxftype()
            if et == "INSERT":
                bname = ent.dxf.name
                if bname in doc.blocks:
                    found.extend(_walk(doc.blocks[bname], depth+1))
            elif et in ("LWPOLYLINE", "POLYLINE"):
                lyr_up = ent.dxf.get("layer", "0").upper()
                if allowed_layers_up and lyr_up not in allowed_layers_up:
                    continue
                try:
                    pts = _pts_lwpoly(ent) if et == "LWPOLYLINE" else _pts_poly(ent)
                except Exception:
                    continue
                if _is_closed(ent, pts):
                    found.append((pts, lyr_up))
        return found

    closed_raw = _walk(msp)
    _log(f"[Mode D] Closed polylines found: {len(closed_raw)}")
    layers_seen = list({lyr for _, lyr in closed_raw})

    # ── Build Shapely polygons ────────────────────────────────────────────
    polys = []
    for pts, lyr in closed_raw:
        if len(pts) < 3: continue
        if pts[0] == pts[-1]: pts = pts[:-1]
        try:
            poly = Polygon(pts)
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_empty: continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            polys.append((poly, lyr))
        except Exception:
            continue

    _log(f"[Mode D] Valid polygons: {len(polys)}")

    # ── Area filter ──────────────────────────────────────────────────────
    min_px = min_area_m2 * unit_factor
    max_px = max_area_m2 * unit_factor
    filtered = [(p, lyr) for p, lyr in polys if min_px <= p.area <= max_px]
    _log(f"[Mode D] After area filter ({min_area_m2}–{max_area_m2} m²): {len(filtered)}")

    if not filtered:
        return [], {"total_closed": len(closed_raw), "accepted": 0,
                    "layers_seen": layers_seen}

    # ── Remove outer envelope ────────────────────────────────────────────
    all_p = [p for p, _ in filtered]
    outer_ids = set()
    for poly in sorted(all_p, key=lambda x: x.area, reverse=True)[:5]:
        if is_outer_envelope(poly, all_p):
            outer_ids.add(id(poly))
            _log(f"  [Mode D] Outer envelope removed: {poly.area/unit_factor:.1f} m²")
            break

    filtered = [(p, lyr) for p, lyr in filtered if id(p) not in outer_ids]

    # ── Deduplicate ───────────────────────────────────────────────────────
    flags = [False] * len(filtered)
    for i in range(len(filtered)):
        if flags[i]: continue
        for j in range(i+1, len(filtered)):
            if flags[j]: continue
            pi, pj = filtered[i][0], filtered[j][0]
            try:
                inter   = pi.intersection(pj).area
                smaller = min(pi.area, pj.area)
                if inter / max(smaller, 1e-9) >= 0.90:
                    if pi.area >= pj.area: flags[j] = True
                    else:                  flags[i] = True
            except Exception: pass
    filtered = [(p, lyr) for (p, lyr), f in zip(filtered, flags) if not f]

    # ── Sort top-to-bottom, left-to-right ────────────────────────────────
    filtered.sort(key=lambda x: (-x[0].centroid.y, x[0].centroid.x))

    # ── Build output dicts ────────────────────────────────────────────────
    rooms = []
    for idx, (poly, lyr) in enumerate(filtered):
        minx, miny, maxx, maxy = poly.bounds
        rooms.append({
            "name":    f"Room {idx+1}",
            "room_id": f"R{idx+1}",
            "polygon": poly,
            "width":   round(maxx-minx, 2),
            "height":  round(maxy-miny, 2),
            "area":    round(poly.area, 2),
            "layer":   lyr,
            "objects_inside": [],
            "gf":      0.0,
            "is_glass": False,
        })

    _log(f"[Mode D] Final rooms: {len(rooms)}")
    return rooms, {
        "total_closed": len(closed_raw),
        "accepted":     len(rooms),
        "layers_seen":  layers_seen,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MODE A — GLASS-PARTITION DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_a(wall_segs, glass_segs, arc_segs,
                         snap_tol, bridge_tol,
                         glass_edge_thresh, glass_proximity_mult,
                         min_area_m2, max_area_m2,
                         min_compact, max_aspect,
                         unit_factor, log=None):
    def _log(msg):
        if log: log(msg)

    def node_snap_segs(segs, tol):
        out = []
        for ls in segs:
            try:
                coords=[(round(x/tol)*tol,round(y/tol)*tol) for x,y in ls.coords]
                dedup=[coords[0]]
                for c in coords[1:]:
                    if c!=dedup[-1]: dedup.append(c)
                if len(dedup)>=2: out.append(LineString(dedup))
            except Exception: pass
        return out

    def bridge_gaps(lines, tol):
        from collections import defaultdict
        ep=defaultdict(int)
        for ls in lines:
            cs=list(ls.coords); ep[cs[0]]+=1; ep[cs[-1]]+=1
        dangling=[pt for pt,cnt in ep.items() if cnt==1]
        if not dangling: return []
        bridges,used=[],set(); arr=np.array(dangling)
        for i,pt in enumerate(dangling):
            if i in used: continue
            diffs=arr-np.array(pt); dists=np.hypot(diffs[:,0],diffs[:,1]); dists[i]=np.inf
            j=int(np.argmin(dists))
            if dists[j]<=tol and j not in used:
                bridges.append(LineString([pt,dangling[j]])); used.add(i); used.add(j)
        return bridges

    wall_sn=node_snap_segs(wall_segs,snap_tol)
    glass_sn=node_snap_segs(glass_segs,snap_tol)
    arc_sn=node_snap_segs(arc_segs,snap_tol)
    boundary_snapped=wall_sn+glass_sn+arc_sn
    bridges=bridge_gaps(boundary_snapped,bridge_tol)
    lines_for_poly=boundary_snapped+bridges
    _log(f"[Mode A] Gap bridges: {len(bridges)}")

    merged=unary_union(lines_for_poly); all_poly=list(polygonize(merged))
    all_poly.sort(key=lambda p: p.area, reverse=True)
    _log(f"[Mode A] Raw polygons: {len(all_poly)}")

    glass_union=unary_union(glass_sn) if glass_sn else None

    outer_ids=set()
    for poly in all_poly[:5]:
        if is_outer_envelope(poly, all_poly):
            outer_ids.add(id(poly))
            _log(f"  Outer envelope excluded: {poly.area/unit_factor:.1f} m²")
            break

    candidates=[]
    for poly in all_poly:
        if id(poly) in outer_ids: continue
        area_m2=poly.area/unit_factor
        if area_m2<min_area_m2 or area_m2>max_area_m2: continue
        if compactness(poly)<min_compact: continue
        if aspect_ratio(poly)>max_aspect: continue
        gf=edge_glass_fraction(poly,glass_union,snap_tol,glass_proximity_mult)
        has_gls=has_any_glass_edge(poly,glass_union,snap_tol,glass_proximity_mult)
        candidates.append((poly,gf,has_gls))

    _log(f"[Mode A] Candidates after shape filter: {len(candidates)}")
    candidates.sort(key=lambda x: x[0].area,reverse=True)

    def is_mostly_inside(small,large,tol=0.90):
        try: return small.intersection(large).area/small.area>=tol
        except: return False

    accepted=[]
    for (poly,gf,has_gls) in candidates:
        parent=next(((ap,ag,ai) for ap,ag,ai in accepted
                     if is_mostly_inside(poly,ap)),None)
        if parent is None: accepted.append((poly,gf,gf>=glass_edge_thresh))
        else:
            if has_gls: accepted.append((poly,gf,True))

    flags=[False]*len(accepted)
    for i in range(len(accepted)):
        if flags[i]: continue
        for j in range(i+1,len(accepted)):
            if flags[j]: continue
            ap,bp=accepted[i][0],accepted[j][0]
            if abs(ap.area-bp.area)/max(ap.area,bp.area)>0.06: continue
            try:
                if ap.intersection(bp).area/min(ap.area,bp.area)>=0.90: flags[j]=True
            except: pass
    accepted=[r for r,f in zip(accepted,flags) if not f]
    accepted.sort(key=lambda r: (-r[0].centroid.y,r[0].centroid.x))

    n_gl=sum(1 for _,_,ig in accepted if ig)
    _log(f"[Mode A] Final: {len(accepted)} rooms ({len(accepted)-n_gl} wall + {n_gl} glass)")
    return accepted,bridges,wall_sn,glass_sn


# ─────────────────────────────────────────────────────────────────────────────
#  MODE B — ENDPOINT-BRIDGING DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_b(raw_wall_lines, extracted_objects,
                         gap_close_tol, max_door_width, min_wall_len,
                         min_area_m2, max_area_m2, unit_factor,
                         outer_area_pct=25.0, exclude_stairs=True,
                         stair_parallel_min=4, stair_angle_tol=8.0,
                         max_stair_area_m2=20.0, min_solidity=0.50,
                         max_aspect_ratio=15.0, max_interior_walls=8,
                         min_closet_area_m2=0.3, log=None):
    def _log(msg):
        if log: log(msg)

    if not raw_wall_lines: return [],[]
    shapely_walls=[LineString(l) for l in raw_wall_lines
                   if LineString(l).length>=min_wall_len]
    if not shapely_walls: return [],[]

    merged_walls=unary_union(shapely_walls)
    if merged_walls.geom_type=="MultiLineString":   lines_list=list(merged_walls.geoms)
    elif merged_walls.geom_type=="LineString":       lines_list=[merged_walls]
    else: lines_list=[g for g in merged_walls.geoms if g.geom_type=="LineString"]

    valid_endpoints=[]
    for line in lines_list:
        if line.length>min_wall_len:
            valid_endpoints.append(Point(line.coords[0]))
            valid_endpoints.append(Point(line.coords[-1]))

    bridges=[]
    for i,ep1 in enumerate(valid_endpoints):
        for j,ep2 in enumerate(valid_endpoints):
            if i<j and ep1.distance(ep2)<=gap_close_tol:
                bridges.append(LineString([ep1,ep2]))
    for i,ep1 in enumerate(valid_endpoints):
        for j,ep2 in enumerate(valid_endpoints):
            if i<j:
                dist=ep1.distance(ep2)
                if gap_close_tol<dist<=max_door_width:
                    dx=abs(ep1.x-ep2.x); dy=abs(ep1.y-ep2.y)
                    if dx<150 or dy<150:
                        br=LineString([ep1,ep2])
                        if not br.crosses(merged_walls): bridges.append(br)

    noded=unary_union(lines_list+bridges); raw_polys=list(polygonize(noded))
    raw_polys.sort(key=lambda p: p.area,reverse=True)
    _log(f"[Mode B] Raw polygons: {len(raw_polys)}")
    if not raw_polys: return [],[]

    mnx,mny,mxx,mxy=noded.bounds
    total_bbox=(mxx-mnx)*(mxy-mny)
    outer_thresh=total_bbox*(outer_area_pct/100.0)
    min_area_px=min_area_m2*unit_factor; max_area_px=max_area_m2*unit_factor
    min_closet_px=min_closet_area_m2*unit_factor
    max_stair_px=max_stair_area_m2*unit_factor

    def is_staircase(poly,wsegs,min_p,atol):
        try:
            angles=[]
            for seg in wsegs:
                mid=Point((seg.coords[0][0]+seg.coords[-1][0])/2,
                           (seg.coords[0][1]+seg.coords[-1][1])/2)
                if poly.contains(mid):
                    dx=seg.coords[-1][0]-seg.coords[0][0]
                    dy=seg.coords[-1][1]-seg.coords[0][1]
                    angles.append(math.degrees(math.atan2(dy,dx))%180)
            if len(angles)<min_p: return False
            angles.sort()
            for ref in angles:
                count=sum(1 for a in angles
                          if abs(a-ref)<=atol or abs(a-ref)>=(180-atol))
                if count>=min_p: return True
        except: pass
        return False

    rooms_data=[]; wall_cavities=[]
    all_p_list=[p for p in raw_polys]

    for poly in raw_polys:
        area=poly.area
        if area>=outer_thresh: continue
        # v22: use improved outer envelope check
        if is_outer_envelope(poly, all_p_list): continue

        if area<min_area_px or area>max_area_px:
            if min_closet_px<=area<min_area_px:
                buf=poly.buffer(50)
                for obj in extracted_objects:
                    if buf.covers(obj["point"]):
                        rooms_data.append({
                            "width":round(poly.bounds[2]-poly.bounds[0],2),
                            "height":round(poly.bounds[3]-poly.bounds[1],2),
                            "area":round(area,2),"polygon":poly,"objects_inside":[]})
                        break
            elif 10_000<area<min_closet_px:
                wall_cavities.append(poly)
            continue

        try:
            hull=poly.convex_hull; solid=area/hull.area if hull.area>0 else 0
        except: solid=1.0
        if solid<min_solidity: continue

        rmx,rmy,rMx,rMy=poly.bounds; w=rMx-rmx; h=rMy-rmy
        ar=max(w,h)/max(min(w,h),1)
        if ar>max_aspect_ratio: continue

        if exclude_stairs and area<=max_stair_px:
            if is_staircase(poly,shapely_walls,stair_parallel_min,stair_angle_tol):
                continue

        piercing=sum(1 for seg in shapely_walls
                     if poly.contains(Point(seg.coords[0])) and
                        poly.contains(Point(seg.coords[-1])))
        if piercing>max_interior_walls: continue

        rooms_data.append({"width":round(w,2),"height":round(h,2),
                            "area":round(area,2),"polygon":poly,"objects_inside":[]})

    clean=[]
    for i,room in enumerate(rooms_data):
        bad=False
        for j,other in enumerate(rooms_data):
            if i==j: continue
            try:
                if other["polygon"].contains(room["polygon"].representative_point()):
                    bad=True; break
                inter=room["polygon"].intersection(other["polygon"])
                if inter.area>0.80*room["area"] and room["area"]<other["area"]:
                    bad=True; break
            except: pass
        if not bad: clean.append(room)

    clean.sort(key=lambda x: x["area"],reverse=True)
    for idx,room in enumerate(clean):
        room["name"]=f"Room {idx+1}"; room["room_id"]=f"R{idx+1}"
    return clean, wall_cavities


# ─────────────────────────────────────────────────────────────────────────────
#  MODE C — ALL-LAYER FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def detect_rooms_mode_c(all_layer_segs,
                         snap_tol, bridge_tol,
                         min_area_m2, max_area_m2,
                         min_compact, max_aspect,
                         unit_factor, log=None):
    def _log(msg):
        if log: log(msg)

    def node_snap(segs,tol):
        out=[]
        for ls in segs:
            try:
                coords=[(round(x/tol)*tol,round(y/tol)*tol) for x,y in ls.coords]
                dedup=[coords[0]]
                for c in coords[1:]:
                    if c!=dedup[-1]: dedup.append(c)
                if len(dedup)>=2: out.append(LineString(dedup))
            except: pass
        return out

    def bridge_gaps(lines,tol):
        from collections import defaultdict
        ep=defaultdict(int)
        for ls in lines:
            cs=list(ls.coords); ep[cs[0]]+=1; ep[cs[-1]]+=1
        dangling=[pt for pt,cnt in ep.items() if cnt==1]
        if not dangling: return []
        bridges,used=[],set(); arr=np.array(dangling)
        for i,pt in enumerate(dangling):
            if i in used: continue
            diffs=arr-np.array(pt); dists=np.hypot(diffs[:,0],diffs[:,1]); dists[i]=np.inf
            j=int(np.argmin(dists))
            if dists[j]<=tol and j not in used:
                bridges.append(LineString([pt,dangling[j]])); used.add(i); used.add(j)
        return bridges

    raw_segs=[ls for (ls,_lyr,_ia) in all_layer_segs]
    _log(f"[Mode C] Total segments: {len(raw_segs)}")

    all_sn=node_snap(raw_segs,snap_tol)
    bridges=bridge_gaps(all_sn,bridge_tol)
    merged=unary_union(all_sn+bridges)
    all_poly=list(polygonize(merged))
    all_poly.sort(key=lambda p: p.area,reverse=True)
    _log(f"[Mode C] Raw polygons: {len(all_poly)}")

    outer_ids=set()
    for poly in all_poly[:5]:
        if is_outer_envelope(poly, all_poly):
            outer_ids.add(id(poly))
            break

    accepted=[]
    for poly in all_poly:
        if id(poly) in outer_ids: continue
        area_m2=poly.area/unit_factor
        if area_m2<min_area_m2 or area_m2>max_area_m2: continue
        if compactness(poly)<min_compact: continue
        if aspect_ratio(poly)>max_aspect: continue
        accepted.append((poly,0.0,False))

    flags=[False]*len(accepted)
    for i in range(len(accepted)):
        if flags[i]: continue
        for j in range(i+1,len(accepted)):
            if flags[j]: continue
            ap,bp=accepted[i][0],accepted[j][0]
            if abs(ap.area-bp.area)/max(ap.area,bp.area)>0.06: continue
            try:
                if ap.intersection(bp).area/min(ap.area,bp.area)>=0.90: flags[j]=True
            except: pass
    accepted=[r for r,f in zip(accepted,flags) if not f]
    accepted.sort(key=lambda r: (-r[0].centroid.y,r[0].centroid.x))

    _log(f"[Mode C] Final rooms: {len(accepted)}")
    return accepted,bridges,all_sn


# ─────────────────────────────────────────────────────────────────────────────
#  FURNITURE → OBJECT CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
def process_furniture_to_objects(furn_lines, gap_tol=50):
    if not furn_lines: return []
    from shapely.strtree import STRtree
    shapely_lines=[LineString(l) for l in furn_lines]
    buffered=[ls.buffer(gap_tol) for ls in shapely_lines]
    tree=STRtree(buffered)
    G=nx.Graph(); G.add_nodes_from(range(len(furn_lines)))
    for i,poly in enumerate(buffered):
        for j in tree.query(poly):
            if i!=j and poly.intersects(buffered[j]): G.add_edge(i,j)
    objects_data=[]
    for comp in nx.connected_components(G):
        lines=[furn_lines[idx] for idx in comp]
        mls=MultiLineString([LineString(l) for l in lines])
        minx,miny,maxx,maxy=mls.bounds
        objects_data.append({
            "object_id":f"Obj {len(objects_data)+1}",
            "length":round(maxx-minx,2),"width":round(maxy-miny,2),
            "center_x":round(mls.centroid.x,2),"center_y":round(mls.centroid.y,2),
            "point":mls.centroid,
        })
    return objects_data