"""
Run this script ONCE from your hvac folder:
    python fix_duct_engine.py

It patches duct_engine.py in-place to fix the KeyError: 'polygon' bug.
"""
import re, sys, os

path = os.path.join(os.path.dirname(__file__), "duct_engine.py")

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

original = src  # keep copy for diff

# ── Fix 1: Add _get_poly helper right before place_ahu if not already present ──
if "_get_poly" not in src:
    src = src.replace(
        "def place_ahu(",
        'def _get_poly(room):\n'
        '    """Return shapely Polygon — works whether key is _poly or polygon."""\n'
        '    return room.get("_poly") or room.get("polygon")\n'
        '\n\n'
        'def place_ahu('
    )
    print("✅ Fix 1 applied: added _get_poly()")
else:
    print("ℹ️  Fix 1 skipped: _get_poly already present")

# ── Fix 2: place_ahu — replace r["polygon"] with _get_poly(r) ──
src = src.replace(
    'xs = [r["polygon"].centroid.x for r in rooms]',
    'xs = [_get_poly(r).centroid.x for r in rooms]'
)
src = src.replace(
    'ys = [r["polygon"].centroid.y for r in rooms]',
    'ys = [_get_poly(r).centroid.y for r in rooms]'
)

# ── Fix 3: route_ducts line — the main crash ──
src = src.replace(
    'poly    = room["polygon"]',
    'poly    = _get_poly(room)'
)
# also handle without extra spaces
src = src.replace(
    'poly = room["polygon"]',
    'poly = _get_poly(room)'
)
print("✅ Fix 3 applied: route_ducts poly lookup")

# ── Fix 4: render_duct_floorplan room loop ──
src = src.replace(
    'poly = row["_poly"]\n        xs, ys = poly.exterior.xy',
    'poly = _get_poly(row)\n        xs, ys = poly.exterior.xy'
)
src = src.replace(
    'poly = row["polygon"]\n        xs, ys = poly.exterior.xy',
    'poly = _get_poly(row)\n        xs, ys = poly.exterior.xy'
)

# ── Fix 5: diffuser lookup inside render ──
src = src.replace(
    'poly = row["_poly"]\n                break',
    'poly = _get_poly(row)\n                break'
)
src = src.replace(
    'poly = row["polygon"]\n                break',
    'poly = _get_poly(row)\n                break'
)

# ── Fix 6: AHU size bounds calculation ──
src = src.replace(
    '(max(r["_poly"].bounds[2] for r in room_data) -\n         min(r["_poly"].bounds[0] for r in room_data))',
    '(max(_get_poly(r).bounds[2] for r in room_data) -\n         min(_get_poly(r).bounds[0] for r in room_data))'
)
src = src.replace(
    '(max(r["polygon"].bounds[2] for r in room_data) -\n         min(r["polygon"].bounds[0] for r in room_data))',
    '(max(_get_poly(r).bounds[2] for r in room_data) -\n         min(_get_poly(r).bounds[0] for r in room_data))'
)

if src == original:
    print("⚠️  No changes made — file may already be patched or pattern not found.")
    print("    Checking for remaining room[\"polygon\"] references:")
    for i, line in enumerate(src.splitlines(), 1):
        if 'room["polygon"]' in line or 'row["polygon"]' in line:
            print(f"    Line {i}: {line.strip()}")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\n✅ duct_engine.py patched successfully: {path}")

# ── Verify no raw polygon key access remains ──
remaining = [
    (i+1, l.strip())
    for i, l in enumerate(src.splitlines())
    if re.search(r'room\["polygon"\]|row\["polygon"\]|r\["polygon"\]', l)
]
if remaining:
    print("\n⚠️  Still found raw [\"polygon\"] accesses:")
    for lineno, line in remaining:
        print(f"  Line {lineno}: {line}")
else:
    print('✅ Verified: no raw ["polygon"] key access remaining.')