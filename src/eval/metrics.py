import math

def dedupe_points(points, decimals=6):
    seen, out = set(), []
    for f1, f2 in points:
        key = (round(f1, decimals), round(f2, decimals))
        if key not in seen:
            seen.add(key); out.append((f1, f2))
    return out

def hypervolume_2d(points, ref):
    # Minimization HV in 2D; ref must be worse than all points (bigger f1,f2)
    pts = sorted(dedupe_points(points), key=lambda p: (p[0], p[1]))
    skyline = []
    best_f2 = float("inf")
    for f1, f2 in pts:
        if f2 < best_f2:
            skyline.append((f1, f2))
            best_f2 = f2
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in skyline:
        hv += max(0.0, (ref[0] - f1)) * max(0.0, (prev_f2 - f2))
        prev_f2 = min(prev_f2, f2)
    return hv

def euclid(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.hypot(dx, dy)

def igd_2d(approx_points, ref_points):
    # IGD: avg distance from each ref point to its nearest approx point
    if not approx_points or not ref_points: 
        return float("inf")
    A = dedupe_points(approx_points)
    R = dedupe_points(ref_points)
    total = 0.0
    for r in R:
        total += min(euclid(r, a) for a in A)
    return total / len(R)

def coverage_C(A, B):
    # Fraction of B that is weakly dominated by A (b is worse or equal in both)
    def weakly_dominated(a, b):
        return (a[0] <= b[0]) and (a[1] <= b[1])
    dominated = 0
    for b in B:
        if any(weakly_dominated(a, b) for a in A):
            dominated += 1
    return dominated / max(1, len(B))
