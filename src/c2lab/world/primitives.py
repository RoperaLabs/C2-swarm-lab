"""Compiled geometry kernels. Primitives are packed as rows of P (n, 17):

    P[:, 1:4]   centre c (capsule: end a)
    P[:, 4:13]  world -> local rotation, row-major (capsule: unused)
    P[:, 13:17] box hx hy hz · cylinder r hh · sphere r · cone r h · torus R r sin(arc) cos(arc) · capsule bx by bz r

Exact SDFs after Inigo Quilez (iquilezles.org/articles/distfunctions). Rays and segments are sphere-traced on them;
the terrain uses the Lipschitz bound (z - h) / sqrt(1 + L^2), so tracing it stays conservative.
"""
import math

import numpy as np
from numba import njit, prange

BOX, CYLINDER, CAPSULE, SPHERE, CONE, TORUS = range(6)
TYPES = {"box": BOX, "cylinder": CYLINDER, "capsule": CAPSULE, "sphere": SPHERE, "cone": CONE, "torus": TORUS}
TERRAIN = -2
EPS = 1e-6          # m, a trace stops this close to a surface
MAX_IT = 20000      # trace steps (grazing rays can need thousands)


@njit(cache=True)
def prim_sdf(t, row, x, y, z):
    if t == CAPSULE:
        pax, pay, paz = x - row[1], y - row[2], z - row[3]
        bax, bay, baz = row[13] - row[1], row[14] - row[2], row[15] - row[3]
        h = min(max((pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz), 0.0), 1.0)
        return math.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2 + (paz - baz * h) ** 2) - row[16]
    dx, dy, dz = x - row[1], y - row[2], z - row[3]
    px = row[4] * dx + row[5] * dy + row[6] * dz
    py = row[7] * dx + row[8] * dy + row[9] * dz
    pz = row[10] * dx + row[11] * dy + row[12] * dz
    if t == BOX:
        qx, qy, qz = abs(px) - row[13], abs(py) - row[14], abs(pz) - row[15]
        return (math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
                + min(max(qx, max(qy, qz)), 0.0))
    if t == SPHERE:
        return math.sqrt(px * px + py * py + pz * pz) - row[13]
    if t == CYLINDER:
        ax, az = math.sqrt(px * px + py * py) - row[13], abs(pz) - row[14]
        return min(max(ax, az), 0.0) + math.sqrt(max(ax, 0.0) ** 2 + max(az, 0.0) ** 2)
    if t == CONE:  # capped cone, base radius r1 at z = 0, apex (r2 = 0) at z = h
        r1, hh = row[13], 0.5 * row[14]
        qx, qy = math.sqrt(px * px + py * py), pz - hh
        cax, cay = qx - min(qx, r1 if qy < 0.0 else 0.0), abs(qy) - hh
        k1x, k1y, k2x, k2y = 0.0, hh, -r1, 2.0 * hh
        s = min(max(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0.0), 1.0)
        cbx, cby = qx - k1x + k2x * s, qy - k1y + k2y * s
        sign = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
        return sign * math.sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby))
    # torus: ring in the local xy plane, kept arc of half-angle `arc` centred on +y
    sa, ca = row[15], row[16]
    ax = abs(px)
    k = ax * sa + py * ca if ca * ax > sa * py else math.sqrt(ax * ax + py * py)
    return math.sqrt(max(px * px + py * py + pz * pz + row[13] ** 2 - 2.0 * row[13] * k, 0.0)) - row[14]


@njit(cache=True)
def ground_h(H, g, x, y):
    """Bilinear terrain height; g = (x0, y0, cell_x, cell_y, 1/sqrt(1+L^2), 1 if every height is 0)."""
    nx, ny = H.shape
    fx = min(max((x - g[0]) / g[2], 0.0), nx - 1.0)
    fy = min(max((y - g[1]) / g[3], 0.0), ny - 1.0)
    i, j = min(int(fx), nx - 2), min(int(fy), ny - 2)
    u, v = fx - i, fy - j
    return ((1 - u) * (1 - v) * H[i, j] + u * (1 - v) * H[i + 1, j]
            + (1 - u) * v * H[i, j + 1] + u * v * H[i + 1, j + 1])


@njit(cache=True)
def ground_points(H, g, xy, out):
    for k in range(len(xy)):
        out[k] = ground_h(H, g, xy[k, 0], xy[k, 1])


@njit(cache=True)
def terr_sdf(H, g, x, y, z):
    if g[5] != 0.0:     # flat terrain at height 0: the bilinear height is exactly 0 and g[4] is 1, so this is exact
        return z
    return (z - ground_h(H, g, x, y)) * g[4]


@njit(cache=True)
def _overlap(ab, lo0, lo1, lo2, hi0, hi1, hi2):
    return not (ab[3] < lo0 or ab[0] > hi0 or ab[4] < lo1 or ab[1] > hi1 or ab[5] < lo2 or ab[2] > hi2)


@njit(cache=True)
def _aabb_dist(ab, x, y, z):
    dx = max(ab[0] - x, 0.0, x - ab[3])
    dy = max(ab[1] - y, 0.0, y - ab[4])
    dz = max(ab[2] - z, 0.0, z - ab[5])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@njit(parallel=True, cache=True)
def sdf_points(T, P, AB, H, g, terrain, grid0, gsize, gdim, cstart, items, pts, max_dist, out):
    n = len(T)
    for k in prange(len(pts)):
        x, y, z = pts[k, 0], pts[k, 1], pts[k, 2]
        best = max_dist
        if terrain:
            best = min(best, terr_sdf(H, g, x, y, z))
        if math.isinf(max_dist):
            for i in range(n):
                if _aabb_dist(AB[i], x, y, z) < best:
                    best = min(best, prim_sdf(T[i], P[i], x, y, z))
        else:
            lo = np.empty(3, np.int64)
            hi = np.empty(3, np.int64)
            for a in range(3):
                lo[a] = min(max(int(math.floor((pts[k, a] - max_dist - grid0[a]) / gsize)), 0), gdim[a] - 1)
                hi[a] = min(max(int(math.floor((pts[k, a] + max_dist - grid0[a]) / gsize)), 0), gdim[a] - 1)
            for cx in range(lo[0], hi[0] + 1):
                for cy in range(lo[1], hi[1] + 1):
                    for cz in range(lo[2], hi[2] + 1):
                        c = (cx * gdim[1] + cy) * gdim[2] + cz
                        for m in range(cstart[c], cstart[c + 1]):
                            i = items[m]
                            if _aabb_dist(AB[i], x, y, z) < best:
                                best = min(best, prim_sdf(T[i], P[i], x, y, z))
        out[k] = best


@njit(cache=True)
def _candidates(AB, ox, oy, oz, ex, ey, ez, cand):
    nc = 0
    for i in range(len(AB)):
        if _overlap(AB[i], min(ox, ex), min(oy, ey), min(oz, ez), max(ox, ex), max(oy, ey), max(oz, ez)):
            cand[nc] = i
            nc += 1
    return nc


@njit(parallel=True, cache=True)
def raycast(T, P, AB, H, g, O, D, R, dist, prim, normal):
    n = len(T)
    for k in prange(len(O)):
        ox, oy, oz, dx, dy, dz, rng = O[k, 0], O[k, 1], O[k, 2], D[k, 0], D[k, 1], D[k, 2], R[k]
        cand = np.empty(n, np.int64)
        nc = _candidates(AB, ox, oy, oz, ox + dx * rng, oy + dy * rng, oz + dz * rng, cand)
        dist[k], prim[k] = np.inf, -1
        normal[k, 0] = normal[k, 1] = normal[k, 2] = 0.0
        t = 0.0
        for it in range(MAX_IT):
            x, y, z = ox + dx * t, oy + dy * t, oz + dz * t
            best, bi = terr_sdf(H, g, x, y, z), TERRAIN
            for j in range(nc):
                s = prim_sdf(T[cand[j]], P[cand[j]], x, y, z)
                if s < best:
                    best, bi = s, cand[j]
            if it == 0 and best < 0.0:
                dist[k], prim[k] = 0.0, bi
                break
            if best < EPS:
                dist[k], prim[k] = t, bi
                h = 1e-6
                for a in range(3):
                    e = np.zeros(3)
                    e[a] = h
                    if bi == TERRAIN:
                        normal[k, a] = (terr_sdf(H, g, x + e[0], y + e[1], z + e[2])
                                        - terr_sdf(H, g, x - e[0], y - e[1], z - e[2]))
                    else:
                        normal[k, a] = (prim_sdf(T[bi], P[bi], x + e[0], y + e[1], z + e[2])
                                        - prim_sdf(T[bi], P[bi], x - e[0], y - e[1], z - e[2]))
                nn = math.sqrt(normal[k, 0] ** 2 + normal[k, 1] ** 2 + normal[k, 2] ** 2)
                for a in range(3):
                    normal[k, a] /= nn
                break
            t += best
            if t > rng:
                break


@njit(parallel=True, cache=True)
def crossings(T, P, AB, H, g, A, B, out):
    n = len(T)
    for k in prange(len(A)):
        ax, ay, az = A[k, 0], A[k, 1], A[k, 2]
        L = max(math.sqrt((B[k, 0] - ax) ** 2 + (B[k, 1] - ay) ** 2 + (B[k, 2] - az) ** 2), 1e-12)
        dx, dy, dz = (B[k, 0] - ax) / L, (B[k, 1] - ay) / L, (B[k, 2] - az) / L
        cand = np.empty(n, np.int64)
        nc = _candidates(AB, ax, ay, az, B[k, 0], B[k, 1], B[k, 2], cand)
        c = 0
        for j in range(nc + 1):
            t = 0.0
            for it in range(MAX_IT):
                x, y, z = ax + dx * t, ay + dy * t, az + dz * t
                s = terr_sdf(H, g, x, y, z) if j == nc else prim_sdf(T[cand[j]], P[cand[j]], x, y, z)
                if s < EPS:
                    c += 1
                    break
                t += s
                if t > L:
                    break
        out[k] = c


@njit(parallel=True, cache=True)
def cast(T, P, AB, H, g, O, D, K, rng, dist):
    """First-hit distance of sensor rays grouped by origin: ray k starts at O[k // K] along the unit D[k] and goes up to
    rng; inf for no hit. The trace of `raycast` bit for bit, without normals and with the candidate primitives
    pre-filtered once per origin by the box that holds all of its rays."""
    n = len(T)
    m = rng * (1.0 + 1e-9) + 1e-9
    for i in prange(len(O)):
        pre, cand = np.empty(n, np.int64), np.empty(n, np.int64)
        ox, oy, oz = O[i, 0], O[i, 1], O[i, 2]
        npre = 0
        for q in range(n):
            if _overlap(AB[q], ox - m, oy - m, oz - m, ox + m, oy + m, oz + m):
                pre[npre] = q
                npre += 1
        for k in range(i * K, (i + 1) * K):
            dx, dy, dz = D[k, 0], D[k, 1], D[k, 2]
            ex, ey, ez = ox + dx * rng, oy + dy * rng, oz + dz * rng
            nc = 0
            for q in range(npre):
                if _overlap(AB[pre[q]], min(ox, ex), min(oy, ey), min(oz, ez), max(ox, ex), max(oy, ey), max(oz, ez)):
                    cand[nc] = pre[q]
                    nc += 1
            dist[k] = np.inf
            t = 0.0
            for it in range(MAX_IT):
                x, y, z = ox + dx * t, oy + dy * t, oz + dz * t
                best = terr_sdf(H, g, x, y, z)
                for j in range(nc):
                    s = prim_sdf(T[cand[j]], P[cand[j]], x, y, z)
                    if s < best:
                        best = s
                if it == 0 and best < 0.0:
                    dist[k] = 0.0
                    break
                if best < EPS:
                    dist[k] = t
                    break
                t += best
                if t > rng:
                    break


@njit(parallel=True, cache=True)
def sphere_hits(O, D, K, owner0, C, alive, r, cap, dist):
    """Lower dist[k] to the first hit of ray k on any live sphere (centre C[j], radius r) other than its owner. Rays are
    grouped by origin: ray k starts at O[k // K] and belongs to sphere owner0 + k // K. Spheres farther than cap + r
    from the origin are skipped, since no ray can hit them within cap (cap = inf tests them all)."""
    reach = (cap + r + 1e-3) ** 2
    rr = r * r
    for i in prange(len(O)):
        for j in range(len(C)):
            if j == owner0 + i or not alive[j]:
                continue
            ox, oy, oz = C[j, 0] - O[i, 0], C[j, 1] - O[i, 1], C[j, 2] - O[i, 2]
            cc = ox * ox + oy * oy + oz * oz
            if cc > reach:
                continue
            for k in range(i * K, (i + 1) * K):
                b = D[k, 0] * ox + D[k, 1] * oy + D[k, 2] * oz
                disc = b * b - cc + rr
                if disc >= 0.0:
                    t = b - math.sqrt(disc)
                    if 0.0 <= t < dist[k]:
                        dist[k] = t
