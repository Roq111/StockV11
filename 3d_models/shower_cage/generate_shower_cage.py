"""
Shower cage head generator -> STL.

Geometry (all units in mm):
  * Male thread G1/2" (BSPP, standard shower hose): major Ø20.955, 14 TPI (pitch 1.814), 55° profile
  * Body section 1: Ø60 for 120 mm (right under the thread)
  * Body section 2: Ø110 for 70 mm
  * Lattice of HOLLOW tubular struts (two opposite helix families -> diamond pattern)
    fed from a hub chamber under the thread and closed by a hollow bottom ring.
  * Many Ø1.0 mm high-pressure jet holes drilled on the inner side of every strut and
    the ring, all pointing into the cage.

Run:  pip install manifold3d trimesh numpy && python generate_shower_cage.py
"""
import math
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, OpType

# ---------------- parameters ----------------
THREAD_MAJOR_D = 20.955
THREAD_PITCH = 25.4 / 14
THREAD_LEN = 12.0
BORE_D = 10.0                 # water inlet bore through the thread

D1, L1 = 60.0, 120.0          # first section: diameter, length
D2, L2 = 110.0, 70.0          # second section: diameter, length
BODY_LEN = L1 + L2

TUBE_R = 4.0                  # strut outer radius (Ø8)
CHANNEL_R = 2.4               # inner water channel radius (wall 1.6)
HUB_R, HUB_LEN = 13.0, 12.0   # solid hub under the thread
HUB_CHAMBER_R = 9.0

N_STRUTS = 6                  # per helix direction (12 total)
TWIST = math.radians(150)     # rotation of each strut over the body length

JET_D = 0.8                  # total jet area kept below the inlet bore area -> high pressure
JET_SPACING = 9.0             # mm along each strut
JET_CLEAR = 7.5               # skip jets this close to a crossing strut

SPHERE_SEGS = 20
OUT = "shower_cage.stl"

R1 = D1 / 2 - TUBE_R          # strut centre-line radii
R2 = D2 / 2 - TUBE_R


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def radius_at(t):
    """Centre-line radius at depth t (0 = top of body under thread, BODY_LEN = bottom)."""
    flare = 8 + (R1 - 8) * smoothstep(HUB_LEN - 4, HUB_LEN + 22, t)
    return flare + (R2 - R1) * smoothstep(L1 - 8, L1 + 18, t)


def strut_path(phi0, direction, n=90):
    ts = np.linspace(HUB_LEN - 4, BODY_LEN, n)
    pts = []
    for t in ts:
        r = radius_at(t)
        # twist mostly happens in the lower part, like the picture (straight-ish neck)
        phi = phi0 + direction * TWIST * smoothstep(HUB_LEN, BODY_LEN, t) ** 1.2
        pts.append((r * math.cos(phi), r * math.sin(phi), BODY_LEN - t))
    return np.array(pts)


def sphere(c, r):
    return Manifold.sphere(r, SPHERE_SEGS).translate(tuple(c))


def tube(path, r):
    segs = [Manifold.batch_hull([sphere(a, r), sphere(b, r)]) for a, b in zip(path[:-1], path[1:])]
    return union(segs)


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add)


def ring_path(z, r, n=96):
    a = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], 1)


def threaded_rod():
    """G1/2 male thread built directly as a radial-surface mesh (55° Whitworth-like profile)."""
    h = 0.640327 * THREAD_PITCH
    r_maj = THREAD_MAJOR_D / 2
    r_min = r_maj - h
    nth, nz = 144, int(THREAD_LEN / 0.12) + 1
    th = np.linspace(0, 2 * math.pi, nth, endpoint=False)
    zs = np.linspace(0, THREAD_LEN, nz)

    def prof(u):  # u in [0,1) along one pitch -> radius
        d = np.abs(u - 0.5) * 2              # 0 at crest centre, 1 at root
        rr = r_maj - h * np.clip((d - 0.12) / 0.76, 0, 1)
        return rr

    verts = []
    for z in zs:
        u = ((z - THREAD_PITCH * th / (2 * math.pi)) / THREAD_PITCH) % 1.0
        r = prof(u)
        # 45° chamfers at both ends
        r = np.minimum(r, r_min + 0.3 + z)
        r = np.minimum(r, r_min + 0.3 + (THREAD_LEN - z))
        verts += list(zip(r * np.cos(th), r * np.sin(th), np.full(nth, z)))
    verts = np.array(verts)
    faces = []
    for i in range(nz - 1):
        for j in range(nth):
            a, b = i * nth + j, i * nth + (j + 1) % nth
            c, d = a + nth, b + nth
            faces += [(a, b, d), (a, d, c)]
    bot = len(verts); top = bot + 1
    verts = np.vstack([verts, [0, 0, 0], [0, 0, THREAD_LEN]])
    for j in range(nth):
        faces.append((bot, (j + 1) % nth, j))
        o = (nz - 1) * nth
        faces.append((top, o + j, o + (j + 1) % nth))
    m = Mesh(vert_properties=verts.astype(np.float32), tri_verts=np.array(faces, dtype=np.uint32))
    return Manifold(m)


def main():
    paths = []
    for k in range(N_STRUTS):
        phi0 = 2 * math.pi * k / N_STRUTS
        paths.append(strut_path(phi0, +1))
        paths.append(strut_path(phi0 + math.pi / N_STRUTS, -1))
    ring = ring_path(0.0, R2)

    print("building outer shell ...")
    outer = [tube(p, TUBE_R) for p in paths] + [tube(ring, TUBE_R)]
    hub = Manifold.cylinder(HUB_LEN, HUB_R, HUB_R, 64).translate((0, 0, BODY_LEN - HUB_LEN))
    thread = threaded_rod().translate((0, 0, BODY_LEN))
    solid = union(outer + [hub, thread])

    print("building water channels ...")
    channels = [tube(p, CHANNEL_R) for p in paths] + [tube(ring, CHANNEL_R)]
    chamber = Manifold.cylinder(HUB_LEN - 2.5, HUB_CHAMBER_R, HUB_CHAMBER_R, 48).translate((0, 0, BODY_LEN - HUB_LEN + 1.5))
    bore = Manifold.cylinder(THREAD_LEN + 5, BORE_D / 2, BORE_D / 2, 48).translate((0, 0, BODY_LEN - 3))
    voids = union(channels + [chamber, bore])

    print("drilling jets ...")
    jets = []
    all_pts = np.vstack(paths)
    for pi, p in enumerate(paths):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        others = np.vstack([q for qi, q in enumerate(paths) if qi != pi] + [ring])
        for sv in np.arange(25.0, s[-1] - 6, JET_SPACING):
            c = np.array([np.interp(sv, s, p[:, i]) for i in range(3)])
            if np.min(np.linalg.norm(others - c, axis=1)) < JET_CLEAR:
                continue
            inward = np.array([-c[0], -c[1], 0.0])
            inward /= np.linalg.norm(inward)
            jets.append(jet(c, inward))
    for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        c = np.array([R2 * math.cos(a), R2 * math.sin(a), 0.0])
        if np.min(np.linalg.norm(all_pts[all_pts[:, 2] < 12] - c, axis=1)) < JET_CLEAR:
            continue
        d = np.array([-math.cos(a), -math.sin(a), 0.6]); d /= np.linalg.norm(d)
        jets.append(jet(c, d))
    jet_area = len(jets) * math.pi * (JET_D / 2) ** 2
    print(f"  {len(jets)} jets, total area {jet_area:.0f} mm2 vs inlet {math.pi * (BORE_D / 2) ** 2:.0f} mm2")

    result = solid - union([voids] + jets)
    out = result.to_mesh()
    tm = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=True)
    print("watertight:", tm.is_watertight, "volume cm3: %.1f" % (tm.volume / 1000), "bounds:", tm.bounds.round(1).tolist())
    tm.export(OUT)
    print("saved", OUT)


def jet(c, d):
    """Ø JET_D cylinder from the channel centre outward along d (only pierces one wall)."""
    L = TUBE_R + 1.5
    cyl = Manifold.cylinder(L, JET_D / 2, JET_D / 2, 12)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); cth = float(np.dot(z, d))
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
    T = np.hstack([R, c.reshape(3, 1)])
    return cyl.transform(T.tolist())


if __name__ == "__main__":
    main()
