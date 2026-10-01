"""
Shower cage head generator -> STL.

Geometry (all units in mm):
  * Male thread G1/2" (BSPP, standard shower hose): major Ø20.955, 14 TPI (pitch 1.814), 55° profile
  * Body section 1: Ø60 for 120 mm (right under the thread), with a gradual neck flare
  * Body section 2: Ø110 for 70 mm
  * Lattice of HOLLOW flat (elliptical section) struts, two opposite helix families -> diamond
    pattern, fed from a hub chamber under the thread and closed by a hollow flat bottom ring.
  * Many small high-pressure jet holes: on the inner side of every strut (pointing at the axis)
    and two staggered horizontal rows on the inner face of the ring (pointing at the centre).
  * 7 keyhole pads under the ring for standard mushroom-head suction cups.

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

# flat struts: thin in the radial direction, wide tangentially
STRUT_RAD, STRUT_TAN = 3.25, 5.5      # outer semi-axes (6.5 x 11 mm)
WALL = 1.6
HUB_R, HUB_LEN = 13.0, 12.0   # solid hub under the thread
HUB_CHAMBER_R = 9.0
NECK_FLARE_END = 60.0         # depth at which the neck reaches Ø60

N_STRUTS = 6                  # per helix direction (12 total)
TWIST = math.radians(150)     # rotation of each strut over the body length

JET_D = 0.7                   # total jet area kept below the inlet bore area -> high pressure
JET_SPACING = 10.0            # mm along each strut
JET_CLEAR = 9.0               # skip jets this close to a crossing strut
RING_JETS_PER_ROW = 48        # ring has 2 staggered horizontal rows
RING_JET_Z = (-2.0, 2.0)

# suction-cup keyhole pads (for mushroom-head cups: head ≤7 mm, neck ≤4 mm)
N_CUPS = 7
CUP_HEAD_HOLE = 7.5
CUP_NECK_SLOT = 4.2
CUP_SLOT_LEN = 6.0
PAD_TOP, PAD_BOTTOM = -4.5, -13.0
PAD_SHEET = 2.0               # thickness the mushroom head locks behind

SPHERE_SEGS = 20
OUT = "shower_cage.stl"

R1 = D1 / 2 - STRUT_RAD       # strut centre-line radii
R2 = D2 / 2 - STRUT_RAD


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def radius_at(t):
    """Centre-line radius at depth t (0 = top of body under thread, BODY_LEN = bottom)."""
    flare = 6 + (R1 - 6) * smoothstep(HUB_LEN - 4, NECK_FLARE_END, t) ** 0.8
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


def ring_path(z, r, n=96):
    a = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], 1)


def union(parts):
    return Manifold.batch_boolean(parts, OpType.Add)


def ellipsoid(c, a_rad, a_tan):
    """Ellipsoid at c: semi-axis a_rad along the radial direction, a_tan tangentially and vertically."""
    er = np.array([c[0], c[1], 0.0])
    er = er / np.linalg.norm(er) if np.linalg.norm(er) > 1e-6 else np.array([1.0, 0, 0])
    et = np.array([-er[1], er[0], 0.0])
    ez = np.array([0, 0, 1.0])
    M = np.column_stack([er * a_rad, et * a_tan, ez * a_tan, c])
    return Manifold.sphere(1.0, SPHERE_SEGS).transform(M.tolist())


def flat_tube(path, a_rad, a_tan):
    segs = [Manifold.batch_hull([ellipsoid(a, a_rad, a_tan), ellipsoid(b, a_rad, a_tan)])
            for a, b in zip(path[:-1], path[1:])]
    return union(segs)


def oriented_cylinder(c, d, length, radius, segs=12):
    """Cylinder starting at c and extending `length` along unit vector d."""
    cyl = Manifold.cylinder(length, radius, radius, segs)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); cth = float(np.dot(z, d))
    if s < 1e-9:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
    return cyl.transform(np.hstack([R, np.reshape(c, (3, 1))]).tolist())


def jet(c, d):
    """Ø JET_D hole from the channel centre outward along d (pierces only the inner wall)."""
    return oriented_cylinder(c, d, STRUT_RAD + 1.5, JET_D / 2)


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
        return r_maj - h * np.clip((d - 0.12) / 0.76, 0, 1)

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
    o = (nz - 1) * nth
    for j in range(nth):
        faces.append((bot, (j + 1) % nth, j))
        faces.append((top, o + j, o + (j + 1) % nth))
    m = Mesh(vert_properties=verts.astype(np.float32), tri_verts=np.array(faces, dtype=np.uint32))
    return Manifold(m)


def cup_pads():
    """Solid pads under the ring and the keyhole cut for each mushroom-head suction cup."""
    pads, cuts = [], []
    h = PAD_TOP - PAD_BOTTOM
    for k in range(N_CUPS):
        a = 2 * math.pi * (k + 0.5) / N_CUPS
        rot = math.degrees(a)
        pad = Manifold.batch_hull([
            Manifold.cylinder(h, 6.0, 6.0, 40).translate((0, y, PAD_BOTTOM))
            for y in (-CUP_SLOT_LEN / 2, CUP_SLOT_LEN / 2)])
        # entry hole for the head (through everything up to the head cavity)
        head_top = PAD_BOTTOM + PAD_SHEET + 3.5
        entry = Manifold.cylinder(head_top - PAD_BOTTOM + 1, CUP_HEAD_HOLE / 2, CUP_HEAD_HOLE / 2, 32) \
            .translate((0, -CUP_SLOT_LEN / 2, PAD_BOTTOM - 1))
        # neck slot through the bottom sheet
        slot = Manifold.batch_hull([
            Manifold.cylinder(PAD_SHEET + 2, CUP_NECK_SLOT / 2, CUP_NECK_SLOT / 2, 24).translate((0, y, PAD_BOTTOM - 1))
            for y in (-CUP_SLOT_LEN / 2, CUP_SLOT_LEN / 2)])
        # cavity behind the sheet where the head slides and locks
        cavity = Manifold.batch_hull([
            Manifold.cylinder(3.5, CUP_HEAD_HOLE / 2, CUP_HEAD_HOLE / 2, 32).translate((0, y, PAD_BOTTOM + PAD_SHEET))
            for y in (-CUP_SLOT_LEN / 2, CUP_SLOT_LEN / 2)])
        place = lambda m: m.translate((R2 - 3.5, 0, 0)).rotate((0, 0, rot))
        pads.append(place(pad))
        cuts.append(place(union([entry, slot, cavity])))
    return pads, cuts


def main():
    paths = []
    for k in range(N_STRUTS):
        phi0 = 2 * math.pi * k / N_STRUTS
        paths.append(strut_path(phi0, +1))
        paths.append(strut_path(phi0 + math.pi / N_STRUTS, -1))
    ring = ring_path(0.0, R2)
    pads, pad_cuts = cup_pads()

    print("building outer shell ...")
    outer = [flat_tube(p, STRUT_RAD, STRUT_TAN) for p in paths] + [flat_tube(ring, STRUT_RAD, STRUT_TAN)]
    hub = Manifold.cylinder(HUB_LEN, HUB_R, HUB_R, 64).translate((0, 0, BODY_LEN - HUB_LEN))
    thread = threaded_rod().translate((0, 0, BODY_LEN))
    solid = union(outer + pads + [hub, thread])

    print("building water channels ...")
    ci, ct = STRUT_RAD - WALL, STRUT_TAN - WALL
    channels = [flat_tube(p, ci, ct) for p in paths] + [flat_tube(ring, ci, ct)]
    chamber = Manifold.cylinder(HUB_LEN - 2.5, HUB_CHAMBER_R, HUB_CHAMBER_R, 48).translate((0, 0, BODY_LEN - HUB_LEN + 1.5))
    bore = Manifold.cylinder(THREAD_LEN + 5, BORE_D / 2, BORE_D / 2, 48).translate((0, 0, BODY_LEN - 3))
    voids = union(channels + [chamber, bore])

    print("drilling jets ...")
    jets = []
    for pi, p in enumerate(paths):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        others = np.vstack([q for qi, q in enumerate(paths) if qi != pi] + [ring])
        for sv in np.arange(30.0, s[-1] - 8, JET_SPACING):
            c = np.array([np.interp(sv, s, p[:, i]) for i in range(3)])
            if np.min(np.linalg.norm(others - c, axis=1)) < JET_CLEAR:
                continue
            inward = np.array([-c[0], -c[1], 0.0])
            jets.append(jet(c, inward / np.linalg.norm(inward)))
    n_strut_jets = len(jets)

    strut_ends = np.array([p[-1] for p in paths])
    for row, z in enumerate(RING_JET_Z):
        for k in range(RING_JETS_PER_ROW):
            a = 2 * math.pi * (k + 0.5 * row) / RING_JETS_PER_ROW
            c = np.array([R2 * math.cos(a), R2 * math.sin(a), z])
            if np.min(np.linalg.norm(strut_ends - c, axis=1)) < 3.0:
                continue
            jets.append(jet(c, np.array([-math.cos(a), -math.sin(a), 0.0])))
    jet_area = len(jets) * math.pi * (JET_D / 2) ** 2
    print(f"  {len(jets)} jets ({n_strut_jets} on struts, {len(jets) - n_strut_jets} on ring), "
          f"total area {jet_area:.0f} mm2 vs inlet {math.pi * (BORE_D / 2) ** 2:.0f} mm2")

    result = solid - union([voids] + jets + pad_cuts)
    out = result.to_mesh()
    tm = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=True)
    print("watertight:", tm.is_watertight, "volume cm3: %.1f" % (tm.volume / 1000), "bounds:", tm.bounds.round(1).tolist())
    tm.export(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
