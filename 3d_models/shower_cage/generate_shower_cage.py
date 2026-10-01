"""
Shower cage head generator -> STL.

Geometry (all units in mm):
  * Male thread G1/2" (BSPP, standard shower hose): major Ø20.955, 14 TPI (pitch 1.814), 55° profile
  * Body section 1: Ø60 for 120 mm (right under the thread), with a gradual neck flare
  * Body section 2: Ø90 for 70 mm
  * Lattice of HOLLOW flat (elliptical section) struts, two opposite helix families -> diamond
    pattern, fed from a hub chamber under the thread and closed by a hollow flat bottom ring.
  * Many Ø0.8 high-pressure jet holes: on the inner side of every strut (pointing at the axis)
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
D2, L2 = 90.0, 70.0           # second section: diameter, length
BODY_LEN = L1 + L2

# flat struts: thin in the radial direction, wide tangentially
STRUT_RAD, STRUT_TAN = 3.5, 6.0       # outer semi-axes (7 x 12 mm)
WALL = 2.0                    # 5 perimeters with a 0.4 nozzle
HUB_R, HUB_LEN = 14.0, 18.0   # hub under the thread
HUB_CHAMBER_R = 9.0           # lens-shaped (double-cone) chamber, no flat walls
STRUT_START = HUB_LEN - 8     # struts start inside the chamber
NECK_FLARE_END = 60.0         # depth at which the neck reaches Ø60

N_STRUTS = 6                  # per helix direction (12 total)
TWIST = math.radians(150)     # rotation of each strut over the body length

# High pressure needs a SMALL total jet area: at a fixed supply flow the jet speed is
# v = Q / (Cd * A_total). ~20-25 mm2 gives >10 m/s at 10 L/min (see engineering_check()).
JET_D = 0.8                   # printable on FDM with a 0.4 nozzle
JET_BAND = 10.0               # mm: height of each coverage band along the body
JETS_PER_BAND = 5             # jets per band, spread around the circumference
JET_Z_MAX = 160.0             # above this the struts are bunched into the neck
JET_CLEAR = 7.0               # pre-filter: skip spots this close to a crossing strut
RING_JETS_PER_ROW = 14        # ring has 2 staggered horizontal rows
RING_JET_Z = (-2.0, 2.0)          # both inside the ring channel height

# suction-cup keyhole pads (for mushroom-head cups: head ≤7 mm, neck ≤4 mm)
N_CUPS = 7
CUP_HEAD_HOLE = 7.5
CUP_NECK_SLOT = 4.2
CUP_SLOT_LEN = 6.0
PAD_TOP, PAD_BOTTOM = -5.0, -13.5
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
    flare = 7 + (R1 - 7) * smoothstep(STRUT_START, NECK_FLARE_END, t) ** 0.8
    return flare + (R2 - R1) * smoothstep(L1 - 8, L1 + 18, t)


def strut_path(phi0, direction, n=90):
    ts = np.linspace(STRUT_START, BODY_LEN, n)
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


def wall_reach(c, d):
    """Max distance from the strut centre-line point c to the outer surface along d
    (support function of the strut's elliptical section)."""
    er = np.array([c[0], c[1], 0.0]); er /= np.linalg.norm(er)
    et = np.array([-er[1], er[0], 0.0])
    return math.sqrt((STRUT_RAD * d @ er) ** 2 + (STRUT_TAN * d @ et) ** 2 + (STRUT_TAN * d[2]) ** 2)


def jet(c, d):
    """Ø JET_D hole from the channel centre outward along d (pierces only the inner wall)."""
    return oriented_cylinder(c, d, wall_reach(c, d) + 1.5, JET_D / 2)


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
    z0 = BODY_LEN - HUB_LEN + 2.0     # chamber bottom tip
    chamber = Manifold.batch_hull([
        Manifold.cylinder(0.1, 3.0, 3.0, 48).translate((0, 0, z0)),
        Manifold.cylinder(BODY_LEN - z0 - 11, HUB_CHAMBER_R, HUB_CHAMBER_R, 48).translate((0, 0, z0 + 6)),
        Manifold.cylinder(0.1, BORE_D / 2, BORE_D / 2, 48).translate((0, 0, BODY_LEN - 1))])
    bore = Manifold.cylinder(THREAD_LEN + 5, BORE_D / 2, BORE_D / 2, 48).translate((0, 0, BODY_LEN - 3))
    voids = union(channels + [chamber, bore])

    print("drilling jets ...")
    # candidate jet spots every 3 mm along every strut, away from crossings
    cands = []
    for pi, p in enumerate(paths):
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        others = np.vstack([q for qi, q in enumerate(paths) if qi != pi] + [ring])
        for sv in np.arange(3.0, s[-1] - 8, 3.0):
            c = np.array([np.interp(sv, s, p[:, i]) for i in range(3)])
            dists = np.linalg.norm(others - c, axis=1)
            if c[2] > JET_Z_MAX:
                continue
            if dists.min() < 2.5:
                # at a crossing node: both channels merge there, drill from the node centre
                c = (c + others[dists.argmin()]) / 2
            elif dists.min() < JET_CLEAR:
                continue
            # point at the axis, but perpendicular to the strut so the hole crosses the wall squarely
            i = np.searchsorted(s, sv); tan_v = p[min(i, len(p) - 1)] - p[max(i - 1, 0)]
            tan_v /= np.linalg.norm(tan_v)
            inward = np.array([-c[0], -c[1], 0.0])
            inward -= np.dot(inward, tan_v) * tan_v
            cands.append((c, inward / np.linalg.norm(inward)))

    # even coverage of the whole inside: every height band gets JETS_PER_BAND jets spread
    # around the circumference (band start angle advances by the golden angle)
    jets, jet_pts = [], []
    golden = math.pi * (3 - math.sqrt(5))
    z_edges = np.arange(5.0, JET_Z_MAX + 1e-6, JET_BAND)
    for b, (z_lo, z_hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        band = [cd for cd in cands if z_lo <= cd[0][2] < z_hi]
        used, rejected = [], set()
        for k in range(JETS_PER_BAND):
            target = b * golden + 2 * math.pi * k / JETS_PER_BAND
            best = None
            for idx, (c, d) in enumerate(band):
                if idx in used or idx in rejected or any(np.linalg.norm(c - band[u][0]) < 6.0 for u in used):
                    continue
                err = abs((math.atan2(c[1], c[0]) - target + math.pi) % (2 * math.pi) - math.pi)
                if best is None or err < best[0]:
                    best = (err, idx)
            while best is not None and not jet_is_clear(solid, *band[best[1]]):
                rejected.add(best[1])   # tip inside a neighbour strut or spray blocked -> next best
                best = min(((abs((math.atan2(c[1], c[0]) - target + math.pi) % (2 * math.pi) - math.pi), idx)
                            for idx, (c, d) in enumerate(band)
                            if idx not in used and idx not in rejected
                            and all(np.linalg.norm(c - band[u][0]) >= 6.0 for u in used)), default=None)
            if best is not None:
                used.append(best[1])
                c, d = band[best[1]]
                jets.append(jet(c, d)); jet_pts.append((c, d))
    n_strut_jets = len(jets)

    strut_ends = np.array([p[-1] for p in paths])
    for row, z in enumerate(RING_JET_Z):
        for k in range(RING_JETS_PER_ROW):
            a = 2 * math.pi * (k + 0.5 * row) / RING_JETS_PER_ROW
            c = np.array([R2 * math.cos(a), R2 * math.sin(a), z])
            if np.min(np.linalg.norm(strut_ends - c, axis=1)) < 3.0:
                continue
            d = np.array([-math.cos(a), -math.sin(a), 0.0])
            jets.append(jet(c, d)); jet_pts.append((c, d))
    jet_area = len(jets) * math.pi * (JET_D / 2) ** 2
    print(f"  {len(jets)} jets ({n_strut_jets} on struts, {len(jets) - n_strut_jets} on ring), "
          f"total area {jet_area:.0f} mm2 vs inlet {math.pi * (BORE_D / 2) ** 2:.0f} mm2")

    np.savetxt("jets.csv", [[*c, *d] for c, d in jet_pts], delimiter=",", fmt="%.2f",
               header="x,y,z,dir_x,dir_y,dir_z (mm; jet centre on the channel axis + spray direction)")
    verify(solid, voids, jets, jet_pts, pad_cuts)
    engineering_check()
    jet_performance(len(jets))
    result = solid - union([voids] + jets + pad_cuts)
    out = result.to_mesh()
    tm = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=False)
    print("watertight:", tm.is_watertight, "volume cm3: %.1f" % (tm.volume / 1000), "bounds:", tm.bounds.round(1).tolist())
    tm.export(OUT)
    print("saved", OUT)


def ellipse_ring_moment(a, b, p, n=2000):
    """Max bending moment (N*mm/mm) in a thin closed elliptical ring (mid-wall semi-axes a>b)
    under internal pressure p (MPa). Quarter-ring statics + zero end rotation (Castigliano)."""
    th = np.linspace(0, math.pi / 2, n)
    x, y = a * np.cos(th), b * np.sin(th)
    ds = np.hypot(np.gradient(x), np.gradient(y))
    m0 = -p * a * (a - x) + p / 2 * ((a - x) ** 2 + y ** 2)   # moment w/o the redundant M_A
    m_a = -np.sum(m0 * ds) / np.sum(ds)
    return np.max(np.abs(m0 + m_a))


def engineering_check():
    """Wall stresses at typical / high / extreme household pressure, and jet performance."""
    print("pressure / strength check (thin-wall theory, PETG):")
    a, b = STRUT_TAN - WALL / 2, STRUT_RAD - WALL / 2
    ai, bi = STRUT_TAN - WALL, STRUT_RAD - WALL
    r_root = THREAD_MAJOR_D / 2 - 0.640327 * THREAD_PITCH
    ri = BORE_D / 2
    layer_strength = 25.0          # MPa, PETG across layers (weakest direction), conservative
    for bar in (3, 6, 10):
        p = bar / 10.0
        sig_strut = 6 * ellipse_ring_moment(a, b, p) / WALL ** 2 + p * ai / WALL
        sig_neck = p * (r_root ** 2 + ri ** 2) / (r_root ** 2 - ri ** 2)
        hose_pull = p * math.pi * (THREAD_MAJOR_D / 2) ** 2 / (math.pi * (r_root ** 2 - ri ** 2))
        worst = max(sig_strut, sig_neck, hose_pull)
        print(f"  {bar:>2} bar: strut/ring wall {sig_strut:5.1f} MPa, thread neck {sig_neck:4.1f} MPa, "
              f"hose pull {hose_pull:4.1f} MPa -> safety factor {layer_strength / worst:4.1f}")
    print("flow check:")
    ch = math.pi * ai * bi
    print(f"  channel area {12 * ch:.0f} mm2 (12 struts) vs inlet {math.pi * ri ** 2:.0f} mm2 "
          f"-> water speed inside channels stays low, all jets get ~the same pressure")


def jet_performance(n_jets):
    a_tot = n_jets * math.pi * (JET_D / 2) ** 2
    cd = 0.62
    for lpm in (8, 10, 12):
        q = lpm / 60000.0
        v = q / (cd * a_tot * 1e-6)
        print(f"  at {lpm:>2} L/min: jet speed {v:4.1f} m/s, pressure used by the jets {1000 * v * v / 2 / 1e5:4.2f} bar")


def jet_is_clear(solid, c, d):
    """True if the jet's exit lies outside the solid and its spray line to the axis is free."""
    reach = wall_reach(c, d)
    tip = c + d * (reach + 1.0)
    start = c + d * (reach + 0.6)
    dist = max(np.hypot(start[0], start[1]) - 1.0, 1.0)
    line = oriented_cylinder(start, d, dist, 0.15, 6)
    tip_in = not (Manifold.sphere(0.1, 8).translate(tuple(tip)) ^ solid).is_empty()
    return not tip_in and (line ^ solid).is_empty()


def verify(solid, voids, jets, jet_pts, pad_cuts):
    """Check that the water path is one connected cavity and that every jet sprays freely."""
    print("verifying water path ...")
    n_cav = len(voids.decompose())
    print(f"  internal cavity pieces (inlet + chamber + all channels): {n_cav} -> "
          + ("OK, one connected network" if n_cav == 1 else "PROBLEM: disconnected channels"))
    fed = sum(1 for j in jets if not (j ^ voids).is_empty())
    print(f"  jets touching the cavity: {fed}/{len(jets)}")
    # a jet is open if its tip lies outside the solid, and its spray line to the axis is clear
    blocked = 0
    for c, d in jet_pts:
        if not jet_is_clear(solid, c, d):
            blocked += 1
            print(f"    blocked jet at {np.round(c, 1)}")
    print(f"  jets with a clear spray to the centre: {len(jets) - blocked}/{len(jets)}")
    zs = np.array([c[2] for c, _ in jet_pts])
    bands = np.arange(-5.0, JET_Z_MAX + 1e-6, JET_BAND)
    counts = np.histogram(zs, bands)[0]
    empty = [f"{lo:.0f}-{lo + JET_BAND:.0f}" for lo, n in zip(bands[:-1], counts) if n == 0]
    print(f"  jets per {JET_BAND:.0f} mm height band (bottom->top): {counts.tolist()}")
    print("  full-height coverage: " + ("OK, every band has jets" if not empty else "GAPS at z=" + ", ".join(empty)))
    leak = union(pad_cuts) ^ voids
    print("  suction-cup keyholes isolated from water: " + ("OK" if leak.is_empty() else "PROBLEM"))
    ok = not empty and n_cav == 1 and fed == len(jets) and blocked == 0 and leak.is_empty()
    print("  RESULT:", "all checks passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    main()
