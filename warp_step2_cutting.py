"""
Step 2: add the actual cutting -- virtual-node duplication + breakable
cohesive constraint, on top of the verified-stable XPBD sheet from step 1.

Simplification vs. the full smooth traction-softening law: this uses a
fixed-compliance cohesive constraint (holds firmly) that BREAKS entirely
once separation crosses delta_c, rather than smoothly softening the
traction from sigma_max down to 0 as the numpy prototype attempted. This
is a standard simplification in real-time PBD fracture (a "brittle"
cohesive law) and is far more numerically robust than the smooth version,
which is what actually caused problems in the earlier numpy prototype.
Smooth softening can be added later (as a compliance that grows with
separation) once this simpler version is confirmed solid -- which is
exactly what we're checking here.
"""
import warp as wp
import numpy as np

wp.init()

nx, ny = 9, 7
size_x, size_y = 0.09, 0.07
dx = size_x / (nx - 1)
dy = size_y / (ny - 1)
r_cut = ny // 2

delta_c = 0.0005  # 0.5mm critical opening -- cohesive constraint breaks here


def idx(r, c):
    return r * nx + c


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            x_pred: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=wp.float32),
            gravity: wp.vec3, dt: wp.float32):
    tid = wp.tid()
    if inv_mass[tid] > 0.0:
        v[tid] = v[tid] + gravity * dt
    x_pred[tid] = x[tid] + v[tid] * dt


@wp.kernel
def solve_distance(x_pred: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=wp.float32),
                    idx_a: wp.array(dtype=wp.int32), idx_b: wp.array(dtype=wp.int32),
                    rest_length: wp.array(dtype=wp.float32), compliance_arr: wp.array(dtype=wp.float32),
                    active: wp.array(dtype=wp.int32),
                    dt: wp.float32, lagrange: wp.array(dtype=wp.float32),
                    corr: wp.array(dtype=wp.vec3), corr_w: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if active[tid] == 0:
        return
    a = idx_a[tid]
    b = idx_b[tid]
    wa = inv_mass[a]
    wb = inv_mass[b]
    if wa + wb == 0.0:
        return
    diff = x_pred[a] - x_pred[b]
    dist = wp.length(diff)
    if dist < 1.0e-8:
        return
    n = diff / dist
    C = dist - rest_length[tid]
    alpha = compliance_arr[tid] / (dt * dt)
    dlambda = (-C - alpha * lagrange[tid]) / (wa + wb + alpha)
    lagrange[tid] = lagrange[tid] + dlambda
    corr_vec = n * dlambda
    wp.atomic_add(corr, a, corr_vec * wa)
    wp.atomic_add(corr, b, -corr_vec * wb)
    wp.atomic_add(corr_w, a, 1.0)
    wp.atomic_add(corr_w, b, 1.0)


@wp.kernel
def apply_correction(x_pred: wp.array(dtype=wp.vec3), corr: wp.array(dtype=wp.vec3),
                      corr_w: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if corr_w[tid] > 0.0:
        x_pred[tid] = x_pred[tid] + corr[tid] / corr_w[tid]


@wp.kernel
def update_velocity(x: wp.array(dtype=wp.vec3), x_pred: wp.array(dtype=wp.vec3),
                     v: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=wp.float32),
                     dt: wp.float32):
    tid = wp.tid()
    if inv_mass[tid] > 0.0:
        v[tid] = (x_pred[tid] - x[tid]) / dt
    x[tid] = x_pred[tid]


def build_mesh():
    positions = []
    for r in range(ny):
        for c in range(nx):
            positions.append([c * dx, -r * dy, 0.0])
    positions = np.array(positions, dtype=np.float32)

    inv_mass = np.full(len(positions), 1.0 / 0.007, dtype=np.float32)
    for c in range(nx):
        inv_mass[idx(0, c)] = 0.0  # pin top row

    edges_a, edges_b, rest, compliance = [], [], [], []
    STRUCT_COMPLIANCE = 1.0e-7

    def add_edge(a, b, comp=STRUCT_COMPLIANCE):
        edges_a.append(a)
        edges_b.append(b)
        rest.append(float(np.linalg.norm(positions[a] - positions[b])))
        compliance.append(comp)

    for r in range(ny):
        for c in range(nx):
            if c < nx - 1:
                add_edge(idx(r, c), idx(r, c + 1))
            if r < ny - 1:
                add_edge(idx(r, c), idx(r + 1, c))
            if r < ny - 1 and c < nx - 1:
                add_edge(idx(r, c), idx(r + 1, c + 1))
                add_edge(idx(r, c + 1), idx(r + 1, c))

    return positions, inv_mass, edges_a, edges_b, rest, compliance


def main():
    positions, inv_mass, edges_a, edges_b, rest, compliance = build_mesh()
    n_orig = len(positions)

    positions = positions.tolist()
    inv_mass = inv_mass.tolist()

    n_active_constraints = len(edges_a)
    active_flags = [1] * n_active_constraints

    # book-keeping for the knife sweep
    duplicated = {}          # column -> duplicate particle index
    cohesive_constraint_idx = {}  # column -> index into the constraint arrays

    gravity_vec = wp.vec3(0.0, -9.81, 0.0)
    dt = 1.0 / 60.0
    substeps = 10
    sub_dt = dt / substeps
    iterations = 8
    COHESIVE_COMPLIANCE = 5.0e-9  # stiffer than structural -> holds firmly until it breaks

    n_frames = 240
    knife_sweep_frames = 150

    def rebuild_gpu_arrays():
        x = wp.array(np.array(positions, dtype=np.float32), dtype=wp.vec3)
        v_np = np.zeros((len(positions), 3), dtype=np.float32)
        return x, v_np

    x = wp.array(np.array(positions, dtype=np.float32), dtype=wp.vec3)
    v = wp.zeros(len(positions), dtype=wp.vec3)

    for frame in range(n_frames):
        # --- knife sweep: duplicate newly-reached columns this frame ---
        knife_col = int((nx - 1) * min(1.0, frame / knife_sweep_frames))
        newly_cut = []
        for c in range(knife_col + 1):
            if c not in duplicated:
                orig_i = idx(r_cut, c)
                dup_i = len(positions)
                positions.append(list(positions[orig_i]))
                inv_mass.append(inv_mass[orig_i])
                duplicated[c] = dup_i
                newly_cut.append(c)

        if newly_cut:
            # retarget lower-side structural connections to the duplicate,
            # and add the cohesive constraint + any lower-side rigid links
            # to already-duplicated neighbor columns
            for c in newly_cut:
                orig_i = idx(r_cut, c)
                dup_i = duplicated[c]
                for i in range(len(edges_a)):
                    a, b = edges_a[i], edges_b[i]
                    if a == orig_i and b // nx > r_cut:
                        edges_a[i] = dup_i
                    elif b == orig_i and a // nx > r_cut:
                        edges_b[i] = dup_i
                # cohesive constraint (zero rest length -- pulls dup back to orig)
                edges_a.append(orig_i)
                edges_b.append(dup_i)
                rest.append(0.0)
                compliance.append(COHESIVE_COMPLIANCE)
                active_flags.append(1)
                cohesive_constraint_idx[c] = len(edges_a) - 1
                # lower-side rigid link to already-cut neighbor(s)
                if (c - 1) in duplicated:
                    edges_a.append(duplicated[c - 1])
                    edges_b.append(dup_i)
                    rest.append(dx)
                    compliance.append(1.0e-7)
                    active_flags.append(1)
                if (c + 1) in duplicated:
                    edges_a.append(dup_i)
                    edges_b.append(duplicated[c + 1])
                    rest.append(dx)
                    compliance.append(1.0e-7)
                    active_flags.append(1)

            # arrays changed size -> rebuild all GPU buffers this frame
            n = len(positions)
            x = wp.array(np.array(positions, dtype=np.float32), dtype=wp.vec3)
            v_np = v.numpy()
            v_np = np.vstack([v_np, np.zeros((n - v_np.shape[0], 3), dtype=np.float32)])
            v = wp.array(v_np, dtype=wp.vec3)

        n = len(positions)
        n_c = len(edges_a)
        x_pred = wp.zeros(n, dtype=wp.vec3)
        inv_mass_wp = wp.array(np.array(inv_mass, dtype=np.float32))
        idx_a_wp = wp.array(np.array(edges_a, dtype=np.int32))
        idx_b_wp = wp.array(np.array(edges_b, dtype=np.int32))
        rest_wp = wp.array(np.array(rest, dtype=np.float32))
        compliance_wp = wp.array(np.array(compliance, dtype=np.float32))
        active_wp = wp.array(np.array(active_flags, dtype=np.int32))
        lagrange = wp.zeros(n_c, dtype=wp.float32)
        corr = wp.zeros(n, dtype=wp.vec3)
        corr_w = wp.zeros(n, dtype=wp.float32)

        for _ in range(substeps):
            wp.launch(predict, dim=n, inputs=[x, v, x_pred, inv_mass_wp, gravity_vec, sub_dt])
            lagrange.zero_()
            for _it in range(iterations):
                corr.zero_()
                corr_w.zero_()
                wp.launch(solve_distance, dim=n_c,
                          inputs=[x_pred, inv_mass_wp, idx_a_wp, idx_b_wp, rest_wp,
                                  compliance_wp, active_wp, sub_dt, lagrange, corr, corr_w])
                wp.launch(apply_correction, dim=n, inputs=[x_pred, corr, corr_w])
            wp.launch(update_velocity, dim=n, inputs=[x, x_pred, v, inv_mass_wp, sub_dt])

        # --- check cohesive constraints for breakage ---
        pos_np = x.numpy()
        for c, ci in list(cohesive_constraint_idx.items()):
            if active_flags[ci] == 0:
                continue
            a, b = edges_a[ci], edges_b[ci]
            dist = np.linalg.norm(pos_np[a] - pos_np[b])
            if dist >= delta_c:
                active_flags[ci] = 0

        if frame % 30 == 0:
            n_active_cohesive = sum(1 for ci in cohesive_constraint_idx.values() if active_flags[ci] == 1)
            print(f"frame={frame:4d}  knife_col={knife_col}  "
                  f"columns_cut={len(duplicated)}  cohesive_still_bonded={n_active_cohesive}  "
                  f"max|pos|={np.abs(pos_np).max():.4f}")

    pos_np = x.numpy()
    if np.any(np.isnan(pos_np)) or np.any(np.abs(pos_np) > 10.0):
        print("!!! INSTABILITY DETECTED")
    else:
        print("Stable throughout cutting.")
        upper_y = pos_np[[idx(r_cut, c) for c in range(nx)]][:, 1]
        lower_y = [pos_np[duplicated[c]][1] for c in duplicated]
        print(f"  upper cut-row y: {upper_y.min():.5f} to {upper_y.max():.5f}")
        print(f"  lower (duplicate) y: {min(lower_y):.5f} to {max(lower_y):.5f}")
        print(f"  separation achieved: {min(lower_y) - upper_y.max():.5f} m "
              f"(should be a real gap if cutting worked)")


if __name__ == "__main__":
    main()
