"""
Step 1: verify a stable XPBD sheet simulation with NO cutting yet.
Must settle sanely under gravity with no explosion -- baseline sanity
check before adding any cutting logic on top.
"""
import warp as wp
import numpy as np

wp.init()

nx, ny = 9, 7
size_x, size_y = 0.09, 0.07
dx = size_x / (nx - 1)
dy = size_y / (ny - 1)


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
                    rest_length: wp.array(dtype=wp.float32), compliance: wp.float32,
                    dt: wp.float32, lagrange: wp.array(dtype=wp.float32),
                    corr: wp.array(dtype=wp.vec3), corr_w: wp.array(dtype=wp.float32)):
    tid = wp.tid()
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
    alpha = compliance / (dt * dt)
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

    edges_a, edges_b, rest = [], [], []

    def add_edge(a, b):
        edges_a.append(a)
        edges_b.append(b)
        rest.append(float(np.linalg.norm(positions[a] - positions[b])))

    for r in range(ny):
        for c in range(nx):
            if c < nx - 1:
                add_edge(idx(r, c), idx(r, c + 1))
            if r < ny - 1:
                add_edge(idx(r, c), idx(r + 1, c))
            if r < ny - 1 and c < nx - 1:
                add_edge(idx(r, c), idx(r + 1, c + 1))
                add_edge(idx(r, c + 1), idx(r + 1, c))

    return positions, inv_mass, edges_a, edges_b, rest


def main():
    positions, inv_mass, edges_a, edges_b, rest = build_mesh()
    n = len(positions)

    x = wp.array(positions, dtype=wp.vec3)
    v = wp.zeros(n, dtype=wp.vec3)
    x_pred = wp.zeros(n, dtype=wp.vec3)
    inv_mass_wp = wp.array(inv_mass, dtype=wp.float32)

    idx_a = wp.array(np.array(edges_a, dtype=np.int32))
    idx_b = wp.array(np.array(edges_b, dtype=np.int32))
    rest_length = wp.array(np.array(rest, dtype=np.float32))
    n_constraints = len(edges_a)
    lagrange = wp.zeros(n_constraints, dtype=wp.float32)
    corr = wp.zeros(n, dtype=wp.vec3)
    corr_w = wp.zeros(n, dtype=wp.float32)

    gravity = wp.vec3(0.0, -9.81, 0.0)
    dt = 1.0 / 60.0
    substeps = 10
    sub_dt = dt / substeps
    iterations = 8
    compliance = 1.0e-7  # small -> stiff-ish but stable structural material

    n_frames = 120
    for frame in range(n_frames):
        for _ in range(substeps):
            wp.launch(predict, dim=n, inputs=[x, v, x_pred, inv_mass_wp, gravity, sub_dt])
            lagrange.zero_()
            for _it in range(iterations):
                corr.zero_()
                corr_w.zero_()
                wp.launch(solve_distance, dim=n_constraints,
                          inputs=[x_pred, inv_mass_wp, idx_a, idx_b, rest_length,
                                  compliance, sub_dt, lagrange, corr, corr_w])
                wp.launch(apply_correction, dim=n, inputs=[x_pred, corr, corr_w])
            wp.launch(update_velocity, dim=n, inputs=[x, x_pred, v, inv_mass_wp, sub_dt])

        if frame % 20 == 0:
            pos_np = x.numpy()
            bottom_y = pos_np[[idx(ny - 1, c) for c in range(nx)]][:, 1]
            print(f"frame={frame:4d}  bottom-row y: {bottom_y.min():.5f} to {bottom_y.max():.5f} "
                  f"(started at {-(ny - 1) * dy:.5f})")

    pos_np = x.numpy()
    if np.any(np.isnan(pos_np)) or np.any(np.abs(pos_np) > 10.0):
        print("!!! INSTABILITY DETECTED (nan or blew up)")
    else:
        print("Stable: no nan, no blow-up.")


if __name__ == "__main__":
    main()
