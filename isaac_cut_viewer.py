"""
isaac_cut_viewer.py
---------------------
Run this in Isaac Sim, ALONGSIDE mouse_control.py running in a separate
terminal. This script:
  - runs the same verified Warp/XPBD cutting simulation (identical
    kernels to mouse_cut_demo.py / warp_step2_cutting_FIXED.py -- no
    changes to the physics, only how it's driven and rendered)
  - reads the current knife-cut target from the shared file that
    mouse_control.py writes every frame
  - writes the resulting particle positions into a USD mesh's points
    each frame, so Isaac Sim renders it

IMPORTANT -- READ BEFORE RUNNING:
I don't have Isaac Sim or a GPU in my own environment, so unlike the
Warp physics (verified on both CPU here and your lab GPUs), the Isaac
Sim/USD side of this script is grounded in current documentation but
NOT executed by me. The physics kernels are the exact ones we already
validated twice; the new, unverified part is purely the USD mesh
creation and per-frame points update below. If something errors, paste
me the exact traceback and we'll fix it the same way we've fixed
everything else.

Run with (from the Isaac Sim install directory):
    ./python.sh /path/to/isaac_cut_viewer.py
"""

import json
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom, Sdf, Vt, Gf, UsdPhysics
import warp as wp
import numpy as np

wp.init()

SHARED_STATE_FILE = "/tmp/warp_cut_knife_state.json"

# ---------------- mesh / material parameters (match mouse_control.py) ----------------
nx, ny = 9, 7
size_x, size_y = 0.09, 0.07
dx = size_x / (nx - 1)
dy = size_y / (ny - 1)
r_cut = ny // 2
delta_c = 0.0005

STRUCT_COMPLIANCE = 1.0e-7
COHESIVE_COMPLIANCE = 5.0e-4


def idx(r, c):
    return r * nx + c


# ---------------- Warp kernels (verified stable on CPU and lab GPUs) ----------------
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
def clamp_velocity(v: wp.array(dtype=wp.vec3), max_speed: wp.float32):
    tid = wp.tid()
    speed = wp.length(v[tid])
    if speed > max_speed:
        v[tid] = v[tid] * (max_speed / speed)


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
            positions.append([c * dx, 0.0, -r * dy])  # USD: Y-up convention

    inv_mass = [1.0 / 0.007] * len(positions)
    for c in range(nx):
        inv_mass[idx(0, c)] = 0.0

    edges_a, edges_b, rest, compliance = [], [], [], []

    def add_edge(a, b, comp=STRUCT_COMPLIANCE):
        edges_a.append(a)
        edges_b.append(b)
        rest.append(float(np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))))
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

    # triangle faces for rendering (top surface only, single-sided sheet)
    faces = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            p0, p1 = idx(r, c), idx(r, c + 1)
            p2, p3 = idx(r + 1, c), idx(r + 1, c + 1)
            faces += [(p0, p1, p3), (p0, p3, p2)]

    return positions, inv_mass, edges_a, edges_b, rest, compliance, faces


class CuttingSim:
    def __init__(self):
        (self.positions, self.inv_mass, self.edges_a, self.edges_b,
         self.rest, self.compliance, self.faces) = build_mesh()
        self.active_flags = [1] * len(self.edges_a)
        self.duplicated = {}
        self.cohesive_constraint_idx = {}
        self.n_orig = nx * ny

        self.gravity_vec = wp.vec3(0.0, -9.81, 0.0)
        self.dt = 1.0 / 60.0
        self.substeps = 10
        self.sub_dt = self.dt / self.substeps
        self.iterations = 8
        self.max_speed = 2.0

        self.x = wp.array(np.array(self.positions, dtype=np.float32), dtype=wp.vec3)
        self.v = wp.zeros(len(self.positions), dtype=wp.vec3)
        self.knife_col = -1

    def set_knife_target(self, col_target):
        self.knife_col = max(self.knife_col, col_target)

    def step(self):
        newly_cut = []
        for c in range(self.knife_col + 1):
            if c not in self.duplicated:
                orig_i = idx(r_cut, c)
                dup_i = len(self.positions)
                self.positions.append(list(self.positions[orig_i]))
                self.inv_mass.append(self.inv_mass[orig_i])
                self.duplicated[c] = dup_i
                newly_cut.append(c)

        if newly_cut:
            for c in newly_cut:
                orig_i = idx(r_cut, c)
                dup_i = self.duplicated[c]
                for i in range(len(self.edges_a)):
                    a, b = self.edges_a[i], self.edges_b[i]
                    if a == orig_i and b // nx > r_cut:
                        self.edges_a[i] = dup_i
                    elif b == orig_i and a // nx > r_cut:
                        self.edges_b[i] = dup_i
                self.edges_a.append(orig_i)
                self.edges_b.append(dup_i)
                self.rest.append(0.0)
                self.compliance.append(COHESIVE_COMPLIANCE)
                self.active_flags.append(1)
                self.cohesive_constraint_idx[c] = len(self.edges_a) - 1
                if (c - 1) in self.duplicated:
                    self.edges_a.append(self.duplicated[c - 1])
                    self.edges_b.append(dup_i)
                    self.rest.append(dx)
                    self.compliance.append(STRUCT_COMPLIANCE)
                    self.active_flags.append(1)
                if (c + 1) in self.duplicated:
                    self.edges_a.append(dup_i)
                    self.edges_b.append(self.duplicated[c + 1])
                    self.rest.append(dx)
                    self.compliance.append(STRUCT_COMPLIANCE)
                    self.active_flags.append(1)

                # also retarget rendering faces below the cut row to use
                # the duplicate, so the rendered mesh actually splits
                for fi in range(len(self.faces)):
                    tri = list(self.faces[fi])
                    changed = False
                    for k in range(3):
                        if tri[k] == orig_i and tri[k] // nx > r_cut:
                            # (this check is always false since orig_i's
                            # row is exactly r_cut; retargeting instead
                            # happens by matching orig_i as a vertex of a
                            # face whose OTHER vertices are below r_cut)
                            pass
                    # simpler: any face that has orig_i as a vertex AND
                    # has at least one vertex below r_cut belongs to the
                    # lower material -> retarget its orig_i reference
                    if orig_i in tri and any((v // nx) > r_cut for v in tri):
                        tri[tri.index(orig_i)] = dup_i
                        self.faces[fi] = tuple(tri)
                        changed = True

            n = len(self.positions)
            self.x = wp.array(np.array(self.positions, dtype=np.float32), dtype=wp.vec3)
            v_np = self.v.numpy()
            v_np = np.vstack([v_np, np.zeros((n - v_np.shape[0], 3), dtype=np.float32)])
            self.v = wp.array(v_np, dtype=wp.vec3)

        n = len(self.positions)
        n_c = len(self.edges_a)
        x_pred = wp.zeros(n, dtype=wp.vec3)
        inv_mass_wp = wp.array(np.array(self.inv_mass, dtype=np.float32))
        idx_a_wp = wp.array(np.array(self.edges_a, dtype=np.int32))
        idx_b_wp = wp.array(np.array(self.edges_b, dtype=np.int32))
        rest_wp = wp.array(np.array(self.rest, dtype=np.float32))
        compliance_wp = wp.array(np.array(self.compliance, dtype=np.float32))
        active_wp = wp.array(np.array(self.active_flags, dtype=np.int32))
        lagrange = wp.zeros(n_c, dtype=wp.float32)
        corr = wp.zeros(n, dtype=wp.vec3)
        corr_w = wp.zeros(n, dtype=wp.float32)

        for _ in range(self.substeps):
            wp.launch(predict, dim=n, inputs=[self.x, self.v, x_pred, inv_mass_wp,
                                               self.gravity_vec, self.sub_dt])
            lagrange.zero_()
            for _it in range(self.iterations):
                corr.zero_()
                corr_w.zero_()
                wp.launch(solve_distance, dim=n_c,
                          inputs=[x_pred, inv_mass_wp, idx_a_wp, idx_b_wp, rest_wp,
                                  compliance_wp, active_wp, self.sub_dt, lagrange, corr, corr_w])
                wp.launch(apply_correction, dim=n, inputs=[x_pred, corr, corr_w])
            wp.launch(update_velocity, dim=n, inputs=[self.x, x_pred, self.v, inv_mass_wp, self.sub_dt])
            wp.launch(clamp_velocity, dim=n, inputs=[self.v, self.max_speed])

        pos_np = self.x.numpy()
        for c, ci in list(self.cohesive_constraint_idx.items()):
            if self.active_flags[ci] == 0:
                continue
            a, b = self.edges_a[ci], self.edges_b[ci]
            if np.linalg.norm(pos_np[a] - pos_np[b]) >= delta_c:
                self.active_flags[ci] = 0


def main():
    stage = omni.usd.get_context().get_stage()

    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    mesh_path = "/World/TissuePad"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.2, 0.0))

    UsdGeom.Camera.Define(stage, "/World/Camera")

    sim = CuttingSim()
    n_faces = len(sim.faces)
    mesh.GetFaceVertexCountsAttr().Set([3] * n_faces)

    def read_knife_target():
        if not os.path.exists(SHARED_STATE_FILE):
            return -1
        try:
            with open(SHARED_STATE_FILE) as f:
                data = json.load(f)
            return int(data.get("knife_col", -1))
        except (json.JSONDecodeError, ValueError):
            return sim.knife_col  # keep previous value if mid-write

    print("Isaac Sim viewer running. Start mouse_control.py in another "
          "terminal now, and drag across it to cut.")

    while simulation_app.is_running():
        target = read_knife_target()
        if target > sim.knife_col:
            sim.set_knife_target(target)

        sim.step()

        pos_np = sim.x.numpy()
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pos_np]))
        face_indices = [i for tri in sim.faces for i in tri]
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)

        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
