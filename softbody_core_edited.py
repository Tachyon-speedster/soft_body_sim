from __future__ import annotations

import math
import numpy as np
import warp as wp
import carb
import omni.usd
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

# ---------------------------------------------------------------------------
# USD prim paths
# ---------------------------------------------------------------------------
SOFT_BODY_PRIM_PATH = "/World/WarpSoftBody"
BASE_PRIM_PATH      = "/World/WarpBase"
PROBE_PRIM_PATH     = "/World/WarpProbe"
GROUND_PRIM_PATH    = "/World/WarpGroundViz"
LIGHT_PRIM_PATH     = "/World/WarpDomeLight"

_OWN_PATHS = {SOFT_BODY_PRIM_PATH, GROUND_PRIM_PATH,
              LIGHT_PRIM_PATH, "/World/Ground"}

# ---------------------------------------------------------------------------
# Geometry — all in metres
# ---------------------------------------------------------------------------
GROUND_Z = 0.0

# Rigid base: 15x15cm footprint, 5cm tall, sitting on ground
BASE_HALF_X  = 0.075   # 15cm / 2
BASE_HALF_Y  = 0.075
BASE_HALF_Z  = 0.025   # 5cm / 2
BASE_CENTER  = (0.0, 0.0, BASE_HALF_Z)   # bottom face at Z=0

# Soft body: 10x10cm footprint, 2cm tall, resting on top of base
SOFT_HALF_X  = 0.05    # 10cm / 2
SOFT_HALF_Y  = 0.05
SOFT_HALF_Z  = 0.01    # 2cm / 2
# Center Z = top of base + soft half-height
SOFT_CENTER  = (0.0, 0.0, BASE_HALF_Z * 2 + SOFT_HALF_Z)

# Resolution: more particles along XY (flat face) than Z (thin dim)
SOFT_RES_X   = 10   # particles along X → spacing = 2*0.05/9 ≈ 1.1 cm
SOFT_RES_Y   = 10   # particles along Y
SOFT_RES_Z   = 4    # particles along Z (thin) → spacing = 2*0.01/3 ≈ 6.7 mm
# Total particles: 10*10*4 = 400
# Total tets: (9*9*3)*6 = 1458

# Probe — pen-shaped cuboid beside the soft body
PROBE_HALF_X = 0.005
PROBE_HALF_Y = 0.005
PROBE_HALF_Z = 0.005
PROBE_CENTER = (0.15, 0.0, BASE_HALF_Z * 2 + PROBE_HALF_Z)
PROBE_COLOR  = np.array([0.75, 0.45, 0.15])

# Keep PIPE_* aliases for gather_colliders compatibility
PIPE_PRIM_PATH       = PROBE_PRIM_PATH
PIPE_RADIUS          = max(PROBE_HALF_X, PROBE_HALF_Y)
PIPE_COLLIDE_RADIUS  = PIPE_RADIUS
PIPE_CENTER          = PROBE_CENTER
PIPE_AXIS            = (1.0, 1.0, 1.0)
PIPE_HALF_LEN        = PROBE_HALF_Z

SKIN = 0.005   # 1 mm collision skin

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT            = 1.0 / 60.0
SUBSTEPS      = 12
SOLVER_ITERS  = 8   # moderate reduction from original 10 -- see note below

# ---------------------------------------------------------------------------
# Cutting constants (virtual-node duplication + breakable cohesive constraint)
# ---------------------------------------------------------------------------
CUT_DELTA_C          = 0.0015   # 1.5mm -- separation at which a cut interface
                                  # fully breaks. Tune against real silicone.
CUT_COHESIVE_STIFFNESS = 0.98    # position-correction stiffness for the
                                  # cohesive constraint while still bonded
                                  # (same [0,1] convention as k_edge/k_vol above)

# ---------------------------------------------------------------------------
# Shape type constants
# ---------------------------------------------------------------------------
SHAPE_SPHERE   = 0
SHAPE_BOX      = 1
SHAPE_CAPSULE  = 2
SHAPE_CYLINDER = 3
SHAPE_CONE     = 4
SHAPE_MESH     = 5


# ===========================================================================
# Warp kernels
# ===========================================================================

@wp.kernel
def integrate(
    pos:      wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    gravity:  wp.vec3,
    dt:       wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        pred[tid] = pos[tid]
        return
    v = vel[tid] + gravity * dt
    pred[tid] = pos[tid] + v * dt


@wp.kernel
def zero_corrections(
    corr:       wp.array(dtype=wp.vec3),
    corr_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    corr[tid]       = wp.vec3(0.0, 0.0, 0.0)
    corr_count[tid] = 0


@wp.kernel
def solve_distance_constraints(
    pred:        wp.array(dtype=wp.vec3),
    inv_mass:    wp.array(dtype=wp.float32),
    spring_i:    wp.array(dtype=wp.int32),
    spring_j:    wp.array(dtype=wp.int32),
    rest_length: wp.array(dtype=wp.float32),
    stiffness:   wp.array(dtype=wp.float32),
    corr:        wp.array(dtype=wp.vec3),
    corr_count:  wp.array(dtype=wp.int32),
):
    tid   = wp.tid()
    i     = spring_i[tid]
    j     = spring_j[tid]
    wi    = inv_mass[i]
    wj    = inv_mass[j]
    w_sum = wi + wj
    if w_sum == 0.0:
        return
    delta = pred[i] - pred[j]
    dist  = wp.length(delta)
    if dist < 1.0e-6:
        return
    n = delta / dist
    c = dist - rest_length[tid]
    s = -stiffness[tid] * c / w_sum
    wp.atomic_add(corr, i,  n * (wi *  s))
    wp.atomic_add(corr, j,  n * (-wj * s))
    wp.atomic_add(corr_count, i, 1)
    wp.atomic_add(corr_count, j, 1)


@wp.kernel
def apply_corrections(
    pred:       wp.array(dtype=wp.vec3),
    corr:       wp.array(dtype=wp.vec3),
    corr_count: wp.array(dtype=wp.int32),
):
    tid   = wp.tid()
    count = corr_count[tid]
    if count > 0:
        pred[tid] = pred[tid] + corr[tid] / float(count)


@wp.kernel
def solve_tet_volume_constraints(
    pred:        wp.array(dtype=wp.vec3),
    inv_mass:    wp.array(dtype=wp.float32),
    tet_a:       wp.array(dtype=wp.int32),
    tet_b:       wp.array(dtype=wp.int32),
    tet_c:       wp.array(dtype=wp.int32),
    tet_d:       wp.array(dtype=wp.int32),
    rest_volume: wp.array(dtype=wp.float32),
    stiffness:   wp.array(dtype=wp.float32),
    corr:        wp.array(dtype=wp.vec3),
    corr_count:  wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    ia  = tet_a[tid];  ib = tet_b[tid]
    ic  = tet_c[tid];  id_ = tet_d[tid]
    wa  = inv_mass[ia]; wb = inv_mass[ib]
    wc  = inv_mass[ic]; wd = inv_mass[id_]
    if wa + wb + wc + wd == 0.0:
        return
    pa = pred[ia]; pb = pred[ib]; pc = pred[ic]; pd = pred[id_]
    e1 = pb - pa;  e2 = pc - pa;  e3 = pd - pa
    vol  = wp.dot(e1, wp.cross(e2, e3)) / 6.0
    c    = vol - rest_volume[tid]
    grad_a = wp.cross(pd - pb, pc - pb) / 6.0
    grad_b = wp.cross(pc - pa, pd - pa) / 6.0
    grad_c = wp.cross(pd - pa, pb - pa) / 6.0
    grad_d = wp.cross(pb - pa, pc - pa) / 6.0
    denom  = (wa * wp.dot(grad_a, grad_a) + wb * wp.dot(grad_b, grad_b) +
              wc * wp.dot(grad_c, grad_c) + wd * wp.dot(grad_d, grad_d))
    if denom < 1.0e-9:
        return
    lam = -stiffness[tid] * c / denom
    wp.atomic_add(corr, ia,  grad_a * (wa * lam))
    wp.atomic_add(corr, ib,  grad_b * (wb * lam))
    wp.atomic_add(corr, ic,  grad_c * (wc * lam))
    wp.atomic_add(corr, id_, grad_d * (wd * lam))
    wp.atomic_add(corr_count, ia, 1)
    wp.atomic_add(corr_count, ib, 1)
    wp.atomic_add(corr_count, ic, 1)
    wp.atomic_add(corr_count, id_, 1)


@wp.kernel
def collide_ground(
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    ground_z: wp.float32,
    friction: wp.float32,
    vel:      wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p = pred[tid]
    if p[2] < ground_z:
        pred[tid] = wp.vec3(p[0], p[1], ground_z)
        v = vel[tid]
        vz_neg = wp.min(v[2], 0.0)
        vel[tid] = wp.vec3(v[0] * friction, v[1] * friction, v[2] - vz_neg)


@wp.kernel
def pin_ground_contacts(
    pos:             wp.array(dtype=wp.vec3),
    pred:            wp.array(dtype=wp.vec3),
    inv_mass:        wp.array(dtype=wp.float32),
    ground_z:        wp.float32,
    static_friction: wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p = pred[tid]
    if p[2] > ground_z + 0.02:
        return
    q  = pos[tid]
    dx = (p[0] - q[0]) * (1.0 - static_friction)
    dy = (p[1] - q[1]) * (1.0 - static_friction)
    pred[tid] = wp.vec3(q[0] + dx, q[1] + dy, p[2])


@wp.kernel
def collide_sphere(
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    vel:      wp.array(dtype=wp.vec3),
    center:   wp.vec3,
    radius:   wp.float32,
    skin:     wp.float32,
    friction: wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    diff = pred[tid] - center
    dist = wp.length(diff)
    lim  = radius + skin
    if dist < lim and dist > 1.0e-6:
        n = diff / dist
        pred[tid] = center + n * lim
        v  = vel[tid]
        vn = wp.dot(v, n)
        if vn < 0.0:
            vel[tid] = v - n * (vn * (1.0 - friction))


@wp.kernel
def collide_box(
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    vel:      wp.array(dtype=wp.vec3),
    center:   wp.vec3,
    row0:     wp.vec3,
    row1:     wp.vec3,
    row2:     wp.vec3,
    half_x:   wp.float32,
    half_y:   wp.float32,
    half_z:   wp.float32,
    skin:     wp.float32,
    friction: wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p  = pred[tid]
    dp = p - center
    lx = wp.dot(dp, row0)
    ly = wp.dot(dp, row1)
    lz = wp.dot(dp, row2)
    hx = half_x + skin
    hy = half_y + skin
    hz = half_z + skin
    if lx > -hx and lx < hx and ly > -hy and ly < hy and lz > -hz and lz < hz:
        dx_neg = lx + hx;  dx_pos = hx - lx
        dy_neg = ly + hy;  dy_pos = hy - ly
        dz_neg = lz + hz;  dz_pos = hz - lz
        min_d = dx_neg
        nx = -1.0; ny = 0.0; nz = 0.0
        if dx_pos < min_d: min_d = dx_pos; nx =  1.0; ny = 0.0; nz = 0.0
        if dy_neg < min_d: min_d = dy_neg; nx =  0.0; ny = -1.0; nz = 0.0
        if dy_pos < min_d: min_d = dy_pos; nx =  0.0; ny =  1.0; nz = 0.0
        if dz_neg < min_d: min_d = dz_neg; nx =  0.0; ny =  0.0; nz = -1.0
        if dz_pos < min_d: min_d = dz_pos; nx =  0.0; ny =  0.0; nz =  1.0
        n_world = row0 * nx + row1 * ny + row2 * nz
        pred[tid] = p + n_world * min_d
        v  = vel[tid]
        vn = wp.dot(v, n_world)
        if vn < 0.0:
            vel[tid] = v - n_world * (vn * (1.0 - friction))


@wp.kernel
def collide_cylinder(
    pred:          wp.array(dtype=wp.vec3),
    inv_mass:      wp.array(dtype=wp.float32),
    vel:           wp.array(dtype=wp.vec3),
    pipe_center:   wp.vec3,
    pipe_axis:     wp.vec3,
    collide_radius: wp.float32,
    pipe_half_len: wp.float32,
    skin:          wp.float32,
    friction:      wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p       = pred[tid]
    dp      = p - pipe_center
    t_ax    = wp.clamp(wp.dot(dp, pipe_axis), -pipe_half_len, pipe_half_len)
    closest = pipe_center + pipe_axis * t_ax
    radial  = p - closest
    dist    = wp.length(radial)
    lim     = collide_radius + skin
    if dist < lim and dist > 1.0e-6:
        n = radial / dist
        pred[tid] = closest + n * lim
        v = vel[tid]; vn = wp.dot(v, n)
        if vn < 0.0: vel[tid] = v - n * (vn * (1.0 - friction))




@wp.kernel
def update_velocity(
    pos:      wp.array(dtype=wp.vec3),
    pred:     wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    inv_dt:   wp.float32,
    damping:  wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        vel[tid] = wp.vec3(0.0, 0.0, 0.0)
        pos[tid] = pred[tid]
        return
    new_vel  = (pred[tid] - pos[tid]) * inv_dt
    vel[tid] = new_vel * damping
    pos[tid] = pred[tid]


@wp.kernel
def apply_translation(
    pos:   wp.array(dtype=wp.vec3),
    pred:  wp.array(dtype=wp.vec3),
    delta: wp.vec3,
):
    tid      = wp.tid()
    pos[tid]  = pos[tid]  + delta
    pred[tid] = pred[tid] + delta


@wp.kernel
def apply_drag_translation(
    pos:    wp.array(dtype=wp.vec3),
    pred:   wp.array(dtype=wp.vec3),
    vel:    wp.array(dtype=wp.vec3),
    delta:  wp.vec3,
    inv_dt: wp.float32,
):
    tid      = wp.tid()
    pos[tid]  = pos[tid]  + delta
    pred[tid] = pred[tid] + delta
    vel[tid]  = delta * inv_dt


# ===========================================================================
# Stage collider scanner
# ===========================================================================

def _normalize3(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def gather_colliders(stage: Usd.Stage, skip_paths: set,
                     xform_cache: UsdGeom.XformCache):
    colliders = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        skip = False
        for own in skip_paths:
            if path == own or path.startswith(own + "/"):
                skip = True; break
        if skip:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat4  = xform_cache.GetLocalToWorldTransform(prim)
        trans = mat4.ExtractTranslation()
        cx, cy, cz = float(trans[0]), float(trans[1]), float(trans[2])

        if prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            r_attr = sphere.GetRadiusAttr()
            radius = float(r_attr.Get()) if r_attr else 0.5
            sx = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
            sy = math.sqrt(mat4[1][0]**2+mat4[1][1]**2+mat4[1][2]**2)
            sz = math.sqrt(mat4[2][0]**2+mat4[2][1]**2+mat4[2][2]**2)
            colliders.append({"shape": SHAPE_SPHERE,
                               "center": (cx,cy,cz),
                               "radius": radius*max(sx,sy,sz)})

        elif prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            s_attr = cube.GetSizeAttr()
            size   = float(s_attr.Get()) if s_attr else 1.0
            half   = size * 0.5
            sx = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
            sy = math.sqrt(mat4[1][0]**2+mat4[1][1]**2+mat4[1][2]**2)
            sz = math.sqrt(mat4[2][0]**2+mat4[2][1]**2+mat4[2][2]**2)
            row0 = Gf.Vec3f(mat4[0][0]/sx, mat4[0][1]/sx, mat4[0][2]/sx)
            row1 = Gf.Vec3f(mat4[1][0]/sy, mat4[1][1]/sy, mat4[1][2]/sy)
            row2 = Gf.Vec3f(mat4[2][0]/sz, mat4[2][1]/sz, mat4[2][2]/sz)
            colliders.append({"shape": SHAPE_BOX, "center": (cx,cy,cz),
                               "row0": row0, "row1": row1, "row2": row2,
                               "half_x": half*sx, "half_y": half*sy,
                               "half_z": half*sz})

        elif prim.IsA(UsdGeom.Capsule):
            cap    = UsdGeom.Capsule(prim)
            radius = float(cap.GetRadiusAttr().Get() or 0.5)
            height = float(cap.GetHeightAttr().Get() or 1.0)
            ax_tok = str(cap.GetAxisAttr().Get() or "Y")
            local_ax = (Gf.Vec3d(1,0,0) if ax_tok=="X" else
                        Gf.Vec3d(0,0,1) if ax_tok=="Z" else Gf.Vec3d(0,1,0))
            wax  = _normalize3(mat4.TransformDir(local_ax))
            sx   = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
            half_h = height * 0.5
            p0   = (cx-wax[0]*half_h, cy-wax[1]*half_h, cz-wax[2]*half_h)
            p1   = (cx+wax[0]*half_h, cy+wax[1]*half_h, cz+wax[2]*half_h)
            colliders.append({"shape": SHAPE_CAPSULE, "p0": p0, "p1": p1,
                               "radius": radius*sx})

        elif prim.IsA(UsdGeom.Cylinder):
            cyl    = UsdGeom.Cylinder(prim)
            radius = float(cyl.GetRadiusAttr().Get() or 0.5)
            height = float(cyl.GetHeightAttr().Get() or 1.0)
            ax_tok = str(cyl.GetAxisAttr().Get() or "Y")
            col0 = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
            col1 = math.sqrt(mat4[1][0]**2+mat4[1][1]**2+mat4[1][2]**2)
            col2 = math.sqrt(mat4[2][0]**2+mat4[2][1]**2+mat4[2][2]**2)
            if ax_tok=="X":
                local_ax=Gf.Vec3d(1,0,0); scale_h=col0; scale_r=(col1+col2)*0.5
            elif ax_tok=="Z":
                local_ax=Gf.Vec3d(0,0,1); scale_h=col2; scale_r=(col0+col1)*0.5
            else:
                local_ax=Gf.Vec3d(0,1,0); scale_h=col1; scale_r=(col0+col2)*0.5
            wax = _normalize3(mat4.TransformDir(local_ax))
            colliders.append({"shape": SHAPE_CYLINDER,
                               "center": (cx,cy,cz), "axis": wax,
                               "radius": radius*scale_r,
                               "collide_radius": radius*scale_r,
                               "half_len": height*0.5*scale_h})

        elif prim.IsA(UsdGeom.Cone):
            cone   = UsdGeom.Cone(prim)
            radius = float(cone.GetRadiusAttr().Get() or 0.5)
            height = float(cone.GetHeightAttr().Get() or 1.0)
            ax_tok = str(cone.GetAxisAttr().Get() or "Y")
            local_ax = (Gf.Vec3d(1,0,0) if ax_tok=="X" else
                        Gf.Vec3d(0,0,1) if ax_tok=="Z" else Gf.Vec3d(0,1,0))
            wax  = _normalize3(mat4.TransformDir(local_ax))
            sx   = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
            apex = (cx-wax[0]*height*0.5, cy-wax[1]*height*0.5,
                    cz-wax[2]*height*0.5)
            colliders.append({"shape": SHAPE_CONE, "apex": apex, "axis": wax,
                               "half_angle": math.atan2(radius*sx, height),
                               "height": height})

        elif prim.IsA(UsdGeom.Mesh):
            mesh     = UsdGeom.Mesh(prim)
            pts_attr = mesh.GetPointsAttr()
            if not pts_attr:
                continue
            lpts = np.array(pts_attr.Get(), dtype=np.float64)
            if len(lpts) < 3:
                continue
            M = np.array([
                [mat4[0][0],mat4[1][0],mat4[2][0],mat4[3][0]],
                [mat4[0][1],mat4[1][1],mat4[2][1],mat4[3][1]],
                [mat4[0][2],mat4[1][2],mat4[2][2],mat4[3][2]],
            ], dtype=np.float64)
            wpts = (M @ np.hstack([lpts, np.ones((len(lpts),1))]).T).T
            approx = "boundingSphere"
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mc  = UsdPhysics.MeshCollisionAPI(prim)
                att = mc.GetApproximationAttr()
                if att:
                    v = att.Get()
                    if v: approx = str(v)
            mn = wpts.min(0); mx = wpts.max(0)
            cen = (mn+mx)*0.5
            if approx in ("convexHull","convexDecomposition",
                          "boundingCube","none"):
                col0 = math.sqrt(mat4[0][0]**2+mat4[0][1]**2+mat4[0][2]**2)
                col1 = math.sqrt(mat4[1][0]**2+mat4[1][1]**2+mat4[1][2]**2)
                col2 = math.sqrt(mat4[2][0]**2+mat4[2][1]**2+mat4[2][2]**2)
                row0 = Gf.Vec3f(mat4[0][0]/max(col0,1e-9),
                                 mat4[0][1]/max(col0,1e-9),
                                 mat4[0][2]/max(col0,1e-9))
                row1 = Gf.Vec3f(mat4[1][0]/max(col1,1e-9),
                                 mat4[1][1]/max(col1,1e-9),
                                 mat4[1][2]/max(col1,1e-9))
                row2 = Gf.Vec3f(mat4[2][0]/max(col2,1e-9),
                                 mat4[2][1]/max(col2,1e-9),
                                 mat4[2][2]/max(col2,1e-9))
                li = np.column_stack([
                    wpts@np.array([row0[0],row0[1],row0[2]]),
                    wpts@np.array([row1[0],row1[1],row1[2]]),
                    wpts@np.array([row2[0],row2[1],row2[2]]),
                ])
                oc = li.mean(0)
                oh = (li.max(0)-li.min(0))*0.5
                obb_cx = oc[0]*row0[0]+oc[1]*row1[0]+oc[2]*row2[0]+cx
                obb_cy = oc[0]*row0[1]+oc[1]*row1[1]+oc[2]*row2[1]+cy
                obb_cz = oc[0]*row0[2]+oc[1]*row1[2]+oc[2]*row2[2]+cz
                colliders.append({"shape": SHAPE_BOX,
                                   "center": (obb_cx,obb_cy,obb_cz),
                                   "row0": row0, "row1": row1, "row2": row2,
                                   "half_x": float(oh[0]),
                                   "half_y": float(oh[1]),
                                   "half_z": float(oh[2])})
            else:
                d = np.linalg.norm(wpts - cen, axis=1)
                colliders.append({"shape": SHAPE_SPHERE,
                                   "center": (float(cen[0]),float(cen[1]),
                                              float(cen[2])),
                                   "radius": float(d.max())})
    return colliders


# ===========================================================================
# SoftBody — tet mesh XPBD, arbitrary box shape (half_x, half_y, half_z)
# ===========================================================================

class SoftBodyCube:
    """XPBD tet-mesh soft body with independent per-axis dimensions and
    resolution.  Works for any box aspect ratio — flat pads, cubes, rods.

    Topology: Freudenthal 6-tet-per-cell on a res_x × res_y × res_z grid.
    Constraints: tet-edge distance + tet volume preservation.
    Surface: boundary-face extraction (faces belonging to exactly one tet).
    """

    def __init__(
        self,
        center=(0.0, 0.0, 0.0),
        half_x=0.05,
        half_y=0.05,
        half_z=0.01,
        res_x=10,
        res_y=10,
        res_z=4,
        total_mass=1.0,
        k_edge=0.65,
        k_vol=0.6,
        device=None,
    ):
        self.device = device
        cx, cy, cz = center
        n = res_x * res_y * res_z

        # Stored for cutting: column-based indexing needs the grid shape
        # and per-particle mass (new duplicated particles need a mass too).
        self.res_x, self.res_y, self.res_z = res_x, res_y, res_z
        self.n_orig = n
        self._total_mass = total_mass
        self._k_edge = k_edge
        self._k_vol = k_vol
        self.cut_progress = -1          # no columns cut yet
        self.cohesive = []              # list of dicts: {a, b, spring_idx}
                                          # tracking each cut interface pair
                                          # still bonded, for break-checking

        # ── Particle grid ────────────────────────────────────────────────
        lx = np.linspace(-half_x, half_x, res_x, dtype=np.float64) + cx
        ly = np.linspace(-half_y, half_y, res_y, dtype=np.float64) + cy
        lz = np.linspace(-half_z, half_z, res_z, dtype=np.float64) + cz
        gx, gy, gz = np.meshgrid(lx, ly, lz, indexing="ij")
        positions = np.stack(
            [gx.flatten(), gy.flatten(), gz.flatten()], axis=1
        ).astype(np.float64)

        inv_mass = np.full(n, n / total_mass, dtype=np.float32)

        def vidx(ix, iy, iz):
            return ix * res_y * res_z + iy * res_z + iz

        # ── Weld bottom face (iz == 0) to the rigid base — no slip, no separation ──
        for ix in range(res_x):
            for iy in range(res_y):
                inv_mass[vidx(ix, iy, 0)] = 0.0

        # ── Tetrahedralization (Freudenthal 6-tet split) ─────────────────
        tets = []
        for ix in range(res_x - 1):
            for iy in range(res_y - 1):
                for iz in range(res_z - 1):
                    v000=vidx(ix,  iy,  iz  ); v100=vidx(ix+1,iy,  iz  )
                    v010=vidx(ix,  iy+1,iz  ); v110=vidx(ix+1,iy+1,iz  )
                    v001=vidx(ix,  iy,  iz+1); v101=vidx(ix+1,iy,  iz+1)
                    v011=vidx(ix,  iy+1,iz+1); v111=vidx(ix+1,iy+1,iz+1)
                    tets += [
                        (v000,v100,v110,v111),
                        (v000,v100,v101,v111),
                        (v000,v010,v110,v111),
                        (v000,v010,v011,v111),
                        (v000,v001,v101,v111),
                        (v000,v001,v011,v111),
                    ]
        tets = np.array(tets, dtype=np.int64)

        # Ensure positive volume
        def tet_vol(p, t):
            a,b,c_,d = p[t[:,0]],p[t[:,1]],p[t[:,2]],p[t[:,3]]
            return np.einsum('ij,ij->i',b-a,np.cross(c_-a,d-a))/6.0
        vols = tet_vol(positions, tets)
        flip = vols < 0
        if flip.any():
            tets[flip,0], tets[flip,1] = tets[flip,1].copy(), tets[flip,0].copy()
            vols = tet_vol(positions, tets)

        self.num_tets = len(tets)

        # ── Unique tet edges → distance constraints ───────────────────────
        edge_set = set()
        for t in tets:
            for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                i,j = int(t[a]), int(t[b])
                if i>j: i,j=j,i
                edge_set.add((i,j))
        si = np.array([e[0] for e in edge_set], dtype=np.int32)
        sj = np.array([e[1] for e in edge_set], dtype=np.int32)
        sr = np.linalg.norm(positions[si]-positions[sj], axis=1).astype(np.float32)
        sk = np.full(len(si), k_edge, dtype=np.float32)
        self.num_springs = len(si)

        # ── Boundary surface extraction ───────────────────────────────────
        tet_faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
        fc = {}; fw = {}
        for t in tets:
            for fa,fb,fc_ in tet_faces:
                ia,ib,ic = int(t[fa]),int(t[fb]),int(t[fc_])
                key = tuple(sorted((ia,ib,ic)))
                fc[key] = fc.get(key,0) + 1
                if key not in fw: fw[key] = (ia,ib,ic)
        boundary = [fw[k] for k,v in fc.items() if v==1]
        pos32 = positions.astype(np.float32)
        gcen  = pos32.mean(0)
        oriented = []
        for ia,ib,ic in boundary:
            pa,pb,pc = pos32[ia],pos32[ib],pos32[ic]
            fn  = np.cross(pb-pa, pc-pa)
            mid = (pa+pb+pc)/3.0
            if np.dot(fn, mid-gcen) < 0.0:
                ia,ib,ic = ia,ic,ib
            oriented.append((ia,ib,ic))
        self.tri_indices = np.array(oriented, dtype=np.int32).flatten()
        self.num_particles = n

        # ── Upload to GPU ─────────────────────────────────────────────────
        self.pos        = wp.array(pos32,                              dtype=wp.vec3,    device=device)
        self.pred       = wp.array(pos32.copy(),                       dtype=wp.vec3,    device=device)
        self.vel        = wp.zeros(n,                                   dtype=wp.vec3,    device=device)
        self.inv_mass   = wp.array(inv_mass,                            dtype=wp.float32, device=device)
        self.spring_i   = wp.array(si,                                  dtype=wp.int32,   device=device)
        self.spring_j   = wp.array(sj,                                  dtype=wp.int32,   device=device)
        self.rest_length= wp.array(sr,                                  dtype=wp.float32, device=device)
        self.stiffness  = wp.array(sk,                                  dtype=wp.float32, device=device)
        self.tet_a      = wp.array(tets[:,0].astype(np.int32),         dtype=wp.int32,   device=device)
        self.tet_b      = wp.array(tets[:,1].astype(np.int32),         dtype=wp.int32,   device=device)
        self.tet_c      = wp.array(tets[:,2].astype(np.int32),         dtype=wp.int32,   device=device)
        self.tet_d      = wp.array(tets[:,3].astype(np.int32),         dtype=wp.int32,   device=device)
        self.tet_vol    = wp.array(vols.astype(np.float32),            dtype=wp.float32, device=device)
        self.tet_stiff  = wp.array(np.full(self.num_tets,k_vol,
                                           dtype=np.float32),          dtype=wp.float32, device=device)
        self.corr       = wp.zeros(n, dtype=wp.vec3,  device=device)
        self.corr_count = wp.zeros(n, dtype=wp.int32, device=device)

        # ── Python-side mutable mirrors, for cutting ──────────────────────
        # (the wp arrays above are fixed-size; cutting adds particles and
        # constraints, so we keep growable Python lists here and rebuild
        # the GPU arrays from them whenever topology changes)
        self._pos_list      = pos32.tolist()
        self._inv_mass_list = inv_mass.tolist()
        self._tets_list     = tets.tolist()
        self._edge_i_list   = si.tolist()
        self._edge_j_list   = sj.tolist()
        self._edge_rest_list  = sr.tolist()
        self._edge_stiff_list = sk.tolist()
        self._tet_vol_list   = vols.astype(np.float32).tolist()
        self._tet_stiff_list = [float(k_vol)] * self.num_tets
        self._tri_list       = [list(t) for t in oriented]

    def centroid(self):
        p = self.pos.numpy(); c = p.mean(0)
        return float(c[0]), float(c[1]), float(c[2])

    # ------------------------------------------------------------------
    # Cutting: virtual-node duplication along constant-X columns, with a
    # breakable cohesive constraint at each duplicated interface.
    #
    # This is the same algorithm validated in isolation (small tet-grid
    # test) before being wired in here: a tet/edge/triangle belongs to the
    # "far" side of a cut at column `ix` if its minimum column index is
    # exactly `ix` (the cell block starting at that column); only THAT
    # element's vertices at column `ix` get retargeted to the duplicate,
    # not the whole element.
    # ------------------------------------------------------------------
    def _vidx(self, ix, iy, iz):
        return ix * self.res_y * self.res_z + iy * self.res_z + iz

    def _column_of(self, v):
        """Column index for an ORIGINAL vertex only (v < n_orig).
        Duplicated vertices don't have a meaningful column -- callers
        must check v < self.n_orig before calling this."""
        return v // (self.res_y * self.res_z)

    def advance_cut(self, target_ix: int):
        """Advance the cut up to and including column `target_ix` (only
        moves forward; no-op if target_ix <= self.cut_progress)."""
        if target_ix <= self.cut_progress:
            return
        if target_ix >= self.res_x - 1:
            target_ix = self.res_x - 2   # never cut the very last column --
                                          # nothing exists beyond it to sever

        # Sync current simulated state from the GPU before mutating topology
        # (self._pos_list from __init__ is stale after any physics steps).
        cur_pos = self.pos.numpy()
        cur_vel = self.vel.numpy()
        self._pos_list = cur_pos.tolist()
        vel_list = cur_vel.tolist()
        # extend vel_list to match if particles were added by a previous
        # cut in the same frame batch (shouldn't happen, but be safe)
        while len(vel_list) < len(self._pos_list):
            vel_list.append([0.0, 0.0, 0.0])

        n_orig = self.n_orig

        for ix in range(self.cut_progress + 1, target_ix + 1):
            col_verts = [self._vidx(ix, iy, iz)
                         for iy in range(self.res_y) for iz in range(self.res_z)]
            col_dup = {}
            for v in col_verts:
                dup_i = len(self._pos_list)
                self._pos_list.append(list(self._pos_list[v]))
                vel_list.append(list(vel_list[v]))
                self._inv_mass_list.append(self._inv_mass_list[v])
                col_dup[v] = dup_i

                # breakable cohesive constraint between original and duplicate
                spring_idx = len(self._edge_i_list)
                self._edge_i_list.append(v)
                self._edge_j_list.append(dup_i)
                self._edge_rest_list.append(0.0)
                self._edge_stiff_list.append(CUT_COHESIVE_STIFFNESS)
                self.cohesive.append({"a": v, "b": dup_i, "spring_idx": spring_idx})

            # retarget tets: only tets whose cell block STARTS at this
            # column (min original column == ix) get their column-ix
            # vertices swapped to the duplicate
            for ti in range(len(self._tets_list)):
                tet = self._tets_list[ti]
                cols = [self._column_of(v) if v < n_orig else -1 for v in tet]
                if -1 in cols:
                    continue  # already touched by an earlier cut this call
                if min(cols) == ix:
                    self._tets_list[ti] = [col_dup.get(v, v) for v in tet]

            # retarget edges (structural springs) the same way
            for ei in range(len(self._edge_i_list) - len(col_verts)):
                # (skip the cohesive springs we just appended this iteration)
                i_, j_ = self._edge_i_list[ei], self._edge_j_list[ei]
                if i_ >= n_orig or j_ >= n_orig:
                    continue
                cols = [self._column_of(i_), self._column_of(j_)]
                if min(cols) == ix:
                    self._edge_i_list[ei] = col_dup.get(i_, i_)
                    self._edge_j_list[ei] = col_dup.get(j_, j_)

            # retarget render-mesh triangles the same way, so the visual
            # split actually shows
            for fi in range(len(self._tri_list)):
                tri = self._tri_list[fi]
                cols = [self._column_of(v) if v < n_orig else -1 for v in tri]
                if -1 in cols:
                    continue
                if min(cols) == ix:
                    self._tri_list[fi] = [col_dup.get(v, v) for v in tri]

        self.cut_progress = target_ix
        self._rebuild_gpu_arrays(vel_list)

    def _check_cohesive_breaks(self):
        """Check all still-bonded cut interfaces; remove any that have
        separated past CUT_DELTA_C. Rebuilds GPU arrays only if something
        actually broke this step (cheap check otherwise)."""
        if not self.cohesive:
            return
        pos_np = self.pos.numpy()
        still_bonded = []
        broken_spring_idxs = set()
        for c in self.cohesive:
            dist = float(np.linalg.norm(pos_np[c["a"]] - pos_np[c["b"]]))
            if dist >= CUT_DELTA_C:
                broken_spring_idxs.add(c["spring_idx"])
            else:
                still_bonded.append(c)

        if not broken_spring_idxs:
            return

        self.cohesive = still_bonded
        keep = [i for i in range(len(self._edge_i_list)) if i not in broken_spring_idxs]
        self._edge_i_list     = [self._edge_i_list[i] for i in keep]
        self._edge_j_list     = [self._edge_j_list[i] for i in keep]
        self._edge_rest_list  = [self._edge_rest_list[i] for i in keep]
        self._edge_stiff_list = [self._edge_stiff_list[i] for i in keep]
        # cohesive spring_idx values are now stale -- remap them
        remap = {old_i: new_i for new_i, old_i in enumerate(keep)}
        for c in self.cohesive:
            c["spring_idx"] = remap[c["spring_idx"]]

        self._rebuild_gpu_arrays(self.vel.numpy().tolist())

    def _rebuild_gpu_arrays(self, vel_list):
        """Re-upload all GPU arrays from the current Python-side lists.
        Called after advance_cut() or a cohesive break changes topology."""
        n = len(self._pos_list)
        self.num_particles = n
        self.num_springs = len(self._edge_i_list)
        self.num_tets = len(self._tets_list)

        pos_np = np.array(self._pos_list, dtype=np.float32)
        vel_np = np.array(vel_list, dtype=np.float32)
        tets_np = np.array(self._tets_list, dtype=np.int64)

        self.pos        = wp.array(pos_np,                                   dtype=wp.vec3,    device=self.device)
        self.pred       = wp.array(pos_np.copy(),                            dtype=wp.vec3,    device=self.device)
        self.vel        = wp.array(vel_np,                                   dtype=wp.vec3,    device=self.device)
        self.inv_mass   = wp.array(np.array(self._inv_mass_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.spring_i   = wp.array(np.array(self._edge_i_list, dtype=np.int32),     dtype=wp.int32,   device=self.device)
        self.spring_j   = wp.array(np.array(self._edge_j_list, dtype=np.int32),     dtype=wp.int32,   device=self.device)
        self.rest_length= wp.array(np.array(self._edge_rest_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.stiffness  = wp.array(np.array(self._edge_stiff_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.tet_a      = wp.array(tets_np[:, 0].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_b      = wp.array(tets_np[:, 1].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_c      = wp.array(tets_np[:, 2].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_d      = wp.array(tets_np[:, 3].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_vol    = wp.array(np.array(self._tet_vol_list, dtype=np.float32),  dtype=wp.float32, device=self.device)
        self.tet_stiff  = wp.array(np.array(self._tet_stiff_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.corr       = wp.zeros(n, dtype=wp.vec3,  device=self.device)
        self.corr_count = wp.zeros(n, dtype=wp.int32, device=self.device)

        self.tri_indices = np.array(self._tri_list, dtype=np.int32).flatten()

    def teleport(self, delta):
        dx,dy,dz = delta
        if abs(dx)<1e-6 and abs(dy)<1e-6 and abs(dz)<1e-6: return
        wp.launch(apply_translation, dim=self.num_particles,
                  inputs=[self.pos,self.pred,
                          wp.vec3(float(dx),float(dy),float(dz))],
                  device=self.device)

    def drag(self, delta, dt=DT):
        dx,dy,dz = delta
        if abs(dx)<1e-6 and abs(dy)<1e-6 and abs(dz)<1e-6: return
        wp.launch(apply_drag_translation, dim=self.num_particles,
                  inputs=[self.pos,self.pred,self.vel,
                          wp.vec3(float(dx),float(dy),float(dz)),
                          float(1.0/max(dt,1e-6))],
                  device=self.device)

    def _dispatch_collider(self, c: dict, friction: float):
        sh = c["shape"]
        if sh == SHAPE_SPHERE:
            wp.launch(collide_sphere, dim=self.num_particles,
                      inputs=[self.pred,self.inv_mass,self.vel,
                               wp.vec3(*c["center"]),float(c["radius"]),
                               float(SKIN),float(friction)],
                      device=self.device)
        elif sh == SHAPE_BOX:
            r0,r1,r2 = c["row0"],c["row1"],c["row2"]
            wp.launch(collide_box, dim=self.num_particles,
                      inputs=[self.pred,self.inv_mass,self.vel,
                               wp.vec3(*c["center"]),
                               wp.vec3(float(r0[0]),float(r0[1]),float(r0[2])),
                               wp.vec3(float(r1[0]),float(r1[1]),float(r1[2])),
                               wp.vec3(float(r2[0]),float(r2[1]),float(r2[2])),
                               float(c["half_x"]),float(c["half_y"]),float(c["half_z"]),
                               float(SKIN),float(friction)],
                      device=self.device)
        elif sh == SHAPE_CAPSULE:
            wp.launch(collide_capsule, dim=self.num_particles,
                      inputs=[self.pred,self.inv_mass,self.vel,
                               wp.vec3(*c["p0"]),wp.vec3(*c["p1"]),
                               float(c["radius"]),float(SKIN),float(friction)],
                      device=self.device)
        elif sh == SHAPE_CYLINDER:
            cr = float(c.get("collide_radius", c["radius"]))
            wp.launch(collide_cylinder, dim=self.num_particles,
                      inputs=[self.pred,self.inv_mass,self.vel,
                               wp.vec3(*c["center"]),wp.vec3(*c["axis"]),
                               cr,float(c["half_len"]),
                               float(SKIN),float(friction)],
                      device=self.device)
        elif sh == SHAPE_CONE:
            wp.launch(collide_cone, dim=self.num_particles,
                      inputs=[self.pred,self.inv_mass,self.vel,
                               wp.vec3(*c["apex"]),wp.vec3(*c["axis"]),
                               float(c["half_angle"]),float(c["height"]),
                               float(SKIN),float(friction)],
                      device=self.device)

    def step(
        self,
        dt=DT,
        substeps=SUBSTEPS,
        solver_iters=SOLVER_ITERS,
        gravity=(0.0, 0.0, -9.81),
        damping=0.995,
        ground_z=None,        # pass float to enable ground collision
        base_box=None,        # dict with SHAPE_BOX params for rigid base
        friction=0.85,
        static_friction=0.98,
        colliders=None,
    ):
        sub_dt    = dt / substeps
        gv        = wp.vec3(*gravity)
        colliders = colliders or []

        for _ in range(substeps):
            wp.launch(integrate, dim=self.num_particles,
                      inputs=[self.pos,self.vel,self.pred,
                               self.inv_mass,gv,sub_dt],
                      device=self.device)

            for _ in range(solver_iters):
                wp.launch(zero_corrections, dim=self.num_particles,
                          inputs=[self.corr,self.corr_count],
                          device=self.device)
                wp.launch(solve_distance_constraints, dim=self.num_springs,
                          inputs=[self.pred,self.inv_mass,
                                  self.spring_i,self.spring_j,
                                  self.rest_length,self.stiffness,
                                  self.corr,self.corr_count],
                          device=self.device)
                wp.launch(solve_tet_volume_constraints, dim=self.num_tets,
                          inputs=[self.pred,self.inv_mass,
                                  self.tet_a,self.tet_b,self.tet_c,self.tet_d,
                                  self.tet_vol,self.tet_stiff,
                                  self.corr,self.corr_count],
                          device=self.device)
                wp.launch(apply_corrections, dim=self.num_particles,
                          inputs=[self.pred,self.corr,self.corr_count],
                          device=self.device)

                # Collide with rigid base (always present, treated like a collider)
                if base_box is not None:
                    self._dispatch_collider(base_box, friction)

                if ground_z is not None:
                    wp.launch(collide_ground, dim=self.num_particles,
                              inputs=[self.pred,self.inv_mass,
                                      float(ground_z),float(friction),self.vel],
                              device=self.device)
                    wp.launch(pin_ground_contacts, dim=self.num_particles,
                              inputs=[self.pos,self.pred,self.inv_mass,
                                      float(ground_z),float(static_friction)],
                              device=self.device)

                for col in colliders:
                    self._dispatch_collider(col, friction)

            wp.launch(update_velocity, dim=self.num_particles,
                      inputs=[self.pos,self.pred,self.vel,
                               self.inv_mass,1.0/sub_dt,damping],
                      device=self.device)


# ===========================================================================
# Helpers
# ===========================================================================

def _vec3f_list(arr: np.ndarray):
    return [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in arr]


# ===========================================================================
# WarpSoftBodySim — Isaac Sim scenario
# ===========================================================================

class WarpSoftBodySim:
    """Flat soft-body pad (skin-colored) resting on a rigid black base.

    Probe (mouse/viewport-dragged) pokes the top surface.
    Any other prim with CollisionAPI also interacts via gather_colliders.
    """

    def __init__(self):
        self._cube               = None
        self._cube_mesh          = None
        self._probe              = None
        self._probe_translate_op = None
        self._base_box           = None   # SHAPE_BOX dict for the rigid base
        self._ground             = None
        self._device             = None
        self._xform_cache        = None
        self._probe_last_good    = None   # last accepted probe world pos,
                                           # used to clamp per-frame movement

    # ------------------------------------------------------------------
    def load_example_assets(self):
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Dome light
        dome = UsdLux.DomeLight.Define(stage, LIGHT_PRIM_PATH)
        dome.CreateIntensityAttr(500.0)

        # Ground visual
        size = 0.5
        gnd = UsdGeom.Mesh.Define(stage, GROUND_PRIM_PATH)
        gnd.CreatePointsAttr([
            Gf.Vec3f(-size,-size,GROUND_Z), Gf.Vec3f(size,-size,GROUND_Z),
            Gf.Vec3f(size,size,GROUND_Z),   Gf.Vec3f(-size,size,GROUND_Z),
        ])
        gnd.CreateFaceVertexCountsAttr([4])
        gnd.CreateFaceVertexIndicesAttr([0,1,2,3])
        gnd.CreateDisplayColorAttr([(0.25,0.25,0.25)])

        # Rigid base — black UsdGeom.Cube with RigidBody + Collision
        base = UsdGeom.Cube.Define(stage, BASE_PRIM_PATH)
        base.CreateSizeAttr(1.0)
        base.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.05)])  # black
        xfb = UsdGeom.Xformable(base.GetPrim())
        xfb.AddTranslateOp().Set(Gf.Vec3d(*BASE_CENTER))
        xfb.AddScaleOp().Set(Gf.Vec3f(
            BASE_HALF_X * 2, BASE_HALF_Y * 2, BASE_HALF_Z * 2))
        # Static rigid body (kinematic, no gravity)
        rba = UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        rba.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(base.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("convexHull")

        # Pre-build the base collider dict — never changes, no need to scan it
        self._base_box = {
            "shape":  SHAPE_BOX,
            "center": BASE_CENTER,
            "row0":   Gf.Vec3f(1.0, 0.0, 0.0),
            "row1":   Gf.Vec3f(0.0, 1.0, 0.0),
            "row2":   Gf.Vec3f(0.0, 0.0, 1.0),
            "half_x": BASE_HALF_X,
            "half_y": BASE_HALF_Y,
            "half_z": BASE_HALF_Z,
        }

        # Probe — pen-shaped, kinematic rigid body
        probe = UsdGeom.Cube.Define(stage, PROBE_PRIM_PATH)
        probe.CreateSizeAttr(0.5)
        probe.CreateDisplayColorAttr([Gf.Vec3f(*PROBE_COLOR.tolist())])
        xfp = UsdGeom.Xformable(probe.GetPrim())
        self._probe_translate_op = xfp.AddTranslateOp()
        self._probe_translate_op.Set(Gf.Vec3d(*PROBE_CENTER))
        xfp.AddScaleOp().Set(Gf.Vec3f(
            PROBE_HALF_X*2, PROBE_HALF_Y*2, PROBE_HALF_Z*2))
        rba2 = UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        rba2.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(probe.GetPrim()).CreateApproximationAttr("convexHull")
        self._probe = probe

        # Soft body render mesh (points written each frame)
        cb = UsdGeom.Mesh.Define(stage, SOFT_BODY_PRIM_PATH)
        cb.CreateDoubleSidedAttr(True)
        cb.CreateDisplayColorAttr([(0.41, 0.22, 0.16)])  # skin tone
        self._cube_mesh = cb

        self._ground = GroundPlane("/World/Ground", visible=False)
        return (self._ground,)

    # ------------------------------------------------------------------
    def setup(self):
        set_camera_view(
            eye=[0.3, 0.25, 0.25],
            target=[0.0, 0.0, 0.06],
            camera_prim_path="/OmniverseKit_Persp",
        )
        wp.init()
        self._device      = wp.get_preferred_device()
        self._xform_cache = UsdGeom.XformCache()
        self._spawn()

    def reset(self):
        self._spawn()

    # ------------------------------------------------------------------
    def update(self, step: float):
        """Thin wrapper around _update_impl(): any exception during a
        frame's update -- USD xformOp quirks, transient stage state, etc --
        gets logged and skipped rather than propagating up and silently
        killing physics stepping for the rest of the session (which is
        what happened before: an uncaught exception here looks exactly
        like "the sim just stopped simulating")."""
        try:
            return self._update_impl(step)
        except Exception as e:
            carb.log_warn(f"[WarpSoftBody] update() failed this frame, "
                           f"skipping: {e}")
            return False

    def _update_impl(self, step: float):
        if self._cube is None:
            return False

        stage = omni.usd.get_context().get_stage()
        DEAD  = 5e-4
        MAX_DRAG = 0.02   # 2cm/frame safety clamp -- prevents viewport
                          # gizmo grid-snap (often 1 unit = 1m by default)
                          # from ever producing a huge, physically
                          # impossible jump regardless of root cause

        def _clamp_delta(dx, dy, dz):
            mag = math.sqrt(dx*dx + dy*dy + dz*dz)
            if mag > MAX_DRAG:
                s = MAX_DRAG / mag
                return dx * s, dy * s, dz * s
            return dx, dy, dz

        # Soft body drag
        soft_body_read = self._prim_world_translation(SOFT_BODY_PRIM_PATH)
        if soft_body_read is not None:
            px, py, pz = soft_body_read
            if abs(px)>DEAD or abs(py)>DEAD or abs(pz)>DEAD:
                px, py, pz = _clamp_delta(px, py, pz)
                self._cube.drag((px,py,pz), dt=DT)
                self._clear_prim_xform(SOFT_BODY_PRIM_PATH)

        # Probe viewport drag -- the probe is a simple kinematic collider,
        # not something that needs delta-accumulation like the soft body
        # does. Just read its current world position directly and use it.
        #
        # IMPORTANT: we read the position, then explicitly clear ALL
        # transform ops on the prim, THEN set _probe_translate_op to the
        # value we read. This matters because we don't know for certain
        # whether Isaac Sim's move gizmo edits _probe_translate_op in
        # place, or stacks a second translate op on top of it. If it's
        # the latter and we only .Set() the first op, the second op would
        # still be sitting there non-zero, and next frame's world reading
        # would include it AGAIN on top of the value we just wrote --
        # compounding a little further every single frame. Clearing
        # everything down to zero first, then setting the one canonical
        # op, is safe regardless of which behavior the gizmo actually has.
        if self._probe is not None:
            probe_read = self._prim_world_translation(PROBE_PRIM_PATH)
            if probe_read is None:
                # Couldn't get a valid read this frame (e.g. mid-drag while
                # the gizmo is rewriting xformOps) -- keep the probe exactly
                # where it already is rather than guessing, so it never
                # snaps to the origin.
                pass
            else:
                vx, vy, vz = probe_read
                if self._probe_last_good is None:
                    # First valid read since spawn -- accept it outright.
                    self._probe_last_good = (vx, vy, vz)
                else:
                    lx, ly, lz = self._probe_last_good
                    dx, dy, dz = _clamp_delta(vx - lx, vy - ly, vz - lz)
                    vx, vy, vz = lx + dx, ly + dy, lz + dz
                    self._probe_last_good = (vx, vy, vz)
                self._clear_prim_xform(PROBE_PRIM_PATH)
                self._probe_translate_op.Set(Gf.Vec3d(vx, vy, vz))

        # Gather external colliders (skip base and probe — handled separately)
        self._xform_cache.Clear()
        skip = _OWN_PATHS | {SOFT_BODY_PRIM_PATH, PROBE_PRIM_PATH, BASE_PRIM_PATH}
        colliders = gather_colliders(stage, skip, self._xform_cache)

        # Probe collider from translate op
        probe_world = None
        if self._probe is not None:
            p = self._probe_translate_op.Get()
            probe_world = (float(p[0]), float(p[1]), float(p[2]))
            colliders.append({
                "shape":  SHAPE_BOX,
                "center": probe_world,
                "row0":   Gf.Vec3f(1.0, 0.0, 0.0),
                "row1":   Gf.Vec3f(0.0, 1.0, 0.0),
                "row2":   Gf.Vec3f(0.0, 0.0, 1.0),
                "half_x": float(PROBE_HALF_X),
                "half_y": float(PROBE_HALF_Y),
                "half_z": float(PROBE_HALF_Z),
            })

        # ---- Cutting: advance the cut to wherever the probe currently is,
        # only while the probe is actually pressed into the pad (within its
        # Y footprint and pushed down to/past its top surface) ----
        if probe_world is not None:
            pwx, pwy, pwz = probe_world
            pad_top_z = SOFT_CENTER[2] + SOFT_HALF_Z
            engaged = (
                abs(pwy - SOFT_CENTER[1]) < SOFT_HALF_Y
                and pwz < pad_top_z + SKIN
            )
            if engaged:
                frac = (pwx - (SOFT_CENTER[0] - SOFT_HALF_X)) / (2 * SOFT_HALF_X)
                target_col = int(round(frac * (SOFT_RES_X - 1)))
                target_col = max(0, min(SOFT_RES_X - 1, target_col))
                self._cube.advance_cut(target_col)

        # Physics step — base collision passed separately so it runs every
        # solver iteration (same priority as ground constraint)
        self._cube.step(
            dt=DT,
            substeps=SUBSTEPS,
            solver_iters=SOLVER_ITERS,
            gravity=(0.0, 0.0, -9.81),
            damping=0.995,
            ground_z=None,         # soft body never touches ground directly
            base_box=self._base_box,
            friction=0.85,
            static_friction=0.98,
            colliders=colliders,
        )
        self._cube._check_cohesive_breaks()

        # Write positions to USD render mesh. Face-vertex INDICES are
        # re-set every frame too (not just at spawn) -- cutting doesn't
        # change how many triangles there are, but it does change which
        # vertices they point to, and the point buffer itself can grow.
        #
        # Safety net: if the solver ever blows up (NaN/Inf positions, from
        # any cause), do NOT push that into the renderer -- a degenerate
        # mesh like that is a plausible way to crash the RTX/Kit renderer
        # outright rather than just looking wrong. Skip the mesh write for
        # this frame and warn instead.
        cube_pos_np = self._cube.pos.numpy()
        if not np.all(np.isfinite(cube_pos_np)):
            carb.log_warn(
                "[WarpSoftBody] non-finite particle positions detected -- "
                "skipping this frame's mesh update to avoid handing the "
                "renderer a degenerate mesh.")
            return False
        self._cube_mesh.GetPointsAttr().Set(_vec3f_list(cube_pos_np))
        self._cube_mesh.GetFaceVertexIndicesAttr().Set(
            self._cube.tri_indices.tolist())
        return False

    # ------------------------------------------------------------------
    def _spawn(self):
        self._probe_last_good = None
        self._cube = SoftBodyCube(
            center=SOFT_CENTER,
            half_x=SOFT_HALF_X,
            half_y=SOFT_HALF_Y,
            half_z=SOFT_HALF_Z,
            res_x=SOFT_RES_X,
            res_y=SOFT_RES_Y,
            res_z=SOFT_RES_Z,
            total_mass=0.5,
            k_edge=0.65,
            k_vol=0.6,
            device=self._device,
        )
        fc = np.full(len(self._cube.tri_indices)//3, 3, dtype=np.int32)
        self._cube_mesh.CreateFaceVertexCountsAttr(fc.tolist())
        self._cube_mesh.CreateFaceVertexIndicesAttr(
            self._cube.tri_indices.tolist())
        self._cube_mesh.CreatePointsAttr(
            _vec3f_list(self._cube.pos.numpy()))

    def _set_probe_position(self, pos: tuple):
        if self._probe is None: return
        self._probe_translate_op.Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _prim_world_translation(self, prim_path):
        """Return (x, y, z) world translation, or None if it can't be read
        right now. Returning None (instead of silently defaulting to the
        origin) matters: whatever calls this must NOT treat a failed read
        as "prim is at (0,0,0)", or a transient read failure (e.g. mid-drag
        while the viewport gizmo is rewriting the prim's xformOps) will
        teleport that prim straight to the origin for a frame."""
        stage = omni.usd.get_context().get_stage()
        if stage is None: return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): return None
        try:
            mat = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            t   = mat.ExtractTranslation()
            x, y, z = float(t[0]), float(t[1]), float(t[2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return None
            return x, y, z
        except Exception as e:
            carb.log_warn(f"[WarpSoftBody] failed to read world transform "
                           f"for {prim_path}: {e}")
            return None

    def _clear_prim_xform(self, prim_path):
        stage = omni.usd.get_context().get_stage()
        if stage is None: return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): return
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            try:
                t = op.GetOpType()
                # IMPORTANT: match the op's own precision. Ops we create
                # ourselves default to double, but Kit's viewport gizmo
                # (move/rotate tool) can add ops -- e.g. xformOp:orient --
                # typed as single-precision (GfQuatf/GfVec3f). Setting a
                # double-precision value on a float-precision op raises a
                # Tf.ErrorException that, if uncaught, aborts the entire
                # update() call before it ever reaches the physics step --
                # which looks like "the sim just stopped simulating".
                precision = op.GetPrecision()
                if t == UsdGeom.XformOp.TypeTranslate:
                    if precision == UsdGeom.XformOp.PrecisionFloat:
                        op.Set(Gf.Vec3f(0, 0, 0))
                    elif precision == UsdGeom.XformOp.PrecisionHalf:
                        op.Set(Gf.Vec3h(0, 0, 0))
                    else:
                        op.Set(Gf.Vec3d(0, 0, 0))
                elif t == UsdGeom.XformOp.TypeRotateXYZ:
                    if precision == UsdGeom.XformOp.PrecisionDouble:
                        op.Set(Gf.Vec3d(0, 0, 0))
                    elif precision == UsdGeom.XformOp.PrecisionHalf:
                        op.Set(Gf.Vec3h(0, 0, 0))
                    else:
                        op.Set(Gf.Vec3f(0, 0, 0))
                elif t == UsdGeom.XformOp.TypeOrient:
                    if precision == UsdGeom.XformOp.PrecisionDouble:
                        op.Set(Gf.Quatd(1, 0, 0, 0))
                    elif precision == UsdGeom.XformOp.PrecisionHalf:
                        op.Set(Gf.Quath(1, 0, 0, 0))
                    else:
                        op.Set(Gf.Quatf(1, 0, 0, 0))
                # scale: never zero
            except Exception as e:
                # Never let a single op's type quirk abort the whole frame
                # (and with it, the physics step and mesh update below it).
                carb.log_warn(
                    f"[WarpSoftBody] failed to clear xformOp "
                    f"{op.GetOpName()} on {prim_path}: {e}")
