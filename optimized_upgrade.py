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
BLADE_PRIM_PATH     = "/World/WarpProbeBlade"
GROUND_PRIM_PATH    = "/World/WarpGroundViz"
LIGHT_PRIM_PATH     = "/World/WarpDomeLight"

_OWN_PATHS = {SOFT_BODY_PRIM_PATH, GROUND_PRIM_PATH,
              LIGHT_PRIM_PATH, "/World/Ground", BLADE_PRIM_PATH}

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

# Resolution: more particles along XY (flat face) than Z (thin dim).
# Bumped up from the prototype's 10x10x4 for smoother cut lines and
# enough Z layers for depth-aware cutting to actually have depth to work
# with. Kept deliberately moderate rather than maxed out: particle/tet
# count (and therefore GPU work per frame) grows roughly with
# res_x*res_y*res_z, so this is a direct trade against frame rate --
# see SUBSTEPS/SOLVER_ITERS below, and use the viewport's FPS readout to
# retune either direction if needed.
SOFT_RES_X   = 12   # particles along X → spacing = 2*0.05/11 ≈ 9.1 mm
SOFT_RES_Y   = 12   # particles along Y
SOFT_RES_Z   = 6    # particles along Z (thin) → spacing = 2*0.01/5 = 4.0 mm
# Total particles: 12*12*6  = 864
# Total tets:      (11*11*5)*6 = 3630

# Probe — surgical "rod" tool: a thin vertical capsule that hangs above
# the pad. Touching it deforms the pad (normal collision); dragging it
# sideways while pressed into the pad advances the cut.
ROD_RADIUS   = 0.0035    # 3.5mm rod radius
ROD_LENGTH   = 0.05      # 5cm rod length
ROD_HALF_LEN = ROD_LENGTH * 0.5
PROBE_CENTER = (0.15, 0.0, BASE_HALF_Z * 2 + ROD_LENGTH + 0.01)
PROBE_COLOR  = np.array([0.75, 0.45, 0.15])

# Tissue colors: the pad's outer surface reads as skin; any face exposed
# by cutting into the interior (i.e. not part of the original outer
# surface) reads as muscle, so a cut visually opens up "into" the body
# rather than just showing more of the same skin-colored material.
SKIN_TISSUE_COLOR   = np.array([0.41, 0.22, 0.16])   # unchanged skin tone
MUSCLE_TISSUE_COLOR = np.array([0.55, 0.05, 0.06])   # deep muscle red

# Keep PIPE_* aliases for gather_colliders compatibility
PIPE_PRIM_PATH       = PROBE_PRIM_PATH
PIPE_RADIUS          = ROD_RADIUS
PIPE_COLLIDE_RADIUS  = PIPE_RADIUS
PIPE_CENTER          = PROBE_CENTER
PIPE_AXIS            = (0.0, 0.0, 1.0)
PIPE_HALF_LEN        = ROD_HALF_LEN

SKIN = 0.005   # 1 mm collision skin

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT            = 1.0 / 60.0
# SUBSTEPS x SOLVER_ITERS directly multiplies how many GPU kernels get
# launched per frame (each solver iteration alone is 4 launches, plus
# collider dispatches) -- at particle/tet counts this small, per-launch
# CPU dispatch overhead dominates frame time far more than the actual
# GPU compute does, so cutting this product is the single biggest lever
# for hitting 60fps. Brought down from 12x8=96 to 8x6=48 (half as many
# launches per frame) as a first, conservative pass. If the pad still
# feels too soft/stretchy at this setting, raise SOLVER_ITERS back up a
# little before touching SUBSTEPS (iteration count affects constraint
# convergence/stiffness more directly than substep count does); if FPS
# still isn't where you want it, this and SOFT_RES_* above are the two
# knobs to keep tuning against the viewport's FPS readout.
SUBSTEPS      = 8
SOLVER_ITERS  = 6   # moderate reduction from original 10 -- see note below

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
def collide_capsule(
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    vel:      wp.array(dtype=wp.vec3),
    p0:       wp.vec3,   # one end of the capsule's centerline segment
    p1:       wp.vec3,   # other end
    radius:   wp.float32,
    skin:     wp.float32,
    friction: wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p        = pred[tid]
    seg      = p1 - p0
    seg_len2 = wp.dot(seg, seg)
    if seg_len2 < 1.0e-12:
        closest = p0
    else:
        t = wp.dot(p - p0, seg) / seg_len2
        t = wp.clamp(t, 0.0, 1.0)
        closest = p0 + seg * t
    diff = p - closest
    dist = wp.length(diff)
    lim  = radius + skin
    if dist < lim and dist > 1.0e-6:
        n = diff / dist
        pred[tid] = closest + n * lim
        v  = vel[tid]
        vn = wp.dot(v, n)
        if vn < 0.0:
            vel[tid] = v - n * (vn * (1.0 - friction))


@wp.kernel
def collide_cone(
    pred:       wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=wp.float32),
    vel:        wp.array(dtype=wp.vec3),
    apex:       wp.vec3,
    axis:       wp.vec3,   # unit vector, apex -> base
    half_angle: wp.float32,
    height:     wp.float32,
    skin:       wp.float32,
    friction:   wp.float32,
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p = pred[tid]
    v = p - apex
    h = wp.dot(v, axis)
    if h < -skin or h > height + skin:
        return
    h_c    = wp.clamp(h, 0.0, height)
    radial = v - axis * h
    dist   = wp.length(radial)
    r_at_h = h_c * wp.tan(half_angle)
    lim    = r_at_h + skin
    if dist < lim:
        if dist > 1.0e-6:
            n = radial / dist
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        pred[tid] = apex + axis * h_c + n * lim
        vv = vel[tid]
        vn = wp.dot(vv, n)
        if vn < 0.0:
            vel[tid] = vv - n * (vn * (1.0 - friction))




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

        # Geometry, kept around so world-space probe coordinates can be
        # converted into grid space for cutting.
        self.center = (cx, cy, cz)
        self.half_x, self.half_y, self.half_z = half_x, half_y, half_z

        # ── Cutting state ───────────────────────────────────────────────
        # Cuts are tracked as a set of severed "walls" -- the shared faces
        # between adjacent grid cells in the XY plane. Unlike the earlier
        # version, a severed wall is no longer all-or-nothing through the
        # full Z thickness: each entry stores the DEEPEST grid layer the
        # cut has reached (iz=0 is the bottom/base, iz=res_z-1 is the top
        # surface), so a shallow touch only splits the top layer or two
        # while pushing the tool further down splits progressively deeper
        # layers -- a wall is severed at a given iz iff iz >= its stored
        # threshold. Repeated strokes only ever deepen a cut (min of old
        # and new threshold), never heal it shallower.
        #
        # _severed_x[(ix, cy)] -> min_iz: wall between cell (ix-1, cy) and
        #   cell (ix, cy) is cut from the top down to layer min_iz (a
        #   "vertical" wall, crossed when the probe moves in +/-X).
        # _severed_y[(cx, iy)] -> min_iz: wall between cell (cx, iy-1) and
        #   cell (cx, iy) is cut the same way (a "horizontal" wall,
        #   crossed when the probe moves in +/-Y).
        # Together these support a cut path in ANY direction and depth,
        # only ever affecting the specific cells/layers the probe tip
        # actually swept through.
        self._severed_x = {}
        self._severed_y = {}

        # _vertex_groups[(vx, vy, iz)]: maps each of the (up to 4) grid
        # cells touching grid NODE (vx, vy, iz) to the particle id it
        # currently uses there. Lazily populated the first time a node is
        # touched by a cut; every entry starts out pointing at the single
        # original particle id (i.e. everything still connected). Tracked
        # per individual node (not per whole column) so a cut can split
        # some depths of a column while leaving others merged.
        self._vertex_groups = {}

        self.cohesive = []              # list of dicts: {a, b} -- pairs of
                                          # particle ids straddling a cut
                                          # interface, still bonded by a
                                          # breakable constraint

        # ── Particle grid ────────────────────────────────────────────────
        lx = np.linspace(-half_x, half_x, res_x, dtype=np.float64) + cx
        ly = np.linspace(-half_y, half_y, res_y, dtype=np.float64) + cy
        lz = np.linspace(-half_z, half_z, res_z, dtype=np.float64) + cz
        gx, gy, gz = np.meshgrid(lx, ly, lz, indexing="ij")
        positions = np.stack(
            [gx.flatten(), gy.flatten(), gz.flatten()], axis=1
        ).astype(np.float64)

        inv_mass = np.full(n, n / total_mass, dtype=np.float32)

        # ── Skin vs. muscle classification ───────────────────────────────
        # A grid vertex sits on the box's true outer surface iff it's on
        # the first/last index along ANY axis. Faces built entirely from
        # such vertices are the original outer skin; any boundary face
        # that later includes a vertex NOT on this surface can only have
        # been exposed by a cut slicing into the interior, so it reads as
        # muscle instead. This list only ever grows (duplicates copy their
        # source's flag), so the classification of a given piece of
        # material never changes once assigned.
        ix_idx, iy_idx, iz_idx = np.meshgrid(
            np.arange(res_x), np.arange(res_y), np.arange(res_z), indexing="ij")
        is_boundary = (
            (ix_idx == 0) | (ix_idx == res_x - 1) |
            (iy_idx == 0) | (iy_idx == res_y - 1) |
            (iz_idx == 0) | (iz_idx == res_z - 1)
        ).flatten()
        self._is_boundary_list = is_boundary.tolist()

        def vidx(ix, iy, iz):
            return ix * res_y * res_z + iy * res_z + iz

        # ── Weld bottom face (iz == 0) to the rigid base — no slip, no separation ──
        for ix in range(res_x):
            for iy in range(res_y):
                inv_mass[vidx(ix, iy, 0)] = 0.0

        # ── Tetrahedralization (Freudenthal 6-tet split) ─────────────────
        # tet_cell[ti]    = (ix, iy) the XY cell this tet belongs to. Used
        #                   to look up "every tet touching this XY cell" --
        #                   which of those actually get retargeted by a
        #                   cut is then narrowed down per-layer using
        #                   tet_corners below, since depth now matters.
        # tet_corners[ti] = the ORIGINAL (ix, iy, iz) grid coordinate of
        #                   each of the tet's 4 corners. These labels never
        #                   change even after a corner gets retargeted to a
        #                   duplicate particle id -- they're what let a cut
        #                   find "every tet corner currently at grid node
        #                   (vx, vy, vz)" without needing to reverse-engineer
        #                   it from whatever id happens to be there now.
        tets = []
        tet_cell = []
        tet_corners = []
        for ix in range(res_x - 1):
            for iy in range(res_y - 1):
                for iz in range(res_z - 1):
                    v000=vidx(ix,  iy,  iz  ); v100=vidx(ix+1,iy,  iz  )
                    v010=vidx(ix,  iy+1,iz  ); v110=vidx(ix+1,iy+1,iz  )
                    v001=vidx(ix,  iy,  iz+1); v101=vidx(ix+1,iy,  iz+1)
                    v011=vidx(ix,  iy+1,iz+1); v111=vidx(ix+1,iy+1,iz+1)
                    c000=(ix,iy,iz);     c100=(ix+1,iy,iz)
                    c010=(ix,iy+1,iz);   c110=(ix+1,iy+1,iz)
                    c001=(ix,iy,iz+1);   c101=(ix+1,iy,iz+1)
                    c011=(ix,iy+1,iz+1); c111=(ix+1,iy+1,iz+1)
                    cell_tets = [
                        (v000,v100,v110,v111),
                        (v000,v100,v101,v111),
                        (v000,v010,v110,v111),
                        (v000,v010,v011,v111),
                        (v000,v001,v101,v111),
                        (v000,v001,v011,v111),
                    ]
                    cell_corners = [
                        (c000,c100,c110,c111),
                        (c000,c100,c101,c111),
                        (c000,c010,c110,c111),
                        (c000,c010,c011,c111),
                        (c000,c001,c101,c111),
                        (c000,c001,c011,c111),
                    ]
                    tets += cell_tets
                    tet_corners += cell_corners
                    tet_cell += [(ix, iy)] * 6
        tets = np.array(tets, dtype=np.int64)

        # Ensure positive volume
        def tet_vol(p, t):
            a,b,c_,d = p[t[:,0]],p[t[:,1]],p[t[:,2]],p[t[:,3]]
            return np.einsum('ij,ij->i',b-a,np.cross(c_-a,d-a))/6.0
        vols = tet_vol(positions, tets)
        flip = vols < 0
        if flip.any():
            tets[flip,0], tets[flip,1] = tets[flip,1].copy(), tets[flip,0].copy()
            for ti in np.nonzero(flip)[0]:
                tc = tet_corners[int(ti)]
                tet_corners[int(ti)] = (tc[1], tc[0], tc[2], tc[3])
            vols = tet_vol(positions, tets)

        self.num_tets = len(tets)
        self._tet_cell = tet_cell               # list[(ix, iy)], len == num_tets
        self._tet_corners = tet_corners          # list[4 x (ix,iy,iz)], len == num_tets
        self._tets_by_cell = {}
        for ti, c in enumerate(tet_cell):
            self._tets_by_cell.setdefault(c, []).append(ti)

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
        self._vel_list      = [[0.0, 0.0, 0.0]] * n
        self._rest_pos_list = pos32.tolist()   # undeformed positions; only
                                                 # ever grows (duplicates
                                                 # copy their origin's rest
                                                 # position) -- used so cut
                                                 # springs get correct rest
                                                 # lengths, not strained ones
        self._inv_mass_list = inv_mass.tolist()
        self._tets_list     = tets.tolist()
        self._tet_vol_list   = vols.astype(np.float32).tolist()
        self._tet_stiff_list = [float(k_vol)] * self.num_tets
        self._tri_list       = [list(t) for t in oriented]

        # Every face at spawn time is, by construction, part of the whole
        # box's true outer surface (no cuts have happened yet) -- so it's
        # all skin. Interior "muscle" faces only ever appear later, once
        # cutting exposes them (see _rebuild_boundary_tris).
        self._tri_color_list = [list(SKIN_TISSUE_COLOR)] * len(oriented)
        self.tri_colors = np.array(self._tri_color_list, dtype=np.float32)

    def centroid(self):
        p = self.pos.numpy(); c = p.mean(0)
        return float(c[0]), float(c[1]), float(c[2])

    # ------------------------------------------------------------------
    # Cutting: local wall severing + per-vertex group splitting.
    #
    # The XY footprint is a grid of (res_x-1) x (res_y-1) cells. Every pair
    # of adjacent cells shares a "wall" (spanning the full Z thickness,
    # since the probe is a vertical rod). Dragging the probe through the
    # pad severs exactly the walls it actually crosses -- nothing else --
    # so a cut can run in any direction (not just along X), can curve, can
    # be applied piecemeal without racing ahead to fill in a whole column,
    # and never affects material the probe didn't actually pass through.
    #
    # At any grid vertex touched by a severed wall, the (up to 4) cells
    # that meet there are grouped by whichever ones are still connected
    # through an un-severed wall. If that grouping is finer than before,
    # the newly-separated group gets a fresh duplicate particle column
    # (copied from wherever the original currently sits, then linked back
    # to it with a breakable cohesive constraint) while the rest of the
    # material keeps using the id it already had -- so already-separated
    # material is never reset or "un-cut" by a later cut elsewhere.
    # ------------------------------------------------------------------
    def _vidx(self, ix, iy, iz):
        return ix * self.res_y * self.res_z + iy * self.res_z + iz

    def _world_to_grid_xy(self, x, y):
        """World (x, y) -> continuous grid-vertex coordinates, i.e. the
        same space column indices (ix, iy) live in (0 .. res-1)."""
        fx = (x - (self.center[0] - self.half_x)) / (2.0 * self.half_x) * (self.res_x - 1)
        fy = (y - (self.center[1] - self.half_y)) / (2.0 * self.half_y) * (self.res_y - 1)
        return fx, fy

    def depth_world_z_to_min_iz(self, tip_z):
        """Convert a probe tip's world-space Z into the deepest grid layer
        a cut at this depth should reach (0 = bottom/base, res_z-1 = top
        surface). This is what makes cutting depth-aware: a shallow touch
        near the top surface only severs the top layer or two, while
        pushing the tool further down toward the rigid base severs
        progressively deeper layers, reaching all the way through only at
        full penetration -- instead of any touch at all slicing clean
        through the whole thickness."""
        top_z = self.center[2] + self.half_z
        bot_z = self.center[2] - self.half_z
        span = max(top_z - bot_z, 1.0e-9)
        depth_frac = (top_z - tip_z) / span
        depth_frac = max(0.0, min(1.0, depth_frac))
        layers_from_top = depth_frac * (self.res_z - 1)
        min_iz = int(round((self.res_z - 1) - layers_from_top))
        return max(0, min(self.res_z - 1, min_iz))

    def cut_segment(self, x0, y0, x1, y1, min_iz):
        """Sever every cell wall the straight line from (x0,y0) to
        (x1,y1) (world-space) actually crosses, down to grid layer
        min_iz (see depth_world_z_to_min_iz), then locally re-split only
        the grid nodes touched by those new/deepened walls. No-op if the
        segment doesn't leave its starting cell and doesn't deepen
        anything already cut there."""
        fx0, fy0 = self._world_to_grid_xy(x0, y0)
        fx1, fy1 = self._world_to_grid_xy(x1, y1)
        touched = self._walk_and_sever(fx0, fy0, fx1, fy1, min_iz)
        if not touched:
            return

        self._sync_from_gpu()
        changed = False
        for (vx, vy) in touched:
            if self._retarget_vertex(vx, vy):
                changed = True
        if changed:
            self._rebuild_gpu_arrays()

    def _walk_and_sever(self, fx0, fy0, fx1, fy1, min_iz):
        """March along the segment in grid-vertex space, severing the
        walls it actually cuts through down to layer min_iz. Returns the
        set of grid vertex COLUMNS (vx, vy) adjacent to any newly cut or
        newly-deepened wall (the columns that need re-splitting), or an
        empty set if nothing changed. Each such column may end up only
        partially re-split (see _retarget_vertex) since depth matters now.

        Wall orientation is the subtle part: a probe advancing along X
        must sever the walls that separate ROWS (Y-walls) -- so the
        result is a cut that runs alongside its own path, splitting
        whatever is above the path from whatever is below it -- exactly
        like dragging a blade left-to-right leaves a cut that separates
        top from bottom, not a cut that chops the row it's in into
        disconnected pieces. Symmetrically, advancing along Y severs
        X-walls, splitting left from right. A locally-diagonal step
        severs both, approximating a diagonal cut against the grid.
        """
        max_ix, max_iy = self.res_x - 1, self.res_y - 1
        fx0 = max(0.0, min(max_ix, fx0)); fy0 = max(0.0, min(max_iy, fy0))
        fx1 = max(0.0, min(max_ix, fx1)); fy1 = max(0.0, min(max_iy, fy1))
        dx, dy = fx1 - fx0, fy1 - fy0
        dist = math.hypot(dx, dy)
        if dist < 1.0e-6:
            return set()

        # Oversample well past one sample per cell so a fast single-frame
        # drag can't jump clean over a cell boundary without registering
        # it. Bumped from 6x to 12x per grid unit: at 6x, a fast or
        # diagonal single-frame drag could still land its samples on the
        # "wrong side" of a wall crossing often enough to produce a
        # visibly jagged/stair-stepped cut line; 12x tracks the tip's
        # true path much more closely without meaningfully increasing
        # cost (this loop is pure Python-side bookkeeping, not physics).
        steps = max(1, int(math.ceil(dist * 12.0)))
        touched = set()
        prev_fx, prev_fy = fx0, fy0
        for s in range(1, steps + 1):
            t = s / steps
            fx = fx0 + dx * t
            fy = fy0 + dy * t
            step_dx = fx - prev_fx
            step_dy = fy - prev_fy
            mx = (prev_fx + fx) * 0.5
            my = (prev_fy + fy) * 0.5
            cx = max(0, min(self.res_x - 2, int(math.floor(mx))))
            cy = max(0, min(self.res_y - 2, int(math.floor(my))))
            rx = int(round(mx))
            ry = int(round(my))

            if abs(step_dx) >= abs(step_dy):
                # advancing through column cx -- sever the row-wall
                # nearest to where it's currently passing through, down
                # to min_iz (only deepening it if already partly cut)
                if 1 <= ry <= self.res_y - 2:
                    key = (cx, ry)
                    prev = self._severed_y.get(key)
                    if prev is None or min_iz < prev:
                        self._severed_y[key] = min_iz
                        touched.add((cx, ry)); touched.add((cx + 1, ry))
            if abs(step_dy) >= abs(step_dx):
                # advancing through row cy -- sever the column-wall
                # nearest to where it's currently passing through, down
                # to min_iz (only deepening it if already partly cut)
                if 1 <= rx <= self.res_x - 2:
                    key = (rx, cy)
                    prev = self._severed_x.get(key)
                    if prev is None or min_iz < prev:
                        self._severed_x[key] = min_iz
                        touched.add((rx, cy)); touched.add((rx, cy + 1))

            prev_fx, prev_fy = fx, fy
        return touched

    def _retarget_vertex(self, vx, vy):
        """Re-evaluate node connectivity at every Z layer of grid column
        (vx, vy), given every severed wall (and its depth) so far, and
        retarget whatever needs it. Returns True if anything changed."""
        cells = [c for c in ((vx-1,vy-1), (vx,vy-1), (vx-1,vy), (vx,vy))
                 if 0 <= c[0] <= self.res_x - 2 and 0 <= c[1] <= self.res_y - 2]
        if len(cells) <= 1:
            return False   # only one owner -- nothing to ever split here

        any_changed = False
        for iz in range(self.res_z):
            if self._retarget_node(vx, vy, iz, cells):
                any_changed = True
        return any_changed

    def _retarget_node(self, vx, vy, iz, cells):
        """Re-evaluate how many disconnected groups the (up to 4) cells
        touching grid NODE (vx, vy, iz) currently form, given every
        severed wall reaching this depth so far. Allocates a new
        duplicate particle for any newly-separated group and retargets
        the corner of every affected tet. Returns True if anything
        actually changed at this layer."""
        key = (vx, vy, iz)
        if key not in self._vertex_groups:
            base0 = self._vidx(vx, vy, iz)
            self._vertex_groups[key] = {c: base0 for c in cells}
        old_map = self._vertex_groups[key]

        # Union-Find over the incident cells, connected unless the wall
        # directly between them has been severed down to this depth.
        parent = {c: c for c in cells}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def severed_here(severed_dict, wall_key):
            thr = severed_dict.get(wall_key)
            return thr is not None and iz >= thr

        c00, c10, c01, c11 = (vx-1,vy-1), (vx,vy-1), (vx-1,vy), (vx,vy)
        cell_set = set(cells)
        for a, b, severed_dict, wall_key in (
            (c00, c10, self._severed_x, (vx, vy - 1)),   # x-wall, row vy-1
            (c01, c11, self._severed_x, (vx, vy)),       # x-wall, row vy
            (c00, c01, self._severed_y, (vx - 1, vy)),   # y-wall, col vx-1
            (c10, c11, self._severed_y, (vx, vy)),       # y-wall, col vx
        ):
            if a in cell_set and b in cell_set and not severed_here(severed_dict, wall_key):
                union(a, b)

        groups = {}
        for c in cells:
            groups.setdefault(find(c), []).append(c)
        new_groups = list(groups.values())

        old_groups = {}
        for c, base in old_map.items():
            old_groups.setdefault(base, []).append(c)
        old_partition = {frozenset(g) for g in old_groups.values()}
        new_partition = {frozenset(g) for g in new_groups}
        if new_partition == old_partition:
            return False   # this node wasn't actually split any further

        new_map = {}
        used_old_bases = set()
        changed = False
        for group in new_groups:
            old_bases_here = {old_map[c] for c in group}
            if len(old_bases_here) == 1 and next(iter(old_bases_here)) not in used_old_bases:
                base = next(iter(old_bases_here))
                used_old_bases.add(base)
            else:
                # Either this group mixes cells that used to be in different
                # groups (shouldn't happen -- walls only get added, groups
                # only ever get finer), or its old base was already claimed
                # by another sub-group this pass. Either way it's a fresh
                # split-off piece: give it its own duplicate particle.
                src_id = old_map[group[0]]
                base = self._alloc_duplicate_node(src_id)
                changed = True
            for c in group:
                new_map[c] = base

        self._vertex_groups[key] = new_map

        if changed or new_map != old_map:
            changed = True
            for c, base in new_map.items():
                if old_map.get(c) != base:
                    for ti in self._tets_by_cell.get(c, []):
                        self._retarget_tet_corner(ti, vx, vy, iz, base)
        return changed

    def _retarget_tet_corner(self, ti, vx, vy, iz, new_id):
        corners = self._tet_corners[ti]
        tet = self._tets_list[ti]
        target = (vx, vy, iz)
        for slot in range(4):
            if corners[slot] == target:
                tet[slot] = new_id

    def _alloc_duplicate_node(self, src_id):
        """Append a single fresh particle, copied from src_id's current
        position/velocity/mass, and bond it back with a breakable
        cohesive constraint. Returns the new particle's id."""
        new_id = len(self._pos_list)
        self._pos_list.append(list(self._pos_list[src_id]))
        self._vel_list.append(list(self._vel_list[src_id]))
        self._inv_mass_list.append(self._inv_mass_list[src_id])
        self._rest_pos_list.append(list(self._rest_pos_list[src_id]))
        self._is_boundary_list.append(self._is_boundary_list[src_id])
        self.cohesive.append({"a": src_id, "b": new_id})
        return new_id

    def _sync_from_gpu(self):
        """Pull current simulated position/velocity down from the GPU
        before mutating topology -- self._pos_list/_vel_list are only
        touched here and in _alloc_duplicate_node, so without this sync
        any rebuild would silently snap particles back to wherever they
        were as of the LAST topology change."""
        self._pos_list = self.pos.numpy().tolist()
        self._vel_list = self.vel.numpy().tolist()

    def _check_cohesive_breaks(self):
        """Check all still-bonded cut interfaces; remove any that have
        separated past CUT_DELTA_C. Rebuilds GPU arrays only if something
        actually broke this step (cheap check otherwise)."""
        if not self.cohesive:
            return
        pos_np = self.pos.numpy()
        still_bonded = []
        broke = False
        for c in self.cohesive:
            dist = float(np.linalg.norm(pos_np[c["a"]] - pos_np[c["b"]]))
            if dist >= CUT_DELTA_C:
                broke = True
            else:
                still_bonded.append(c)
        if not broke:
            return
        self.cohesive = still_bonded
        self._sync_from_gpu()
        self._rebuild_gpu_arrays()

    def _rebuild_structural_edges(self):
        """Recompute the tet-edge distance constraints from scratch from
        the CURRENT self._tets_list. Simpler and more robust than trying
        to incrementally patch an edge list: whichever tets a cut just
        retargeted onto different particle ids will naturally stop sharing
        an edge with their old neighbors here, with no extra bookkeeping."""
        edge_set = set()
        for tet in self._tets_list:
            for a, b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                i, j = tet[a], tet[b]
                if i > j: i, j = j, i
                edge_set.add((i, j))
        rp = self._rest_pos_list
        ei, ej, er, es = [], [], [], []
        for i, j in edge_set:
            ei.append(i); ej.append(j)
            dx = rp[i][0]-rp[j][0]; dy = rp[i][1]-rp[j][1]; dz = rp[i][2]-rp[j][2]
            er.append(math.sqrt(dx*dx + dy*dy + dz*dz))
            es.append(self._k_edge)
        return ei, ej, er, es

    def _rebuild_boundary_tris(self):
        """Recompute the render-mesh boundary faces from scratch from the
        current self._tets_list (a face belongs to exactly one tet).

        Also classifies each face as skin or muscle: a face keeps its
        "skin" look only if all 3 corners are original outer-surface
        vertices (see _is_boundary_list); any boundary face touching a
        vertex that was originally interior can only exist because a cut
        exposed it, so it reads as muscle instead. Returns
        (oriented_triangles, per_face_colors)."""
        tet_faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
        fc, fw = {}, {}
        for tet in self._tets_list:
            for fa, fb, fcx in tet_faces:
                ia, ib, ic = tet[fa], tet[fb], tet[fcx]
                key = tuple(sorted((ia, ib, ic)))
                fc[key] = fc.get(key, 0) + 1
                if key not in fw:
                    fw[key] = (ia, ib, ic)
        boundary = [fw[k] for k, v in fc.items() if v == 1]
        pos = np.array(self._pos_list, dtype=np.float32)
        gcen = pos.mean(0)
        oriented = []
        colors = []
        is_bnd = self._is_boundary_list
        skin, muscle = list(SKIN_TISSUE_COLOR), list(MUSCLE_TISSUE_COLOR)
        for ia, ib, ic in boundary:
            pa, pb, pc = pos[ia], pos[ib], pos[ic]
            fn  = np.cross(pb - pa, pc - pa)
            mid = (pa + pb + pc) / 3.0
            if np.dot(fn, mid - gcen) < 0.0:
                ia, ib, ic = ia, ic, ib
            oriented.append([ia, ib, ic])
            colors.append(skin if (is_bnd[ia] and is_bnd[ib] and is_bnd[ic]) else muscle)
        return oriented, colors

    def _rebuild_gpu_arrays(self):
        """Re-upload all GPU arrays from the current Python-side lists.
        Called after cut_segment() or a cohesive break changes topology.
        Assumes self._pos_list / self._vel_list are already current (via
        _sync_from_gpu, plus any new columns appended on top)."""
        struct_i, struct_j, struct_r, struct_s = self._rebuild_structural_edges()
        cohesive_i = [c["a"] for c in self.cohesive]
        cohesive_j = [c["b"] for c in self.cohesive]
        edge_i = struct_i + cohesive_i
        edge_j = struct_j + cohesive_j
        edge_r = struct_r + [0.0] * len(cohesive_i)
        edge_s = struct_s + [CUT_COHESIVE_STIFFNESS] * len(cohesive_i)

        self._tri_list, self._tri_color_list = self._rebuild_boundary_tris()

        n = len(self._pos_list)
        self.num_particles = n
        self.num_springs = len(edge_i)
        self.num_tets = len(self._tets_list)

        pos_np  = np.array(self._pos_list, dtype=np.float32)
        vel_np  = np.array(self._vel_list, dtype=np.float32)
        tets_np = np.array(self._tets_list, dtype=np.int64)

        self.pos        = wp.array(pos_np,                                   dtype=wp.vec3,    device=self.device)
        self.pred       = wp.array(pos_np.copy(),                            dtype=wp.vec3,    device=self.device)
        self.vel        = wp.array(vel_np,                                   dtype=wp.vec3,    device=self.device)
        self.inv_mass   = wp.array(np.array(self._inv_mass_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.spring_i   = wp.array(np.array(edge_i, dtype=np.int32),         dtype=wp.int32,   device=self.device)
        self.spring_j   = wp.array(np.array(edge_j, dtype=np.int32),         dtype=wp.int32,   device=self.device)
        self.rest_length= wp.array(np.array(edge_r, dtype=np.float32),       dtype=wp.float32, device=self.device)
        self.stiffness  = wp.array(np.array(edge_s, dtype=np.float32),       dtype=wp.float32, device=self.device)
        self.tet_a      = wp.array(tets_np[:, 0].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_b      = wp.array(tets_np[:, 1].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_c      = wp.array(tets_np[:, 2].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_d      = wp.array(tets_np[:, 3].astype(np.int32),           dtype=wp.int32,   device=self.device)
        self.tet_vol    = wp.array(np.array(self._tet_vol_list, dtype=np.float32),  dtype=wp.float32, device=self.device)
        self.tet_stiff  = wp.array(np.array(self._tet_stiff_list, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.corr       = wp.zeros(n, dtype=wp.vec3,  device=self.device)
        self.corr_count = wp.zeros(n, dtype=wp.int32, device=self.device)

        self.tri_indices = np.array(self._tri_list, dtype=np.int32).flatten()
        self.tri_colors  = np.array(self._tri_color_list, dtype=np.float32)

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


def _build_scalpel_blade_geometry():
    """Static, hand-authored low-poly scalpel-blade silhouette -- VISUAL
    ONLY. The probe's actual collider (used by both PhysX and the soft
    body's own cutting/collision math) stays the plain capsule it always
    was, completely untouched; this mesh just rides alongside it as a
    child prim so the tool *looks* like a blade with a sharp tip instead
    of a rod, without risking any change to the (working) physics.

    Local space matches the probe capsule's own local space: Z is the
    blade axis (tip at -ROD_HALF_LEN, handle butt at +ROD_HALF_LEN, same
    as the capsule's own extent), X is blade width, Y is thickness.
    Returns (points, face_vertex_counts, face_vertex_indices) ready to
    hand straight to a UsdGeom.Mesh.
    """
    z_tip      = -ROD_HALF_LEN
    z_belly    = -ROD_HALF_LEN + 0.30 * ROD_LENGTH   # widest point (edge)
    z_shoulder = -ROD_HALF_LEN + 0.55 * ROD_LENGTH   # necks in toward handle
    z_butt     =  ROD_HALF_LEN                       # handle end

    w_belly    = 4.0 * ROD_RADIUS   # widest point of the cutting edge
    w_neck     = 1.6 * ROD_RADIUS
    w_handle   = 1.1 * ROD_RADIUS   # slender handle
    half_thick = 0.5 * ROD_RADIUS   # blade thickness

    # 2D outline (X, Z) going around the silhouette once: tip, up the
    # right edge to the handle butt, then back down the mirrored left
    # edge to the tip. Roughly star-shaped around its centroid, which is
    # what the cap-triangulation below relies on.
    outline = [
        (0.0,       z_tip),
        (+w_belly,  z_belly),
        (+w_neck,   z_shoulder),
        (+w_handle, z_butt),
        (-w_handle, z_butt),
        (-w_neck,   z_shoulder),
        (-w_belly,  z_belly),
    ]
    m = len(outline)
    cx = sum(p[0] for p in outline) / m
    cz = sum(p[1] for p in outline) / m

    points = []
    for (x, z) in outline:
        points.append((x, +half_thick, z))     # front copy: indices 0..m-1
    for (x, z) in outline:
        points.append((x, -half_thick, z))     # back copy:  indices m..2m-1
    front_c = len(points); points.append((cx, +half_thick, cz))
    back_c  = len(points); points.append((cx, -half_thick, cz))

    tris = []
    for i in range(m):                          # front cap (fan)
        j = (i + 1) % m
        tris.append((front_c, i, j))
    for i in range(m):                          # back cap (fan, reversed)
        j = (i + 1) % m
        tris.append((back_c, m + j, m + i))
    for i in range(m):                          # side walls
        j = (i + 1) % m
        fi, fj, bi, bj = i, j, m + i, m + j
        tris.append((fi, fj, bj))
        tris.append((fi, bj, bi))

    face_vertex_counts = [3] * len(tris)
    face_vertex_indices = [idx for tri in tris for idx in tri]
    return points, face_vertex_counts, face_vertex_indices


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
        self._last_tri_count     = 0      # triangle count faceVertexCounts
                                           # was last set for -- see update()
        self._cube_color_pv      = None   # per-face skin/muscle displayColor
                                           # primvar handle, created in _spawn()

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

        # Probe — surgical rod: a vertical capsule, kinematic rigid body.
        # No scale op needed -- radius/height attrs define the capsule
        # directly, so the only xformOp on this prim we ever author is
        # translate (rotate/orient ops the viewport gizmo may add are
        # excluded from the transform order each frame, not zeroed --
        # see _clear_prim_xform).
        probe = UsdGeom.Capsule.Define(stage, PROBE_PRIM_PATH)
        probe.CreateRadiusAttr(ROD_RADIUS)
        probe.CreateHeightAttr(ROD_LENGTH)
        probe.CreateAxisAttr(UsdGeom.Tokens.z)
        probe.CreateDisplayColorAttr([Gf.Vec3f(*PROBE_COLOR.tolist())])
        xfp = UsdGeom.Xformable(probe.GetPrim())
        self._probe_translate_op = xfp.AddTranslateOp()
        self._probe_translate_op.Set(Gf.Vec3d(*PROBE_CENTER))
        rba2 = UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        rba2.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        # PhysX supports capsules natively -- no convex-hull approximation
        # needed here (that was only for the old box-shaped probe).
        self._probe = probe

        # The capsule above stays exactly as it was -- it's still what
        # PhysX and the soft body's own capsule-vs-particle collision/
        # cutting math actually use, untouched. Hide it and show a
        # scalpel-blade-shaped mesh in its place instead.
        #
        # IMPORTANT: the blade is its own top-level sibling prim, NOT a
        # child of the capsule. USD visibility is inherited downward with
        # no override -- a prim set invisible makes every descendant
        # invisible too, no exceptions -- so nesting the blade under the
        # now-hidden capsule would hide the blade right along with it
        # (which is exactly what made the probe disappear entirely). As
        # an independent prim its own translate op is synced to match the
        # capsule's every frame instead (see the per-frame update below).
        UsdGeom.Imageable(probe.GetPrim()).MakeInvisible()
        blade_pts, blade_fc, blade_fi = _build_scalpel_blade_geometry()
        blade = UsdGeom.Mesh.Define(stage, Sdf.Path(BLADE_PRIM_PATH))
        blade.CreateDoubleSidedAttr(True)
        blade.CreatePointsAttr([Gf.Vec3f(*p) for p in blade_pts])
        blade.CreateFaceVertexCountsAttr(blade_fc)
        blade.CreateFaceVertexIndicesAttr(blade_fi)
        blade.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.82, 0.85)])  # steel
        blade_xfp = UsdGeom.Xformable(blade.GetPrim())
        self._blade_translate_op = blade_xfp.AddTranslateOp()
        self._blade_translate_op.Set(Gf.Vec3d(*PROBE_CENTER))
        self._blade = blade

        # Soft body render mesh (points written each frame). Color is a
        # per-face primvar (skin vs. muscle), authored once self._cube
        # exists -- see _spawn().
        cb = UsdGeom.Mesh.Define(stage, SOFT_BODY_PRIM_PATH)
        cb.CreateDoubleSidedAttr(True)
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

        # Probe collider from translate op -- the rod is a vertical
        # capsule, so its collider is defined by the two endpoints of its
        # centerline (p0 = top, p1 = bottom tip) rather than a box.
        probe_world = None
        probe_tip_z = None
        if self._probe is not None:
            p = self._probe_translate_op.Get()
            cx, cy, cz = float(p[0]), float(p[1]), float(p[2])
            probe_world = (cx, cy, cz)
            probe_tip_z = cz - ROD_HALF_LEN   # bottom end -- the working tip
            colliders.append({
                "shape":  SHAPE_CAPSULE,
                "p0":     (cx, cy, cz + ROD_HALF_LEN),
                "p1":     (cx, cy, probe_tip_z),
                "radius": float(ROD_RADIUS),
            })

        # ---- Cutting: cut along wherever the rod's tip actually travels
        # in XY while it's pressed into the pad (within its footprint and
        # pushed down to/past its top surface). The tip -- not the rod's
        # center -- is the working end that pokes/cuts. Unlike a single
        # column index, this follows the tip's real path in any direction
        # (X, Y, or a diagonal drag) and only severs the cells the tip
        # actually swept through, one probe-radius segment at a time.
        # Depth-aware: how far DOWN the tip has actually penetrated (not
        # just whether it's touching) decides how deep the cut goes --
        # a shallow graze only nicks the top layer(s), not the whole
        # thickness. ----
        if probe_world is not None:
            pwx, pwy, _ = probe_world
            pad_top_z = SOFT_CENTER[2] + SOFT_HALF_Z
            engaged = (
                abs(pwx - SOFT_CENTER[0]) < SOFT_HALF_X
                and abs(pwy - SOFT_CENTER[1]) < SOFT_HALF_Y
                and probe_tip_z < pad_top_z + SKIN
            )
            if engaged:
                min_iz = self._cube.depth_world_z_to_min_iz(probe_tip_z)
                if self._cut_last_xy is not None:
                    lx, ly = self._cut_last_xy
                    self._cube.cut_segment(lx, ly, pwx, pwy, min_iz)
                self._cut_last_xy = (pwx, pwy)
            else:
                # Tip lifted clear of the pad -- forget the trail so the
                # next press-in starts a fresh cut instead of drawing a
                # phantom line from wherever it last was.
                self._cut_last_xy = None

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
        # re-set every frame too (not just at spawn) -- and, critically,
        # so is faceVertexCounts. Cutting DOES change how many triangles
        # exist: severing a wall splits a grid vertex into two duplicate
        # particle columns, which turns previously-internal faces (shared
        # by two tets, so not boundary) into faces that are now only
        # claimed by one tet on each side -- i.e. new boundary faces
        # appear on both sides of the cut, so the triangle count goes UP.
        # If faceVertexCounts is left stale (its length no longer matches
        # how faceVertexIndices actually chunks into triangles), USD sees
        # an inconsistent mesh and Hydra culls/hides it outright -- this
        # was the "mesh deletes itself while cutting" bug. Recomputing it
        # whenever the triangle count changes keeps the two attributes
        # consistent no matter how topology changed this step.
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
        tri_indices = self._cube.tri_indices
        num_tris = len(tri_indices) // 3
        if num_tris != self._last_tri_count:
            self._cube_mesh.GetFaceVertexCountsAttr().Set(
                np.full(num_tris, 3, dtype=np.int32).tolist())
            if self._cube_color_pv is not None:
                self._cube_color_pv.Set(_vec3f_list(self._cube.tri_colors))
            self._last_tri_count = num_tris
        self._cube_mesh.GetPointsAttr().Set(_vec3f_list(cube_pos_np))
        self._cube_mesh.GetFaceVertexIndicesAttr().Set(
            tri_indices.tolist())
        return False

    # ------------------------------------------------------------------
    def _spawn(self):
        self._probe_last_good = None
        self._cut_last_xy = None
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
        num_tris = len(self._cube.tri_indices) // 3
        fc = np.full(num_tris, 3, dtype=np.int32)
        self._cube_mesh.CreateFaceVertexCountsAttr(fc.tolist())
        self._cube_mesh.CreateFaceVertexIndicesAttr(
            self._cube.tri_indices.tolist())
        self._cube_mesh.CreatePointsAttr(
            _vec3f_list(self._cube.pos.numpy()))
        # Per-face (uniform) display color: skin on the outer surface,
        # muscle red on any face a cut has exposed. CreatePrimvar is a
        # no-op if it already exists from a prior spawn (e.g. RESET), so
        # this is safe to call every time -- only the values change.
        if self._cube_color_pv is None:
            self._cube_color_pv = UsdGeom.PrimvarsAPI(
                self._cube_mesh.GetPrim()
            ).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.uniform)
        self._cube_color_pv.Set(_vec3f_list(self._cube.tri_colors))
        self._last_tri_count = num_tris

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
        """Force the prim's applied transform down to just its translate
        op. Rotate/orient ops the viewport gizmo adds are EXCLUDED from
        the xformOpOrder rather than reset in place.

        Resetting them in place requires matching their exact value type
        and precision (GfVec3f vs GfVec3d, GfQuatf vs GfQuatd, etc) --
        which is exactly what kept breaking here: whatever type Kit
        happened to create the op as, our hardcoded reset value didn't
        match, and an uncaught type-mismatch exception killed the entire
        frame (physics step and mesh update included) before it could run.
        Dropping those ops from the order sidesteps the type question
        completely -- they simply stop contributing to the transform.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None: return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): return
        try:
            xformable = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is not None:
                xformable.SetXformOpOrder([translate_op])
        except Exception as e:
            carb.log_warn(f"[WarpSoftBody] failed to reset xform order on "
                           f"{prim_path}: {e}")
