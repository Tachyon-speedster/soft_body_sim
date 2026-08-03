from __future__ import annotations

import math
import time
import numpy as np
import warp as wp
import carb
import omni.usd
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from .force_sensor import ForceFeedbackSensor

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

# Resolution: more particles along XY (flat face) than Z (thin dim).
# Z was bumped from 6 to 8 to give the layered tissue stack below (skin/
# fat/muscle) enough Z-cell-slices to actually read as distinct bands
# instead of one or two cells each. XY was trimmed from 20 to 18 to
# compensate -- tet count (and therefore rebuild-on-cut and per-frame
# physics cost) lands close to the previously-tuned 20x20x6 budget
# rather than growing ~40% on top of it. The layer system itself works
# at any resolution (it defines layers by FRACTION of thickness, not by
# fixed cell index), so either axis is a free knob to retune from here
# if you want denser cut lines (raise XY) or thicker/smoother layer
# transitions (raise Z) -- just keep an eye on the frame-budget numbers
# from the adaptive solver-iteration controller if you push either up.
SOFT_RES_X   = 18   # particles along X -> spacing = 2*0.05/17 ~= 5.9 mm
SOFT_RES_Y   = 18   # particles along Y
SOFT_RES_Z   = 8    # particles along Z (thin) -> spacing = 2*0.01/7 ~= 2.9 mm
# Total particles: 18*18*8 = 2592
# Total tets: (17*17*7)*6 = 12138

# Probe -- surgical "rod" tool: a thin vertical capsule that hangs above
# the pad. Touching it deforms the pad (normal collision); dragging it
# sideways while pressed into the pad advances the cut.
# Slimmed down to read as an actual fine surgical instrument rather than
# a blunt rod -- a smaller radius pushes far less tissue aside per unit
# of travel, which keeps deformation localized to the incision line
# instead of ballooning the whole local mesh outward.
ROD_RADIUS   = 0.0012    # 1.2mm rod radius (was 3.5mm)
ROD_LENGTH   = 0.05      # 5cm rod length
ROD_HALF_LEN = ROD_LENGTH * 0.5
PROBE_CENTER = (0.15, 0.0, BASE_HALF_Z * 2 + ROD_LENGTH + 0.01)
PROBE_COLOR  = np.array([0.75, 0.45, 0.15])   # handle grip color

# Scalpel visual: blade (working end) + handle (grip end), combined into
# ONE mesh prim rather than separate blade/handle prims.
#
# This is deliberate, not a simplification for its own sake: an earlier
# attempt built the blade and handle as separate prims (sibling or
# parent/child) and hit exactly the failure this whole design avoids --
# clicking/dragging one part in the viewport only moves that part's own
# transform, so the blade and handle (and, worse, the actual collision
# capsule driving the cut) can end up at different positions with no way
# to detect or prevent it short of per-frame syncing code that's easy to
# miss a spot in. Merging both shapes into a single mesh's vertex data
# means there is only ever ONE transform for the whole tool to have --
# structurally, not by convention -- so there is no separate "handle
# prim" to independently select, drag, or desync. Whatever you click on
# anywhere along the tool is this one prim.
#
# BLADE_LENGTH is the fraction of the tool's local Z extent (from the
# tip) that's blade rather than handle; the ACTUAL collision capsule used
# for poking/cutting (see probe_world/probe_tip_z in _update_impl) only
# spans this blade portion at the slim ROD_RADIUS above -- the handle is
# visual only and never touches the pad, same as a real scalpel.
BLADE_LENGTH        = 0.35 * ROD_LENGTH
BLADE_TIP_LOCAL_Z   = -ROD_HALF_LEN                    # local Z, tip end
BLADE_BUTT_LOCAL_Z  = BLADE_TIP_LOCAL_Z + BLADE_LENGTH  # blade/handle join
HANDLE_BUTT_LOCAL_Z = ROD_HALF_LEN                      # local Z, grip end
HANDLE_RADIUS       = 0.0025    # 2.5mm -- a normal round grip, independent
                                 # of the slim blade/collision radius above
STEEL_COLOR         = np.array([0.80, 0.82, 0.85])

# ---------------------------------------------------------------------------
# Tissue layers -- skin / fat / muscle, stacked through the pad's Z depth
# ---------------------------------------------------------------------------
# Each layer is defined by how much of the pad's TOTAL thickness it
# occupies, as a fraction measured from the TOP (skin) surface down to
# the rigid base -- 0.0 = top surface, 1.0 = welded to the base. This is
# resolution-independent (works the same whether SOFT_RES_Z is 6 or 20),
# so retuning SOFT_RES_Z for a smoother cut never requires re-tuning
# where each layer sits -- only "bottom_frac" below controls that.
#
# This sample pad is a stand-in for a real limb cross-section: proportions
# and colors here are illustrative starting points, tuned to look right
# at THIS pad's 2cm thickness. Scaling up to an actual limb-sized soft
# body (bigger half_z, more realistic per-layer thickness like 2-3mm
# skin / 5-15mm fat / rest muscle) is just changing bottom_frac below --
# the lookup mechanism, per-layer stiffness, per-layer color, and
# depth-aware cutting through them all stay exactly the same.
#
# k_edge/k_vol follow the same [0,1]-ish XPBD stiffness convention used
# everywhere else in this file (1.0 = rigid, near 0 = very compliant):
#   skin   -- thin but tough: resists stretching more than fat or muscle
#   fat    -- soft and squishy: the most compliant layer by far
#   muscle -- denser and more elastic than fat, but still gives more
#             than skin
TISSUE_LAYERS = [
    {"name": "skin",   "bottom_frac": 0.12,
     "color": np.array([0.41, 0.22, 0.16]), "k_edge": 0.90, "k_vol": 0.85},
    {"name": "fat",    "bottom_frac": 0.42,
     "color": np.array([0.93, 0.80, 0.55]), "k_edge": 0.18, "k_vol": 0.15},
    {"name": "muscle", "bottom_frac": 1.00,
     "color": np.array([0.55, 0.05, 0.06]), "k_edge": 0.60, "k_vol": 0.55},
]
_TISSUE_BOUNDS = np.array([l["bottom_frac"] for l in TISSUE_LAYERS], dtype=np.float64)
_TISSUE_COLORS = np.array([l["color"]       for l in TISSUE_LAYERS], dtype=np.float32)
_TISSUE_KEDGE  = np.array([l["k_edge"]      for l in TISSUE_LAYERS], dtype=np.float32)
_TISSUE_KVOL   = np.array([l["k_vol"]       for l in TISSUE_LAYERS], dtype=np.float32)

# Kept as the visible fallback/default tissue tone for materials and any
# legacy reference -- just an alias onto the top (skin) layer's color now
# that per-layer coloring has replaced the old binary skin/muscle scheme.
SKIN_TISSUE_COLOR = TISSUE_LAYERS[0]["color"]

# Keep PIPE_* aliases for gather_colliders compatibility
PIPE_PRIM_PATH       = PROBE_PRIM_PATH
PIPE_RADIUS          = ROD_RADIUS
PIPE_COLLIDE_RADIUS  = PIPE_RADIUS
PIPE_CENTER          = PROBE_CENTER
PIPE_AXIS            = (0.0, 0.0, 1.0)
PIPE_HALF_LEN        = ROD_HALF_LEN

# Collision skin margin added on top of every collider's raw radius/half
# extents (see collide_* kernels: `lim = radius + skin`). Previously this
# was 0.005 (5mm) despite its comment claiming 1mm -- combined with the
# old 3.5mm rod radius that gave an effective probe collision extent of
# 8.5mm, WIDER than the ~5.3mm grid spacing, which is what caused the
# mesh to billow out during cuts instead of parting cleanly. Brought down
# to match the comment's original intent and to stay comfortably thinner
# than the grid spacing once combined with the slimmed ROD_RADIUS above.
SKIN = 0.0008   # 0.8mm collision skin

# How far PAST the top surface the tip must actually be before a cut is
# allowed to fire, measured independently of the collision SKIN margin
# above. This used to just be "probe_tip_z < pad_top_z + SKIN" -- i.e.
# cutting engaged the instant the tip so much as grazed within 0.8mm of
# the surface, which is also exactly the condition collide_capsule_sensed
# uses to start pushing particles. Since cut_segment() ran BEFORE the
# physics step in the same frame (see _update_impl), that meant: the
# instant you touched the pad, the tissue right under the tip was severed
# before collision ever got a chance to resolve any resistance against
# it -- so the force sensor never saw anything but 0N, no matter how hard
# or how long you pressed, while still visibly "cutting" a channel open.
# Requiring a real puncture depth here (not just skin-margin contact)
# gives contact resistance a chance to build and be measured -- matching
# the real reading_1/reading_2 traces, which show a genuine ramp up to
# ~-3 to -4N BEFORE the tissue actually gives way -- rather than parting
# on first touch.
CUT_ENGAGE_DEPTH = 0.0015   # 1.5mm past the top surface before cutting

# Fixed N/m used ONLY by the diagnostic penetration-depth force estimate
# (pen_force_accum in collide_capsule_sensed) -- a rough order-of-magnitude
# stand-in for tissue stiffness, not fit to anything. Its only job is to
# give a nonzero, non-collapsing signal to compare against the "real"
# m*dx/h^2 recovered force while debugging why the latter reads 0N.
CONTACT_STIFFNESS = 500.0   # N per metre of penetration

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT            = 1.0 / 60.0
SUBSTEPS      = 12
# Raised from 8 to 10 to compensate for the ~7.4x jump in tet/edge
# constraint count from the resolution increase above. XPBD convergence
# per substep degrades as more constraints share the same particles (a
# denser mesh means more neighbors pulling on each node per Gauss-Seidel
# sweep); without this bump the mesh would visually read softer/jigglier
# than the coarse version even though k_edge/k_vol didn't change. This
# keeps effective stiffness roughly consistent with the original
# resolution instead of drifting purely from the density increase.
SOLVER_ITERS  = 10

# ---------------------------------------------------------------------------
# Adaptive solver-iteration budget ("dynamic quality", the same idea
# games use for dynamic resolution): SOLVER_ITERS above is the CEILING,
# not a fixed cost paid every frame. WarpSoftBodySim measures how long
# self._cube.step() actually takes and scales the iteration count within
# [SOLVER_ITERS_MIN, SOLVER_ITERS] to hold TARGET_FRAME_MS -- so a heavier
# moment (e.g. right after a cut roughly doubles constraint count in the
# affected region) costs a few fewer iterations for a few frames instead
# of a dropped/stuttered frame, and it climbs back up automatically once
# there's headroom again. This only ever changes HOW MANY of the exact
# same solver iterations run -- never their math -- so solve quality
# degrades gracefully (softer convergence that frame) rather than
# incorrectly.
# ---------------------------------------------------------------------------
TARGET_FRAME_MS   = 33.3   # 30fps budget for the physics step specifically
SOLVER_ITERS_MIN  = 4      # never go below this -- much less and the pad
                            # visibly loses stiffness/starts to look mushy
ADAPT_UP_MARGIN   = 0.85   # step() taking < 85% of budget -> try +1 iter
ADAPT_DOWN_MARGIN = 1.05   # step() taking > 105% of budget -> try -1 iter
ADAPT_EMA_ALPHA   = 0.15   # smoothing factor on the measured step time,
                            # so one slow/fast outlier frame doesn't yank
                            # the iteration count around

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
def collide_sphere_sensed(
    pred:     wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    vel:      wp.array(dtype=wp.vec3),
    center:   wp.vec3,
    radius:   wp.float32,
    skin:     wp.float32,
    friction: wp.float32,
    inv_sub_dt2:  wp.float32,
    force_accum:  wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    pen_force_accum: wp.array(dtype=wp.float32),
    contact_stiffness: wp.float32,
):
    # Same contact math as collide_sphere, but also recovers a Newton-
    # scale contact force (F = m*dx/h^2, see collide_capsule_sensed /
    # force_sensor.py) so any manually-added sphere collider (Create >
    # Shapes > Sphere + Collider Preset) can drive the force-feedback
    # panel, not just the built-in WarpProbe.
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return
    p    = pred[tid]
    diff = p - center
    dist = wp.length(diff)
    lim  = radius + skin
    if dist < lim and dist > 1.0e-6:
        n = diff / dist
        new_p = center + n * lim
        disp  = new_p - p
        pred[tid] = new_p
        mass = 1.0 / inv_mass[tid]
        f_on_particle = disp * (mass * inv_sub_dt2)
        wp.atomic_add(force_accum, 0, -f_on_particle)
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(pen_force_accum, 0, (lim - dist) * contact_stiffness)
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
def collide_box_sensed(
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
    inv_sub_dt2:  wp.float32,
    force_accum:  wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    pen_force_accum: wp.array(dtype=wp.float32),
    contact_stiffness: wp.float32,
):
    # Same contact math as collide_box, plus force recovery (see
    # collide_sphere_sensed above) -- lets a manually-added Cube collider
    # drive the force-feedback panel too.
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
        new_p = p + n_world * min_d
        disp  = new_p - p
        pred[tid] = new_p
        mass = 1.0 / inv_mass[tid]
        f_on_particle = disp * (mass * inv_sub_dt2)
        wp.atomic_add(force_accum, 0, -f_on_particle)
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(pen_force_accum, 0, min_d * contact_stiffness)
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
def collide_cylinder_sensed(
    pred:          wp.array(dtype=wp.vec3),
    inv_mass:      wp.array(dtype=wp.float32),
    vel:           wp.array(dtype=wp.vec3),
    pipe_center:   wp.vec3,
    pipe_axis:     wp.vec3,
    collide_radius: wp.float32,
    pipe_half_len: wp.float32,
    skin:          wp.float32,
    friction:      wp.float32,
    inv_sub_dt2:  wp.float32,
    force_accum:  wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    pen_force_accum: wp.array(dtype=wp.float32),
    contact_stiffness: wp.float32,
):
    # Same contact math as collide_cylinder, plus force recovery (see
    # collide_sphere_sensed above) -- lets a manually-added Cylinder
    # collider drive the force-feedback panel too.
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
        new_p = closest + n * lim
        disp  = new_p - p
        pred[tid] = new_p
        mass = 1.0 / inv_mass[tid]
        f_on_particle = disp * (mass * inv_sub_dt2)
        wp.atomic_add(force_accum, 0, -f_on_particle)
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(pen_force_accum, 0, (lim - dist) * contact_stiffness)
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
def zero_vec3_single(arr: wp.array(dtype=wp.vec3)):
    # Resets a length-1 accumulator array. Used once per step() call to
    # clear the probe's force accumulator before the substep loop -- see
    # collide_capsule_sensed and WarpSoftBodySim.step().
    arr[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_int32_single(arr: wp.array(dtype=wp.int32)):
    # Same as zero_vec3_single but for the contact_count diagnostic
    # counter -- see collide_capsule_sensed. Purely a debugging aid: lets
    # the Python side tell "no particles are within collision range of
    # the probe at all" apart from "particles are in range but the
    # recovered force is ~0", which look identical from the force reading
    # alone.
    arr[0] = 0


@wp.kernel
def zero_float32_single(arr: wp.array(dtype=wp.float32)):
    # Zeroes the pen_force_accum diagnostic (see collide_capsule_sensed).
    arr[0] = 0.0


@wp.kernel
def collide_capsule_sensed(
    pred:         wp.array(dtype=wp.vec3),
    inv_mass:     wp.array(dtype=wp.float32),
    vel:          wp.array(dtype=wp.vec3),
    p0:           wp.vec3,
    p1:           wp.vec3,
    radius:       wp.float32,
    skin:         wp.float32,
    friction:     wp.float32,
    inv_sub_dt2:  wp.float32,
    force_accum:  wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    pen_force_accum: wp.array(dtype=wp.float32),
    contact_stiffness: wp.float32,
):
    # Identical contact math to collide_capsule -- this is the probe's
    # own capsule collider, tagged "sense" in _update_impl -- but it also
    # recovers a Newton-scale contact force from the correction it just
    # applied: F = m * dx / h^2 (see force_sensor.py's module docstring
    # for the derivation), and atomically adds the REACTION force
    # (Newton's third law: -sum of per-particle contact force) onto
    # force_accum[0] so the Python side can read one net (Fx,Fy,Fz) per
    # substep for the whole probe.
    #
    # NOTE: this is no longer the ONLY sensed collider. gather_colliders()
    # now tags every scene-scanned collider (any Sphere/Cube/Capsule/
    # Cylinder/Cone you add via Create > Shapes + Collider Preset) with
    # "sense": True as well, and _dispatch_collider routes each shape to
    # its own *_sensed kernel (collide_sphere_sensed, collide_box_sensed,
    # collide_cylinder_sensed, collide_cone_sensed -- same F=m*dx/h^2
    # math, defined near each shape's plain kernel above) when that flag
    # is set. So force_accum[0] is now the net reaction from WHATEVER is
    # touching the pad this substep, probe or otherwise -- fixing the
    # "deforms the mesh but Force_Z stays 0.0 N" bug that showed up when
    # poking with a hand-added shape instead of dragging WarpProbe.
    #
    # ALSO tracks two diagnostics, purely to separate "geometry never
    # touches" from "geometry touches but the recovered force is ~0":
    #   contact_count    -- how many particles were actually within
    #                        collision range this call, at all.
    #   pen_force_accum  -- an INDEPENDENT force estimate, computed
    #                        directly from penetration depth (lim - dist)
    #                        times a fixed contact_stiffness, rather than
    #                        from the position correction. If the mesh
    #                        solver has already converged the correction
    #                        signal away to ~0 at steady contact (a known
    #                        failure mode of the m*dx/h^2 method when the
    #                        constraint solver runs enough iterations to
    #                        nearly satisfy every spring each substep --
    #                        see softbody_core.py's diagnostic notes),
    #                        this penetration-depth signal should still
    #                        be nonzero as long as contact_count > 0,
    #                        since it doesn't depend on how much the
    #                        *previous* substep already corrected.
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
        new_p = closest + n * lim
        disp  = new_p - p
        pred[tid] = new_p
        mass = 1.0 / inv_mass[tid]
        f_on_particle = disp * (mass * inv_sub_dt2)
        wp.atomic_add(force_accum, 0, -f_on_particle)
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(pen_force_accum, 0, (lim - dist) * contact_stiffness)
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
def collide_cone_sensed(
    pred:       wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=wp.float32),
    vel:        wp.array(dtype=wp.vec3),
    apex:       wp.vec3,
    axis:       wp.vec3,
    half_angle: wp.float32,
    height:     wp.float32,
    skin:       wp.float32,
    friction:   wp.float32,
    inv_sub_dt2:  wp.float32,
    force_accum:  wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    pen_force_accum: wp.array(dtype=wp.float32),
    contact_stiffness: wp.float32,
):
    # Same contact math as collide_cone, plus force recovery (see
    # collide_sphere_sensed above) -- lets a manually-added Cone collider
    # drive the force-feedback panel too.
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
        new_p = apex + axis * h_c + n * lim
        disp  = new_p - p
        pred[tid] = new_p
        mass = 1.0 / inv_mass[tid]
        f_on_particle = disp * (mass * inv_sub_dt2)
        wp.atomic_add(force_accum, 0, -f_on_particle)
        wp.atomic_add(contact_count, 0, 1)
        wp.atomic_add(pen_force_accum, 0, (lim - dist) * contact_stiffness)
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
                               "radius": radius*max(sx,sy,sz),
                               "sense": True})

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
                               "half_z": half*sz,
                               "sense": True})

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
                               "radius": radius*sx,
                               "sense": True})

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
                               "half_len": height*0.5*scale_h,
                               "sense": True})

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
                               "height": height,
                               "sense": True})

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
                                   "half_z": float(oh[2]),
                                   "sense": True})
            else:
                d = np.linalg.norm(wpts - cen, axis=1)
                colliders.append({"shape": SHAPE_SPHERE,
                                   "center": (float(cen[0]),float(cen[1]),
                                              float(cen[2])),
                                   "radius": float(d.max()),
                                   "sense": True})
    return colliders


# ===========================================================================
# SoftBody -- tet mesh XPBD, arbitrary box shape (half_x, half_y, half_z)
# ===========================================================================

class SoftBodyCube:
    """XPBD tet-mesh soft body with independent per-axis dimensions and
    resolution.  Works for any box aspect ratio -- flat pads, cubes, rods.

    Topology: Freudenthal 6-tet-per-cell on a res_x x res_y x res_z grid.
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

        # Geometry, kept around so world-space probe coordinates can be
        # converted into grid space for cutting.
        self.center = (cx, cy, cz)
        self.half_x, self.half_y, self.half_z = half_x, half_y, half_z

        # -- Cutting state ---------------------------------------------------
        # Cuts are tracked as a set of severed "walls" -- the shared faces
        # between adjacent grid cells in the XY plane. A severed wall is
        # no longer all-or-nothing through the full Z thickness: each
        # entry stores the DEEPEST grid layer the cut has reached (iz=0
        # is the bottom/base, iz=res_z-1 is the top/skin surface), so a
        # shallow touch only splits the top layer or two while pushing
        # the tool further down splits progressively deeper layers -- a
        # wall is severed at a given iz iff iz >= its stored threshold.
        # Repeated strokes only ever deepen a cut (min of old and new
        # threshold), never heal it shallower.
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

        # -- Particle grid ----------------------------------------------------
        lx = np.linspace(-half_x, half_x, res_x, dtype=np.float64) + cx
        ly = np.linspace(-half_y, half_y, res_y, dtype=np.float64) + cy
        lz = np.linspace(-half_z, half_z, res_z, dtype=np.float64) + cz
        gx, gy, gz = np.meshgrid(lx, ly, lz, indexing="ij")
        positions = np.stack(
            [gx.flatten(), gy.flatten(), gz.flatten()], axis=1
        ).astype(np.float64)

        inv_mass = np.full(n, n / total_mass, dtype=np.float32)

        # -- Tissue layer per grid node --------------------------------------
        # Every node's layer (skin/fat/muscle -- see TISSUE_LAYERS) is
        # derived purely from its Z depth fraction, which never needs to
        # be tracked/copied for duplicates the way the old boundary flag
        # was: a duplicate's REST position is already copied from its
        # source (see _alloc_duplicate_node), and layer is always
        # re-derivable from rest position on demand (_tissue_layer_index
        # below), so there's no separate per-particle list to keep in
        # sync here at all -- one less thing that could ever drift.
        local_z_grid = (gz.flatten() - cz)
        layer_idx_grid = self._tissue_layer_index(self._tissue_zfrac(local_z_grid))

        def vidx(ix, iy, iz):
            return ix * res_y * res_z + iy * res_z + iz

        # -- Weld bottom face (iz == 0) to the rigid base -- no slip, no separation --
        for ix in range(res_x):
            for iy in range(res_y):
                inv_mass[vidx(ix, iy, 0)] = 0.0

        # -- Tetrahedralization (Freudenthal 6-tet split) ----------------------
        # tet_cell[ti]    = (ix, iy) the XY cell this tet belongs to (its Z
        #                   layer doesn't matter for cutting -- a cut wall
        #                   always spans the full thickness).
        # tet_corners[ti] = the ORIGINAL (ix, iy, iz) grid coordinate of
        #                   each of the tet's 4 corners. These labels never
        #                   change even after a corner gets retargeted to a
        #                   duplicate particle id -- they're what let a cut
        #                   find "every tet corner currently at grid vertex
        #                   (vx, vy)" without needing to reverse-engineer
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

        # -- Per-tet tissue-layer stiffness (k_vol) --------------------------
        # A tet's layer is looked up from its CENTROID's rest Z -- fixed
        # forever once computed here, since tets never change identity or
        # position (cutting only ever re-wires which particle id each
        # corner references, never adds/removes/moves a tet), so this
        # only ever needs computing once, at construction.
        tet_centroid_local_z = (positions[tets[:,0],2] + positions[tets[:,1],2]
                                 + positions[tets[:,2],2] + positions[tets[:,3],2]) / 4.0 - cz
        tet_layer_idx = self._tissue_layer_index(self._tissue_zfrac(tet_centroid_local_z))
        tet_kvol = _TISSUE_KVOL[tet_layer_idx]

        # -- Unique tet edges -> distance constraints ---------------------------
        # Each edge's k_edge is looked up the same way, from its own
        # midpoint's rest Z (re-derived on every cut-triggered rebuild
        # too -- see _rebuild_structural_edges -- since new edges appear
        # there that don't exist yet at construction time).
        edge_set = set()
        for t in tets:
            for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                i,j = int(t[a]), int(t[b])
                if i>j: i,j=j,i
                edge_set.add((i,j))
        si = np.array([e[0] for e in edge_set], dtype=np.int32)
        sj = np.array([e[1] for e in edge_set], dtype=np.int32)
        sr = np.linalg.norm(positions[si]-positions[sj], axis=1).astype(np.float32)
        edge_mid_local_z = (positions[si,2] + positions[sj,2]) / 2.0 - cz
        edge_layer_idx = self._tissue_layer_index(self._tissue_zfrac(edge_mid_local_z))
        sk = _TISSUE_KEDGE[edge_layer_idx]
        self.num_springs = len(si)

        # -- Boundary surface extraction -----------------------------------------
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

        # Per-face color -- average of its 3 corners' own tissue-layer
        # color, so a face straddling a layer boundary blends smoothly
        # between them instead of hard-banding. This is what gives the
        # pad its layered look even before any cut: the outer side walls
        # span the full Z depth, so they show skin at the top edge,
        # fading through fat, down to muscle near the base -- a cut
        # exposing fat/muscle inside just continues the same coloring
        # rule the outer surface already used.
        particle_colors = _TISSUE_COLORS[layer_idx_grid]   # (n, 3), vidx-ordered
        if oriented:
            oriented_np = np.array(oriented, dtype=np.int64)
            tri_colors_init = particle_colors[oriented_np].mean(axis=1)
        else:
            tri_colors_init = np.zeros((0, 3), dtype=np.float32)

        # -- Upload to GPU --------------------------------------------------------
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
        self.tet_stiff  = wp.array(tet_kvol.astype(np.float32),        dtype=wp.float32, device=device)
        self.corr       = wp.zeros(n, dtype=wp.vec3,  device=device)
        self.corr_count = wp.zeros(n, dtype=wp.int32, device=device)

        # Force-feedback: net reaction the pad exerts back on the probe,
        # accumulated by collide_capsule_sensed over one step()'s worth
        # of substeps (see step() and force_sensor.py). Fixed size (1),
        # independent of particle count n, so it's never touched by the
        # cutting rebuild in _rebuild_gpu_arrays.
        self.force_accum      = wp.zeros(1, dtype=wp.vec3, device=device)
        self.last_probe_force = (0.0, 0.0, 0.0)   # raw Newtons, last step()
        # Diagnostics -- see collide_capsule_sensed docstring. contact_count
        # tells you whether the probe geometrically touched ANY particle
        # at all this step(); pen_force_raw is an independent force
        # estimate from penetration depth * CONTACT_STIFFNESS, to compare
        # against last_probe_force (the position-correction-based one)
        # when hunting a "touches but reads 0N" bug.
        self.contact_count      = wp.zeros(1, dtype=wp.int32, device=device)
        self.pen_force_accum    = wp.zeros(1, dtype=wp.float32, device=device)
        self.last_contact_count = 0
        self.last_pen_force_raw = 0.0

        # -- Python-side mutable mirrors, for cutting ------------------------------
        # (the wp arrays above are fixed-size; cutting adds particles and
        # constraints, so we keep growable Python lists here and rebuild
        # the GPU arrays from them whenever topology changes)
        self._pos_list      = pos32.tolist()
        # NOTE: use a list comprehension here, not `[[0.0,0.0,0.0]] * n` --
        # the latter would create n references to the SAME inner list
        # object rather than n independent [0,0,0] lists. It was harmless
        # in the original code only because every element also started
        # at the same value and nothing mutated an entry in place before
        # the first _sync_from_gpu() call replaced the whole list -- but
        # it's a latent aliasing bug, so it's fixed here for robustness.
        self._vel_list      = [[0.0, 0.0, 0.0] for _ in range(n)]
        self._rest_pos_list = pos32.tolist()   # undeformed positions; only
                                                 # ever grows (duplicates
                                                 # copy their origin's rest
                                                 # position) -- used so cut
                                                 # springs get correct rest
                                                 # lengths, not strained ones,
                                                 # AND so tissue layer is
                                                 # always re-derivable for
                                                 # any particle, duplicated
                                                 # or original (see
                                                 # _tissue_zfrac/_tissue_layer_index)
        self._inv_mass_list = inv_mass.tolist()
        self._tets_list     = tets.tolist()
        self._tet_vol_list   = vols.astype(np.float32).tolist()
        self._tet_stiff_list = tet_kvol.astype(np.float32).tolist()
        self._tri_list       = [list(t) for t in oriented]
        self._tri_color_list = tri_colors_init.tolist()
        self.tri_colors = np.array(self._tri_color_list, dtype=np.float32)

    def centroid(self):
        p = self.pos.numpy(); c = p.mean(0)
        return float(c[0]), float(c[1]), float(c[2])

    # ------------------------------------------------------------------
    # Tissue layers (see TISSUE_LAYERS) -- pure functions of Z depth, so
    # any particle's layer/stiffness/color is always re-derivable from
    # its rest position with no separate per-particle bookkeeping to
    # keep in sync (including for particles created later by cutting).
    # ------------------------------------------------------------------
    def _tissue_zfrac(self, local_z):
        """local_z: Z position(s) relative to this pad's center (i.e.
        rest_z - self.center[2]), ranging -half_z (bottom, welded to the
        rigid base) to +half_z (top, skin surface). Returns the
        top(0.0)-to-bottom(1.0) depth fraction TISSUE_LAYERS is defined
        against. Accepts a scalar or a numpy array."""
        return np.clip((self.half_z - local_z) / (2.0 * self.half_z), 0.0, 1.0)

    def _tissue_layer_index(self, zfrac):
        """zfrac -> index into TISSUE_LAYERS (0=skin, 1=fat, 2=muscle).
        Accepts a scalar or a numpy array."""
        idx = np.searchsorted(_TISSUE_BOUNDS, zfrac, side="left")
        return np.clip(idx, 0, len(TISSUE_LAYERS) - 1)

    # ------------------------------------------------------------------
    # Cutting: local, depth-aware wall severing + per-node group splitting.
    #
    # The XY footprint is a grid of (res_x-1) x (res_y-1) cells. Every pair
    # of adjacent cells shares a "wall". Dragging the probe through the
    # pad severs exactly the walls it actually crosses -- nothing else --
    # so a cut can run in any direction (not just along X), can curve, can
    # be applied piecemeal without racing ahead to fill in a whole column,
    # and never affects material the probe didn't actually pass through.
    #
    # DEPTH now matters too: a severed wall isn't all-or-nothing through
    # the full Z thickness. Each entry records the DEEPEST grid layer the
    # cut has reached so far (iz = res_z-1 is the top/skin surface, iz = 0
    # is the bottom, welded to the base) -- a wall is severed at a given
    # iz iff iz >= its stored threshold. A shallow graze near the top
    # only nicks the skin layer or two; pushing the tool further down
    # toward the base severs progressively deeper layers (through fat,
    # into muscle), reaching all the way through only at full
    # penetration. Repeated strokes only ever deepen a cut (min of old
    # and new threshold), never heal it shallower.
    #
    # At any grid NODE (a specific x/y/z point, not a whole column)
    # touched by a severed wall, the (up to 4) cells that meet there are
    # grouped by whichever ones are still connected through an un-severed
    # wall AT THAT DEPTH. If that grouping is finer than before, the
    # newly-separated group gets a fresh duplicate particle (copied from
    # wherever the original currently sits, then linked back to it with a
    # breakable cohesive constraint) while the rest of the material keeps
    # using the id it already had -- so already-separated material is
    # never reset or "un-cut" by a later cut elsewhere, and a shallow nick
    # only ever splits the node(s) at the depth it actually reached.
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
        """Convert a probe tip's world-space Z into the deepest grid
        layer a cut at this depth should reach (0 = bottom/base,
        res_z-1 = top/skin surface). This is what makes cutting
        depth-aware: a shallow touch near the top only severs the top
        layer or two (nicking skin, maybe grazing fat), while pushing
        the tool further down toward the rigid base severs progressively
        deeper layers -- through fat, into muscle -- reaching all the
        way through only at full penetration, instead of any touch at
        all slicing clean through the whole tissue stack."""
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
        partially re-split (see _retarget_vertex) since depth matters.

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
        # it (this loop is pure Python-side bookkeeping, not physics, so
        # oversampling here is cheap).
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
        cohesive constraint. Returns the new particle's id.

        No layer/color bookkeeping needed here -- rest position (copied
        below) is all _tissue_zfrac/_tissue_layer_index ever need, so a
        duplicate's tissue layer just falls out of the same lookup its
        source uses, automatically, forever."""
        new_id = len(self._pos_list)
        self._pos_list.append(list(self._pos_list[src_id]))
        self._vel_list.append(list(self._vel_list[src_id]))
        self._inv_mass_list.append(self._inv_mass_list[src_id])
        self._rest_pos_list.append(list(self._rest_pos_list[src_id]))
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

    def _rebuild_structural_edges(self, tets_np):
        """Recompute the tet-edge distance constraints from scratch from
        the CURRENT tets_np. Simpler and more robust than trying to
        incrementally patch an edge list: whichever tets a cut just
        retargeted onto different particle ids will naturally stop
        sharing an edge with their old neighbors here, with no extra
        bookkeeping.

        Vectorized with numpy rather than a per-tet Python loop + set,
        for the same reason noted in _rebuild_boundary_tris below --
        this runs every time a cut crosses a new cell wall, potentially
        every frame while actively dragging a cut. Dedup is done by
        packing each (i, j) pair into a single int64 key and running 1D
        np.unique on that -- np.unique(..., axis=0) on raw 2-column rows
        takes a much slower structured-array-sort code path in numpy;
        packing to a scalar key first and using plain 1D unique (a
        regular sort, no structured-view overhead) is dramatically
        cheaper at this row count. Returns plain lists (via .tolist())
        so every caller downstream is unchanged.
        """
        pair_idx = np.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
        all_edges = tets_np[:, pair_idx].reshape(-1, 2)
        all_edges = np.sort(all_edges, axis=1)

        n = len(self._pos_list)   # always current; self.num_particles isn't
                                    # updated until AFTER this call returns
        if n >= (1 << 31):
            # 32 bits/field gives an enormous budget (2 billion particles)
            # before this packing could overflow int64 -- unreachable in
            # practice, but fail loudly rather than silently wrap/corrupt
            # if it somehow ever is reached.
            raise RuntimeError(
                f"_rebuild_structural_edges: particle count {n} exceeds "
                f"the packed-edge-key dedup scheme's budget -- this soft "
                f"body has grown far beyond its intended scale.")
        keys = (all_edges[:, 0].astype(np.int64) << 32) | all_edges[:, 1].astype(np.int64)
        _, first_idx = np.unique(keys, return_index=True)
        uniq = all_edges[first_idx]

        rp = np.asarray(self._rest_pos_list, dtype=np.float64)
        i, j = uniq[:, 0], uniq[:, 1]
        lengths = np.linalg.norm(rp[i] - rp[j], axis=1)

        # Per-edge stiffness from its own midpoint's tissue layer -- an
        # edge freshly created by a cut (between an original particle and
        # its new duplicate, or between two duplicates) gets exactly the
        # same stiffness it would have had before the cut, since it's a
        # pure function of rest position, not something that needs
        # tracking/copying through _alloc_duplicate_node.
        mid_local_z = (rp[i, 2] + rp[j, 2]) / 2.0 - self.center[2]
        edge_layer_idx = self._tissue_layer_index(self._tissue_zfrac(mid_local_z))
        es_np = _TISSUE_KEDGE[edge_layer_idx]

        ei = i.tolist()
        ej = j.tolist()
        er = lengths.tolist()
        es = es_np.tolist()
        return ei, ej, er, es

    # Particle-id bit budget for the packed face key below -- 2**21 is
    # ~2 million particles, wildly more than this sim will ever reach
    # from cutting alone (it starts at a few thousand). Guarded by an
    # assert with an explicit fallback rather than silently producing
    # wrong results if that budget is ever actually exceeded.
    _FACE_KEY_BITS = 21

    def _rebuild_boundary_tris(self, tets_np):
        """Recompute the render-mesh boundary faces from scratch from the
        current tets_np (a face belongs to exactly one tet).

        Colors each face by averaging its 3 corners' own tissue-layer
        color (skin/fat/muscle -- see TISSUE_LAYERS), each derived purely
        from that corner's rest Z depth -- same rule used at construction
        (see __init__), so a cut exposing a deeper layer just shows that
        layer's color automatically; no separate boundary/skin flag to
        maintain. Returns (oriented_triangles, per_face_colors).

        Vectorized with numpy instead of a per-tet Python dict/Counter
        loop -- at ~15k tets (4 faces each = ~60k raw faces before
        dedup) the Python version costs several milliseconds *every time
        a cut crosses a new cell wall*, i.e. potentially every frame
        while actively dragging a cut. Dedup uses the same packed-int-key
        1D-unique trick as _rebuild_structural_edges above, for the same
        reason (np.unique(..., axis=0) is much slower per row here)."""
        face_idx = np.array([(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
        all_faces = tets_np[:, face_idx].reshape(-1, 3)   # original per-tet winding kept
        sorted_faces = np.sort(all_faces, axis=1)

        n = len(self._pos_list)   # always current; self.num_particles isn't
                                    # updated until AFTER this call returns
        bits = self._FACE_KEY_BITS
        if n >= (1 << bits):
            # Defensive fallback for the (currently unreachable) case of
            # an enormous particle count outgrowing the packed key budget.
            bits = int(np.ceil(np.log2(n + 1))) + 1
            if 3 * bits > 63:
                # Packing 3 fields of this width would silently overflow/
                # wrap the int64 key (numpy doesn't raise on shift
                # overflow) and corrupt face dedup rather than just being
                # slow -- fail loudly instead. This would need millions
                # of particles to ever actually trigger.
                raise RuntimeError(
                    f"_rebuild_boundary_tris: particle count {n} is too "
                    f"large for the packed-face-key dedup scheme (needs "
                    f"{3 * bits} bits, int64 has 63 usable) -- this soft "
                    f"body has grown far beyond its intended scale.")
        a64 = sorted_faces[:, 0].astype(np.int64)
        b64 = sorted_faces[:, 1].astype(np.int64)
        c64 = sorted_faces[:, 2].astype(np.int64)
        keys = (a64 << (2 * bits)) | (b64 << bits) | c64

        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        # A face belongs to exactly one tet iff its packed key occurs
        # exactly once across every tet's 4 faces -- same "v == 1" test
        # as the original dict-Counter version, just vectorized. Since
        # count==1 means that row is the ONLY occurrence of its key,
        # keeping the row's own (unsorted) winding is exactly equivalent
        # to the original code's "first (and only) encountered winding".
        boundary_mask = counts[inverse] == 1
        boundary = all_faces[boundary_mask]

        pos = np.array(self._pos_list, dtype=np.float32)
        gcen = pos.mean(0)
        ia, ib, ic = boundary[:, 0], boundary[:, 1], boundary[:, 2]
        pa, pb, pc = pos[ia], pos[ib], pos[ic]
        fn  = np.cross(pb - pa, pc - pa)
        mid = (pa + pb + pc) / 3.0
        flip = np.einsum('ij,ij->i', fn, mid - gcen) < 0.0

        oriented = boundary.copy()
        oriented[flip, 1] = boundary[flip, 2]
        oriented[flip, 2] = boundary[flip, 1]

        rp = np.asarray(self._rest_pos_list, dtype=np.float64)
        local_z_all = rp[:, 2] - self.center[2]
        layer_idx_all = self._tissue_layer_index(self._tissue_zfrac(local_z_all))
        particle_colors = _TISSUE_COLORS[layer_idx_all]   # (n_particles, 3)
        colors = particle_colors[oriented].mean(axis=1)    # (F, 3), blends at layer seams

        return oriented.tolist(), colors.tolist()

    def _rebuild_gpu_arrays(self):
        """Re-upload all GPU arrays from the current Python-side lists.
        Called after cut_segment() or a cohesive break changes topology.
        Assumes self._pos_list / self._vel_list are already current (via
        _sync_from_gpu, plus any new columns appended on top)."""
        tets_np = np.array(self._tets_list, dtype=np.int64)

        struct_i, struct_j, struct_r, struct_s = self._rebuild_structural_edges(tets_np)
        cohesive_i = [c["a"] for c in self.cohesive]
        cohesive_j = [c["b"] for c in self.cohesive]
        edge_i = struct_i + cohesive_i
        edge_j = struct_j + cohesive_j
        edge_r = struct_r + [0.0] * len(cohesive_i)
        edge_s = struct_s + [CUT_COHESIVE_STIFFNESS] * len(cohesive_i)

        self._tri_list, self._tri_color_list = self._rebuild_boundary_tris(tets_np)

        n = len(self._pos_list)
        self.num_particles = n
        self.num_springs = len(edge_i)
        self.num_tets = len(self._tets_list)

        pos_np  = np.array(self._pos_list, dtype=np.float32)
        vel_np  = np.array(self._vel_list, dtype=np.float32)

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

    # ------------------------------------------------------------------
    # Collider packing -- see step()/`_pack_collider` for why this exists.
    # ------------------------------------------------------------------
    def _pack_collider(self, c: dict):
        """Convert a collider dict's raw Python floats/tuples into wp.vec3
        objects ONCE per step() call.

        Previously this conversion happened inside _dispatch_collider,
        which is invoked once per solver iteration -- SUBSTEPS *
        SOLVER_ITERS times per frame (120 with the defaults below). Since
        a collider's world transform doesn't change mid-step, that meant
        rebuilding the same wp.vec3 objects up to 120x per frame per
        collider for no reason -- pure wasted CPU-side Python object
        construction sitting in the hot path. Packing once here and
        reusing the packed dict across every inner iteration removes that
        redundant work without changing any collision math.
        """
        sh = c["shape"]
        if sh == SHAPE_SPHERE:
            return {"shape": sh,
                    "center": wp.vec3(*c["center"]),
                    "radius": float(c["radius"]),
                    "sense": bool(c.get("sense", False))}
        elif sh == SHAPE_BOX:
            r0, r1, r2 = c["row0"], c["row1"], c["row2"]
            return {"shape": sh,
                    "center": wp.vec3(*c["center"]),
                    "row0": wp.vec3(float(r0[0]), float(r0[1]), float(r0[2])),
                    "row1": wp.vec3(float(r1[0]), float(r1[1]), float(r1[2])),
                    "row2": wp.vec3(float(r2[0]), float(r2[1]), float(r2[2])),
                    "half_x": float(c["half_x"]),
                    "half_y": float(c["half_y"]),
                    "half_z": float(c["half_z"]),
                    "sense": bool(c.get("sense", False))}
        elif sh == SHAPE_CAPSULE:
            return {"shape": sh,
                    "p0": wp.vec3(*c["p0"]),
                    "p1": wp.vec3(*c["p1"]),
                    "radius": float(c["radius"]),
                    "sense": bool(c.get("sense", False))}
        elif sh == SHAPE_CYLINDER:
            return {"shape": sh,
                    "center": wp.vec3(*c["center"]),
                    "axis": wp.vec3(*c["axis"]),
                    "collide_radius": float(c.get("collide_radius", c["radius"])),
                    "half_len": float(c["half_len"]),
                    "sense": bool(c.get("sense", False))}
        elif sh == SHAPE_CONE:
            return {"shape": sh,
                    "apex": wp.vec3(*c["apex"]),
                    "axis": wp.vec3(*c["axis"]),
                    "half_angle": float(c["half_angle"]),
                    "height": float(c["height"]),
                    "sense": bool(c.get("sense", False))}
        # Unknown shape -- pass through unchanged (dispatch will just
        # silently no-op on it, same as before).
        return c

    def _dispatch_collider(self, c: dict, friction: float):
        """c is a PRE-PACKED collider (see _pack_collider): its vec3
        fields are already real wp.vec3 objects, so this only launches
        the matching kernel -- no per-call Python object construction."""
        sh = c["shape"]
        sensed = c.get("sense", False)
        # Common tail of args shared by every _sensed kernel variant --
        # kept in one place so the "any shape can report force" wiring
        # can't drift out of sync between shapes.
        sense_args = [self._inv_sub_dt2, self.force_accum,
                      self.contact_count, self.pen_force_accum,
                      float(CONTACT_STIFFNESS)]
        if sh == SHAPE_SPHERE:
            if sensed:
                wp.launch(collide_sphere_sensed, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"],c["radius"],
                                   float(SKIN),float(friction)] + sense_args,
                          device=self.device)
            else:
                wp.launch(collide_sphere, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"],c["radius"],
                                   float(SKIN),float(friction)],
                          device=self.device)
        elif sh == SHAPE_BOX:
            if sensed:
                wp.launch(collide_box_sensed, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"], c["row0"], c["row1"], c["row2"],
                                   c["half_x"], c["half_y"], c["half_z"],
                                   float(SKIN),float(friction)] + sense_args,
                          device=self.device)
            else:
                wp.launch(collide_box, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"], c["row0"], c["row1"], c["row2"],
                                   c["half_x"], c["half_y"], c["half_z"],
                                   float(SKIN),float(friction)],
                          device=self.device)
        elif sh == SHAPE_CAPSULE:
            if sensed:
                wp.launch(collide_capsule_sensed, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["p0"], c["p1"],
                                   c["radius"],float(SKIN),float(friction)] + sense_args,
                          device=self.device)
            else:
                wp.launch(collide_capsule, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["p0"], c["p1"],
                                   c["radius"],float(SKIN),float(friction)],
                          device=self.device)
        elif sh == SHAPE_CYLINDER:
            if sensed:
                wp.launch(collide_cylinder_sensed, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"], c["axis"],
                                   c["collide_radius"], c["half_len"],
                                   float(SKIN),float(friction)] + sense_args,
                          device=self.device)
            else:
                wp.launch(collide_cylinder, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["center"], c["axis"],
                                   c["collide_radius"], c["half_len"],
                                   float(SKIN),float(friction)],
                          device=self.device)
        elif sh == SHAPE_CONE:
            if sensed:
                wp.launch(collide_cone_sensed, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["apex"], c["axis"],
                                   c["half_angle"], c["height"],
                                   float(SKIN),float(friction)] + sense_args,
                          device=self.device)
            else:
                wp.launch(collide_cone, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,self.vel,
                                   c["apex"], c["axis"],
                                   c["half_angle"], c["height"],
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

        # Force-feedback bookkeeping for this step() call -- see
        # collide_capsule_sensed / force_sensor.py. inv_sub_dt2 is the
        # h^-2 term in F = m*dx/h^2; the accumulator is cleared once per
        # step() (not per substep) so the average below is the mean
        # contact force over the whole frame, across all substeps.
        self._inv_sub_dt2 = 1.0 / (sub_dt * sub_dt)
        wp.launch(zero_vec3_single, dim=1, inputs=[self.force_accum], device=self.device)
        wp.launch(zero_int32_single, dim=1, inputs=[self.contact_count], device=self.device)
        wp.launch(zero_float32_single, dim=1, inputs=[self.pen_force_accum], device=self.device)

        # Pack every collider's wp.vec3 fields ONCE per step() call --
        # was previously done inside the inner substeps*solver_iters loop
        # (see _pack_collider docstring above for why that was wasteful).
        packed_base       = self._pack_collider(base_box) if base_box is not None else None
        packed_colliders  = [self._pack_collider(c) for c in colliders]

        # ---- Kernel-launch-count optimization -----------------------------
        # At this particle/tet count (thousands, not millions), a single
        # kernel launch finishes its actual GPU work in far less time than
        # the CPU takes to dispatch the NEXT one -- so total frame time is
        # dominated by *how many* launches happen, not how much math is in
        # each. The previous structure launched the 4 core solver kernels
        # AND every collider kernel once per solver ITERATION (substeps *
        # solver_iters = up to 120 times/frame with the defaults below),
        # so a single collider alone could cost 120 launches/frame.
        #
        # Collision is moved to run once per SUBSTEP instead (12x/frame
        # instead of 120x/frame with the defaults) -- collision still gets
        # re-resolved every substep, which is the standard structure used
        # by real-time PBD/XPBD engines (Gauss-Seidel-style internal
        # constraint iteration happens several times per substep; contact
        # projection happens once per substep, after the internal solve
        # has had a chance to converge). This is the single biggest lever
        # here: it cuts collider-related launches by up to solver_iters x
        # with no change to any constraint's math and no change to how
        # often collisions are actually checked against wall-clock time.
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

            # Collision, once per substep (see note above) -- rigid base,
            # ground, and every external/probe collider.
            if packed_base is not None:
                self._dispatch_collider(packed_base, friction)

            if ground_z is not None:
                wp.launch(collide_ground, dim=self.num_particles,
                          inputs=[self.pred,self.inv_mass,
                                  float(ground_z),float(friction),self.vel],
                          device=self.device)
                wp.launch(pin_ground_contacts, dim=self.num_particles,
                          inputs=[self.pos,self.pred,self.inv_mass,
                                  float(ground_z),float(static_friction)],
                          device=self.device)

            for col in packed_colliders:
                self._dispatch_collider(col, friction)

            wp.launch(update_velocity, dim=self.num_particles,
                      inputs=[self.pos,self.pred,self.vel,
                               self.inv_mass,1.0/sub_dt,damping],
                      device=self.device)

        # Mean contact force over this frame's substeps -- raw Newtons,
        # not yet run through ForceFeedbackSensor's calibration/tare/
        # noise/smoothing (see get_probe_force_raw()).
        fa = self.force_accum.numpy()[0]
        self.last_probe_force = (float(fa[0]) / substeps,
                                  float(fa[1]) / substeps,
                                  float(fa[2]) / substeps)
        # Diagnostics -- see collide_capsule_sensed / get_probe_debug_info().
        self.last_contact_count = int(self.contact_count.numpy()[0])
        self.last_pen_force_raw = float(self.pen_force_accum.numpy()[0]) / substeps

    def get_probe_force_raw(self):
        """(Fx, Fy, Fz) in Newtons, recovered from this frame's contact
        corrections against the probe capsule (see collide_capsule_sensed).
        Real Newton-scale numbers, but not yet scale-matched to a real F/T
        sensor -- feed them through ForceFeedbackSensor for that."""
        return self.last_probe_force

    def get_probe_debug_info(self):
        """(contact_count, pen_force_raw_n) for this frame -- see
        collide_capsule_sensed's docstring. contact_count > 0 means the
        probe geometrically touched at least one particle this step();
        pen_force_raw is an independent penetration-depth-based force
        estimate (N), which should stay nonzero for as long as
        contact_count > 0, even if get_probe_force_raw() has decayed to
        ~0 at a steady hold."""
        return self.last_contact_count, self.last_pen_force_raw


# ===========================================================================
# Helpers
# ===========================================================================

def _vec3f_list(arr: np.ndarray):
    return [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in arr]


def _vec3f_array(arr: np.ndarray):
    """Build a VtVec3fArray directly from a numpy buffer instead of
    looping through Python and constructing one Gf.Vec3f per element.

    This is the hot path called every single frame for particle
    positions (and on topology-change frames for face colors), so
    avoiding a Python-level per-particle loop matters as particle count
    grows from cutting. Falls back to the slow per-element path only if
    this USD build's Vt bindings don't accept a numpy buffer directly.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        return Vt.Vec3fArray(arr)
    except Exception:
        return _vec3f_list(arr)


def _bind_matte_material(stage, prim, mat_path, fallback_color, use_vertex_color=True):
    """Bind a fully flat/matte material to `prim`: no specular highlight,
    no clearcoat, non-metallic. Without this, a mesh with no material
    bound picks up the renderer's default preview material, which DOES
    include a specular response under the dome light -- every visible
    mesh in the scene (pad, scalpel, rigid base) was picking up an
    unwanted glossy/plastic highlight for no functional reason, which
    costs extra shading work per pixel for zero benefit here (there's no
    "shiny material" this sim actually needs). Reused across every
    visible mesh instead of duplicating the same shader graph per-mesh.

    If use_vertex_color, diffuse albedo is read from the mesh's own
    "displayColor" primvar (so per-face skin/muscle or steel/grip color
    still works); otherwise it's a flat fallback_color everywhere (used
    for the plain rigid base, which has no per-face primvar).
    """
    mat_path = Sdf.Path(mat_path)
    material = UsdShade.Material.Define(stage, mat_path)
    surf = UsdShade.Shader.Define(stage, mat_path.AppendChild("PreviewSurface"))
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0))
    surf.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    surf.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(1.0)
    surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

    if use_vertex_color:
        color_reader = UsdShade.Shader.Define(stage, mat_path.AppendChild("ColorReader"))
        color_reader.CreateIdAttr("UsdPrimvarReader_float3")
        color_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        color_reader.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*fallback_color))
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            color_reader.ConnectableAPI(), "result")
    else:
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*fallback_color))

    material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return material


def _build_scalpel_tool_geometry():
    """Build ONE combined mesh -- blade + handle -- as a single set of
    points/faceVertexCounts/faceVertexIndices/per-face-colors, so the
    whole tool is structurally one prim (see the design note by
    BLADE_LENGTH above for why that matters).

    Local space matches the probe's own local space: Z is the tool's
    long axis (tip at BLADE_TIP_LOCAL_Z, handle butt at
    HANDLE_BUTT_LOCAL_Z), X/Y are cross-section width/thickness. Returns
    (points, face_vertex_counts, face_vertex_indices, face_colors).
    """
    z_tip      = BLADE_TIP_LOCAL_Z
    z_belly    = BLADE_TIP_LOCAL_Z + 0.55 * BLADE_LENGTH   # widest point (edge)
    z_shoulder = BLADE_TIP_LOCAL_Z + 0.85 * BLADE_LENGTH   # necks in toward handle
    z_join     = BLADE_BUTT_LOCAL_Z                        # meets the handle

    w_belly    = 2.0 * HANDLE_RADIUS   # widest point of the cutting edge
    w_neck     = 0.9 * HANDLE_RADIUS
    w_join     = 0.6 * HANDLE_RADIUS   # necks down to meet the handle radius
    half_thick = 0.35 * HANDLE_RADIUS  # blade thickness

    # -- Blade: star-shaped outline (X, Z), extruded with a small
    # thickness in Y, front/back caps + side walls. --
    outline = [
        (0.0,       z_tip),
        (+w_belly,  z_belly),
        (+w_neck,   z_shoulder),
        (+w_join,   z_join),
        (-w_join,   z_join),
        (-w_neck,   z_shoulder),
        (-w_belly,  z_belly),
    ]
    m = len(outline)
    ocx = sum(p[0] for p in outline) / m
    ocz = sum(p[1] for p in outline) / m

    points = []
    for (x, z) in outline:
        points.append((x, +half_thick, z))     # front copy: 0..m-1
    for (x, z) in outline:
        points.append((x, -half_thick, z))     # back copy:  m..2m-1
    front_c = len(points); points.append((ocx, +half_thick, ocz))
    back_c  = len(points); points.append((ocx, -half_thick, ocz))

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
    blade_face_count = len(tris)

    # -- Handle: a plain round cylinder (as requested -- nothing fancy),
    # from the blade join up to the grip butt, with end caps. --
    HANDLE_SIDES = 14
    ring_bottom, ring_top = [], []
    for k in range(HANDLE_SIDES):
        ang = 2.0 * math.pi * k / HANDLE_SIDES
        hx, hy = HANDLE_RADIUS * math.cos(ang), HANDLE_RADIUS * math.sin(ang)
        ring_bottom.append(len(points)); points.append((hx, hy, z_join))
    for k in range(HANDLE_SIDES):
        ang = 2.0 * math.pi * k / HANDLE_SIDES
        hx, hy = HANDLE_RADIUS * math.cos(ang), HANDLE_RADIUS * math.sin(ang)
        ring_top.append(len(points)); points.append((hx, hy, HANDLE_BUTT_LOCAL_Z))
    bottom_c = len(points); points.append((0.0, 0.0, z_join))
    top_c    = len(points); points.append((0.0, 0.0, HANDLE_BUTT_LOCAL_Z))

    for k in range(HANDLE_SIDES):
        k2 = (k + 1) % HANDLE_SIDES
        # side quad, split into two tris
        tris.append((ring_bottom[k], ring_bottom[k2], ring_top[k2]))
        tris.append((ring_bottom[k], ring_top[k2], ring_top[k]))
        # bottom cap (facing -Z, toward the blade)
        tris.append((bottom_c, ring_bottom[k2], ring_bottom[k]))
        # top cap (facing +Z, the grip butt)
        tris.append((top_c, ring_top[k], ring_top[k2]))

    face_vertex_counts = [3] * len(tris)
    face_vertex_indices = [idx for tri in tris for idx in tri]

    steel, grip = list(STEEL_COLOR), list(PROBE_COLOR)
    face_colors = [steel] * blade_face_count + [grip] * (len(tris) - blade_face_count)
    return points, face_vertex_counts, face_vertex_indices, face_colors





# ===========================================================================
# WarpSoftBodySim -- Isaac Sim scenario
# ===========================================================================

class WarpSoftBodySim:
    """Flat soft-body pad (skin-colored) resting on a rigid black base.

    Probe (mouse/viewport-dragged) pokes the top surface.
    Any other prim with CollisionAPI also interacts via gather_colliders.
    """

    # How many frames to reuse a cached stage-wide collider scan before
    # re-scanning. gather_colliders() walks the ENTIRE stage every time
    # it's called, which is wasted work on frames where nothing besides
    # the softbody/probe/base (already excluded) has moved. If you have
    # OTHER dynamic rigid bodies that need to interact with the pad every
    # single frame, lower this (e.g. to 1-3) so they stay responsive.
    COLLIDER_RESCAN_EVERY = 15

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
        self._cut_last_xy        = None   # last probe XY while engaged in a
                                           # cut, used by cut_segment tracing

        # Throttled stage-wide collider scan cache (see COLLIDER_RESCAN_EVERY).
        self._collider_cache          = []
        self._collider_rescan_counter = 0

        # Adaptive solver-iteration controller state (see TARGET_FRAME_MS
        # / SOLVER_ITERS_MIN above). Starts at the ceiling and only ever
        # backs off once actual measured cost says it needs to.
        self._adaptive_iters   = SOLVER_ITERS
        self._step_ms_ema       = None

        # Force feedback -- see force_sensor.py. _sim_time is a free-
        # running clock (seconds) fed by update(step), used only to
        # timestamp the force history; it never resets on its own so a
        # long session's plot keeps a stable time axis. reset_force_trace()
        # clears the sensor's history (call it whenever you want the plot
        # to start over, e.g. on RESET).
        # Clean/live sensor: no synthetic noise, no tare (see
        # force_sensor.py's __init__ docstring -- the old defaults added
        # ~0.5N of Gaussian noise to EVERY reading, which is the same
        # order of magnitude as the real recovered contact signal at this
        # rod radius, so it was drowning out actual touch response and
        # making rest readings look like random noise). Once you've
        # confirmed the raw signal tracks touch (see calibrate_from_real_reading
        # below and force_sensor property), use
        # ForceFeedbackSensor.for_report_matching() for a *separate*
        # sensor instance when building your final sim-vs-real figure.
        self._force_sensor = ForceFeedbackSensor()
        self._sim_time     = 0.0

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
        _bind_matte_material(stage, gnd.GetPrim(),
                              GROUND_PRIM_PATH + "/MatteMaterial",
                              (0.25, 0.25, 0.25), use_vertex_color=False)

        # Rigid base -- black UsdGeom.Cube with RigidBody + Collision
        base = UsdGeom.Cube.Define(stage, BASE_PRIM_PATH)
        base.CreateSizeAttr(1.0)
        xfb = UsdGeom.Xformable(base.GetPrim())
        xfb.AddTranslateOp().Set(Gf.Vec3d(*BASE_CENTER))
        xfb.AddScaleOp().Set(Gf.Vec3f(
            BASE_HALF_X * 2, BASE_HALF_Y * 2, BASE_HALF_Z * 2))
        # Static rigid body (kinematic, no gravity)
        rba = UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        rba.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(base.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(base.GetPrim()).CreateApproximationAttr("convexHull")
        _bind_matte_material(stage, base.GetPrim(),
                              BASE_PRIM_PATH + "/MatteMaterial",
                              (0.05, 0.05, 0.05), use_vertex_color=False)

        # Pre-build the base collider dict -- never changes, no need to scan it
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

        # Probe -- a scalpel: blade (working end, slim collision radius)
        # + handle (grip end, visual only), built as ONE mesh prim -- see
        # the design note above BLADE_LENGTH for why this is one prim
        # rather than a blade+handle assembly of separate prims.
        # No scale op needed -- the mesh's own vertex data already has
        # the tool's real-world dimensions baked in, so the only xformOp
        # on this prim we ever author is translate (rotate/orient ops
        # the viewport gizmo may add are excluded from the transform
        # order each frame, not zeroed -- see _clear_prim_xform).
        tool_pts, tool_fc, tool_fi, tool_colors = _build_scalpel_tool_geometry()
        probe = UsdGeom.Mesh.Define(stage, PROBE_PRIM_PATH)
        # Not double-sided: this is a closed, watertight solid (blade +
        # handle both have proper end caps) whose faces already all
        # point outward, so there's never a legitimate camera angle that
        # needs to see a backface -- unlike the soft body pad, which
        # stays double-sided because a cut can open a gap thin enough to
        # expose a backface at grazing angles. Disabling it here means
        # the renderer shades roughly half as many fragments for this
        # mesh, for a shape that never benefited from the other half.
        probe.CreateDoubleSidedAttr(False)
        probe.CreatePointsAttr([Gf.Vec3f(*p) for p in tool_pts])
        probe.CreateFaceVertexCountsAttr(tool_fc)
        probe.CreateFaceVertexIndicesAttr(tool_fi)
        xfp = UsdGeom.Xformable(probe.GetPrim())
        self._probe_translate_op = xfp.AddTranslateOp()
        self._probe_translate_op.Set(Gf.Vec3d(*PROBE_CENTER))
        rba2 = UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        rba2.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        # Mesh colliders need an approximation (same pattern already used
        # for the rigid base cube above) -- PhysX's own handling of this
        # prim is essentially vestigial anyway, since the actual
        # poke/cut collision against the soft body is our own analytic
        # capsule below, read directly from this prim's translate op.
        UsdPhysics.MeshCollisionAPI.Apply(probe.GetPrim()).CreateApproximationAttr("convexHull")
        UsdGeom.PrimvarsAPI(probe.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.uniform).Set([Gf.Vec3f(*c) for c in tool_colors])
        _bind_matte_material(stage, probe.GetPrim(),
                              PROBE_PRIM_PATH + "/MatteMaterial",
                              tuple(STEEL_COLOR), use_vertex_color=True)
        self._probe = probe

        # Soft body render mesh (points written each frame). Color is a
        # per-face primvar (skin vs. muscle), authored once self._cube
        # exists -- see _spawn().
        cb = UsdGeom.Mesh.Define(stage, SOFT_BODY_PRIM_PATH)
        cb.CreateDoubleSidedAttr(True)
        self._cube_mesh = cb

        # Flat/matte material -- see _bind_matte_material for why every
        # visible mesh in this scene gets one. The per-face skin/muscle
        # displayColor primvar still drives the diffuse albedo (via a
        # primvar-reader node), so cut visualization is unaffected.
        _bind_matte_material(stage, cb.GetPrim(),
                              SOFT_BODY_PRIM_PATH + "/MatteMaterial",
                              tuple(SKIN_TISSUE_COLOR), use_vertex_color=True)

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
        self._device      = self._select_device()
        self._xform_cache = UsdGeom.XformCache()
        self._spawn()

    def _select_device(self):
        """Prefer a second CUDA GPU for Warp physics so it doesn't contend
        with whichever GPU is driving the RTX viewport. Falls back to
        Warp's normal preferred-device pick if only one GPU (or no CUDA
        device) is available -- this is a pure "use what's there" upgrade,
        never a hard requirement."""
        try:
            cuda_devices = wp.get_cuda_devices()
            if cuda_devices and len(cuda_devices) > 1:
                return cuda_devices[1]
        except Exception as e:
            carb.log_warn(f"[WarpSoftBody] multi-GPU device probe failed, "
                           f"falling back to preferred device: {e}")
        return wp.get_preferred_device()

    def reset(self):
        self._spawn()
        self.reset_force_trace()

    # ---- Force feedback (for UI_builder.py) --------------------------
    def get_force_reading(self):
        """(t_seconds, force_z_newtons) for the most recent frame, or
        None if nothing has been recorded yet."""
        return self._force_sensor.latest()

    def get_force_history(self):
        """(t_array, force_z_array) -- full rolling trace, newest last."""
        return self._force_sensor.history_arrays()

    def get_force_peak(self):
        """(peak_force_n, t_of_peak) over the current trace, or None."""
        return self._force_sensor.peak()

    def get_force_baseline(self):
        """Mean force over the current trace's resting (non-contact)
        samples."""
        return self._force_sensor.baseline_mean()

    def reset_force_trace(self):
        self._force_sensor.reset()

    def export_force_trace_csv(self, path: str):
        self._force_sensor.export_csv(path)

    @property
    def force_sensor(self):
        """Direct access to the ForceFeedbackSensor, e.g. to retune
        calibration_gain/tare_n/noise_std_n from the UI or a console."""
        return self._force_sensor

    def get_force_reading_raw(self):
        """(t_seconds, raw_force_z_newtons) for the most recent frame --
        the UNCALIBRATED signal straight out of get_probe_force_raw(),
        before tare/gain/noise. Watch THIS while poking the pad: if it
        moves as you press in and settles back near 0 when you lift off,
        the physics/contact recovery itself is fine and any remaining
        "randomness" you see elsewhere is the sensor model (noise/tare/
        gain), not the sim. If this stays ~0 no matter how hard you
        press, the probe isn't actually registering contact (check rod
        position/radius, SKIN margin, and that the capsule collider is
        being appended in _update_impl)."""
        t, f = self._force_sensor.raw_history_arrays()
        if t.size == 0:
            return None
        return float(t[-1]), float(f[-1])

    def get_probe_debug_info(self):
        """(contact_count, pen_force_raw_n) for the most recent physics
        step -- see WarpSoftBodyCube.get_probe_debug_info(). Use this to
        tell apart "the probe never geometrically touches the mesh"
        (contact_count stays 0) from "it touches, but the m*dx/h^2 force
        recovery still reads ~0" (contact_count > 0, pen_force_raw_n
        nonzero, get_force_reading_raw() near 0 anyway)."""
        if self._cube is None:
            return 0, 0.0
        return self._cube.get_probe_debug_info()

    def calibrate_from_real_reading(self, reading_path: str, **find_kwargs):
        """One-call calibration: call this right after driving ONE full
        press-through-the-pad stroke in the live sim (noise/tare already
        off by default -- see _spawn()). Reads this run's raw contact
        peak, the matching real TISSUE-ONLY peak from a supervisor
        reading file (table-contact artifact excluded automatically --
        see force_sensor.find_tissue_only_region), and sets
        self._force_sensor.calibration_gain so the two match. Returns the
        fitted gain.

        NOTE: calibrates against raw_peak(), not peak() -- peak() is
        already tare/gain/noise-processed, so calibrating against it
        would double-apply tare/gain. See ForceFeedbackSensor.raw_peak().
        """
        from .force_sensor import real_tissue_peak_from_reading, calibrate_gain

        raw = self._force_sensor.raw_peak()
        if raw is None:
            raise RuntimeError(
                "no contact recorded yet -- press into the pad once "
                "before calibrating")
        sim_raw_peak_n, _t = raw
        real_peak_n, _baseline = real_tissue_peak_from_reading(
            reading_path, **find_kwargs)
        gain = calibrate_gain(sim_raw_peak_n, self._force_sensor.tare_n,
                               real_peak_n)
        self._force_sensor.calibration_gain = gain
        return gain

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

        # Gather external colliders (skip base and probe -- handled separately).
        # Throttled: gather_colliders() walks the ENTIRE stage, so on
        # frames where we don't rescan we just reuse last scan's result.
        # See COLLIDER_RESCAN_EVERY docstring for when to tighten this.
        self._collider_rescan_counter += 1
        if (self._collider_rescan_counter >= self.COLLIDER_RESCAN_EVERY
                or not self._collider_cache):
            self._xform_cache.Clear()
            skip = _OWN_PATHS | {SOFT_BODY_PRIM_PATH, PROBE_PRIM_PATH, BASE_PRIM_PATH}
            self._collider_cache = gather_colliders(stage, skip, self._xform_cache)
            self._collider_rescan_counter = 0
        colliders = list(self._collider_cache)

        # Probe collider from translate op -- the rod is a vertical
        # capsule, so its collider is defined by the two endpoints of its
        # centerline (p0 = top, p1 = bottom tip) rather than a box.
        #
        # p0/p1 span only the BLADE portion (BLADE_BUTT_LOCAL_Z down to
        # the tip) -- not the full tool including the handle -- so the
        # handle visually rides above the pad without ever deforming or
        # cutting it, same as a real scalpel's grip.
        probe_world = None
        probe_tip_z = None
        if self._probe is not None:
            p = self._probe_translate_op.Get()
            cx, cy, cz = float(p[0]), float(p[1]), float(p[2])
            probe_world = (cx, cy, cz)
            probe_tip_z = cz + BLADE_TIP_LOCAL_Z   # bottom end -- the working tip
            colliders.append({
                "shape":  SHAPE_CAPSULE,
                "p0":     (cx, cy, cz + BLADE_BUTT_LOCAL_Z),
                "p1":     (cx, cy, probe_tip_z),
                "radius": float(ROD_RADIUS),
                "sense":  True,   # this is the probe -- recover force feedback
            })

        # ---- Physics step FIRST -- base collision passed separately so
        # it runs every solver iteration (same priority as ground
        # constraint).
        #
        # IMPORTANT ORDERING: this used to run AFTER the cutting block
        # below, which meant a cut fired THIS frame (at wherever the tip
        # currently is) before collision ever got to push back against
        # that same tissue -- so collide_capsule_sensed was resolving
        # contact against topology that had just been severed out from
        # under it, and the force sensor never saw anything but 0N (see
        # CUT_ENGAGE_DEPTH docstring above for the full story). Stepping
        # physics first means this frame's collision/force-sensing always
        # sees the topology as it was BEFORE any cut this frame fires;
        # the cut then applies for next frame's render, one frame later,
        # which is imperceptible at 60fps and is the same lag every other
        # deferred-mutation system in this file already tolerates.
        #
        # solver_iters comes from the adaptive controller (see
        # TARGET_FRAME_MS/_adaptive_iters): timed end-to-end below,
        # including the syncing .numpy() calls, since those forced
        # GPU->CPU syncs are part of the real per-frame cost this
        # controller is trying to budget against.
        _t0 = time.perf_counter()
        self._cube.step(
            dt=DT,
            substeps=SUBSTEPS,
            solver_iters=self._adaptive_iters,
            gravity=(0.0, 0.0, -9.81),
            damping=0.995,
            ground_z=None,         # soft body never touches ground directly
            base_box=self._base_box,
            friction=0.85,
            static_friction=0.98,
            colliders=colliders,
        )
        self._cube._check_cohesive_breaks()

        # Force feedback: pull this frame's raw recovered force (Newtons)
        # and run it through the sensor model (calibration gain, tare,
        # noise, smoothing) -- see force_sensor.py.
        #
        # WHY pen_force_raw and not the plain m*dx/h^2 value: the m*dx/h^2
        # recovery (get_probe_force_raw()) only sees a nonzero number the
        # instant a particle is FIRST pushed -- once contact has settled
        # even slightly (which happens at ANY press speed, not just slow
        # ones -- a fast flick just makes it settle after a bigger first
        # spike), each new substep's *incremental* correction shrinks
        # toward 0 even though real contact is still happening. This is
        # the documented failure mode in collide_capsule_sensed's own
        # docstring.
        #
        # Earlier this only fell back to pen_force_raw when m_dx_h2 was
        # essentially exactly 0.0 -- but at a normal, un-flicked press
        # speed m_dx_h2 typically settles to a small NONZERO-but-noisy
        # residual (not exactly 0), so that condition silently kept using
        # the noisy, unreliable m_dx_h2 number instead of ever falling
        # back. Fast, high-velocity drags only "worked" because the
        # violent motion produced large transient m_dx_h2 spikes that
        # happened to be big enough to look like a real signal -- not
        # because the underlying method became more correct.
        #
        # Fix: whenever there IS real contact (contact_count > 0), always
        # use pen_force_raw -- a direct penetration-depth *
        # CONTACT_STIFFNESS spring estimate (see collide_capsule_sensed)
        # that stays proportional to how deep the probe actually sits,
        # at any press speed, transient or held. Negated here
        # (pen_force_raw is an unsigned magnitude that's positive while
        # penetrating) to match the real sensor's convention where
        # pressing into the pad reads NEGATIVE.
        self._sim_time += step
        _fx, _fy, _fz = self._cube.get_probe_force_raw()
        _contacts, _pen_force = self._cube.get_probe_debug_info()
        effective_fz = -_pen_force if _contacts > 0 else 0.0
        self._force_sensor.record(self._sim_time, effective_fz)

        # TEMPORARY debug instrumentation -- see get_probe_debug_info().
        # Prints every ~0.5s (30 frames at 60fps) so it's readable instead
        # of spamming every frame. Remove once the 0N-force bug is
        # confirmed fixed; this is purely to tell "the probe never
        # geometrically touches anything" apart from "it touches, but the
        # recovered force collapses to ~0 anyway" -- those look identical
        # from the Raw/Force_Z labels alone.
        #
        # ALSO logs whether the probe tip is even over the pad's XY
        # footprint and past its top surface in Z -- the most common
        # reason for a permanent 0.0 N reading isn't a code bug at all,
        # it's the probe having been dragged beside the pad (outside
        # +/-SOFT_HALF_X / +/-SOFT_HALF_Y) rather than above it, in which
        # case contact_count will correctly stay 0 forever regardless of
        # how far down you push it.
        self._debug_frame_counter = getattr(self, "_debug_frame_counter", 0) + 1
        if self._debug_frame_counter >= 30:
            self._debug_frame_counter = 0
            contacts, pen_force = self._cube.get_probe_debug_info()
            pad_top_z_dbg = SOFT_CENTER[2] + SOFT_HALF_Z
            if probe_world is not None:
                pwx_dbg, pwy_dbg, _ = probe_world
                in_footprint = (abs(pwx_dbg - SOFT_CENTER[0]) < SOFT_HALF_X
                                and abs(pwy_dbg - SOFT_CENTER[1]) < SOFT_HALF_Y)
                carb.log_warn(
                    f"[WarpSoftBody][force-debug] contact_count={contacts} "
                    f"pen_force_raw={pen_force:.4f}N  m_dx_h2_force_z={_fz:.4f}N  "
                    f"effective_fz_recorded={effective_fz:.4f}N  "
                    f"probe_xy=({pwx_dbg:.4f},{pwy_dbg:.4f}) "
                    f"pad_half=({SOFT_HALF_X:.4f},{SOFT_HALF_Y:.4f}) "
                    f"in_footprint={in_footprint}  "
                    f"probe_tip_z={probe_tip_z:.4f} pad_top_z={pad_top_z_dbg:.4f}")
                if contacts == 0 and not in_footprint:
                    carb.log_warn(
                        "[WarpSoftBody][force-debug] probe is OUTSIDE the pad's "
                        f"footprint (need |x|<{SOFT_HALF_X:.3f} and "
                        f"|y|<{SOFT_HALF_Y:.3f} around the pad center) -- drag "
                        "it back over the pad, this is not a code bug.")
            else:
                carb.log_warn(
                    f"[WarpSoftBody][force-debug] contact_count={contacts} "
                    f"pen_force_raw={pen_force:.4f}N  m_dx_h2_force_z={_fz:.4f}N")

        # ---- Cutting: cut along wherever the rod's tip actually travels
        # in XY while it's pressed into the pad (within its footprint and
        # actually PAST the top surface by CUT_ENGAGE_DEPTH -- not just
        # skin-margin contact, see the constant's docstring above). The
        # tip -- not the rod's center -- is the working end that
        # pokes/cuts. Unlike a single column index, this follows the
        # tip's real path in any direction (X, Y, or a diagonal drag) and
        # only severs the cells the tip actually swept through, one
        # probe-radius segment at a time. Depth-aware: how far DOWN the
        # tip has actually penetrated (not just whether it's touching)
        # decides how deep the cut goes -- a shallow graze only nicks the
        # skin layer, not the whole skin/fat/muscle stack. Runs AFTER
        # step() now (see ordering note above), so this frame's force
        # reading reflects the pre-cut contact, and the cut itself takes
        # effect starting next frame. ----
        if probe_world is not None:
            pwx, pwy, _ = probe_world
            pad_top_z = SOFT_CENTER[2] + SOFT_HALF_Z
            engaged = (
                abs(pwx - SOFT_CENTER[0]) < SOFT_HALF_X
                and abs(pwy - SOFT_CENTER[1]) < SOFT_HALF_Y
                and probe_tip_z < pad_top_z - CUT_ENGAGE_DEPTH
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

        # Write positions to USD render mesh. Face-vertex INDICES (and
        # per-face colors) are only re-uploaded on frames where the
        # triangle count actually changed (i.e. a cut just fired) -- not
        # every single frame. Cutting DOES change how many triangles
        # exist: severing a wall splits a grid vertex into two duplicate
        # particle columns, which turns previously-internal faces (shared
        # by two tets, so not boundary) into faces that are now only
        # claimed by one tet on each side -- i.e. new boundary faces
        # appear on both sides of the cut, so the triangle count goes UP.
        # If faceVertexCounts is left stale (its length no longer matches
        # how faceVertexIndices actually chunks into triangles), USD sees
        # an inconsistent mesh and Hydra culls/hides it outright -- this
        # was the "mesh deletes itself while cutting" bug. Recomputing
        # both attributes together whenever topology changes keeps them
        # consistent; only POINTS need a write every frame (particles are
        # always moving), which is why that Set() call sits outside the
        # `if` below.
        #
        # Safety net: if the solver ever blows up (NaN/Inf positions, from
        # any cause), do NOT push that into the renderer -- a degenerate
        # mesh like that is a plausible way to crash the RTX/Kit renderer
        # outright rather than just looking wrong. Skip the mesh write for
        # this frame and warn instead.
        cube_pos_np = self._cube.pos.numpy()
        _step_ms = (time.perf_counter() - _t0) * 1000.0
        self._update_adaptive_iters(_step_ms)
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
            self._cube_mesh.GetFaceVertexIndicesAttr().Set(tri_indices.tolist())
            if self._cube_color_pv is not None:
                self._cube_color_pv.Set(_vec3f_array(self._cube.tri_colors))
            self._last_tri_count = num_tris
        self._cube_mesh.GetPointsAttr().Set(_vec3f_array(cube_pos_np))
        return False

    def _update_adaptive_iters(self, step_ms: float):
        """Smooth the just-measured step() wall-clock cost into an EMA,
        then nudge self._adaptive_iters by at most 1 per frame within
        [SOLVER_ITERS_MIN, SOLVER_ITERS] to hold TARGET_FRAME_MS.

        One-at-a-time adjustment (not jumping straight to whatever the
        budget "allows") plus EMA smoothing on the input both matter
        here: a single spiky frame (e.g. the frame a cut fires and
        rebuilds every GPU array) shouldn't yank quality down and then
        immediately back up -- that oscillation would be more visible
        than just holding steady through the blip.
        """
        if self._step_ms_ema is None:
            self._step_ms_ema = step_ms
        else:
            self._step_ms_ema = (ADAPT_EMA_ALPHA * step_ms
                                  + (1.0 - ADAPT_EMA_ALPHA) * self._step_ms_ema)

        if self._step_ms_ema > TARGET_FRAME_MS * ADAPT_DOWN_MARGIN:
            self._adaptive_iters = max(SOLVER_ITERS_MIN, self._adaptive_iters - 1)
        elif self._step_ms_ema < TARGET_FRAME_MS * ADAPT_UP_MARGIN:
            self._adaptive_iters = min(SOLVER_ITERS, self._adaptive_iters + 1)

    # ------------------------------------------------------------------
    def _spawn(self):
        self._probe_last_good = None
        self._cut_last_xy = None
        self._collider_cache = []
        self._collider_rescan_counter = 0
        self._adaptive_iters = SOLVER_ITERS
        self._step_ms_ema = None
        self._cube = SoftBodyCube(
            center=SOFT_CENTER,
            half_x=SOFT_HALF_X,
            half_y=SOFT_HALF_Y,
            half_z=SOFT_HALF_Z,
            res_x=SOFT_RES_X,
            res_y=SOFT_RES_Y,
            res_z=SOFT_RES_Z,
            total_mass=0.5,
            device=self._device,
        )
        num_tris = len(self._cube.tri_indices) // 3
        fc = np.full(num_tris, 3, dtype=np.int32)
        self._cube_mesh.CreateFaceVertexCountsAttr(fc.tolist())
        self._cube_mesh.CreateFaceVertexIndicesAttr(
            self._cube.tri_indices.tolist())
        self._cube_mesh.CreatePointsAttr(
            _vec3f_array(self._cube.pos.numpy()))
        # Per-face (uniform) display color: skin/fat/muscle by depth (see
        # TISSUE_LAYERS), blended at layer transitions. CreatePrimvar is a
        # no-op if it already exists from a prior spawn (e.g. RESET), so
        # this is safe to call every time -- only the values change.
        if self._cube_color_pv is None:
            self._cube_color_pv = UsdGeom.PrimvarsAPI(
                self._cube_mesh.GetPrim()
            ).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.uniform)
        self._cube_color_pv.Set(_vec3f_array(self._cube.tri_colors))
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
