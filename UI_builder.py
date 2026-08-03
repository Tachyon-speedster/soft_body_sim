import os
from datetime import datetime

import carb
import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .plot_export import save_force_plot
from .softbody_core import WarpSoftBodySim


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._force_plot_counter = 0   # throttles the live plot redraw
        self._export_dir = os.path.expanduser("~/warp_softbody_force_logs")
        self._on_init()

    # ------------------------------------------------------------------
    # Callbacks called by extension.py
    # ------------------------------------------------------------------

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        # Refresh the live Force(Z) plot every few physics frames rather
        # than every single one -- the reading itself is computed every
        # frame in softbody_core.py (nothing is lost), this just throttles
        # how often the widget redraws.
        self._force_plot_counter += 1
        if self._force_plot_counter < 3:
            return
        self._force_plot_counter = 0
        self._refresh_force_ui()

    def _refresh_force_ui(self):
        if not hasattr(self, "_scenario") or self._scenario is None:
            return
        reading = self._scenario.get_force_reading()
        if reading is None:
            return
        _t, f_now = reading
        t_hist, f_hist = self._scenario.get_force_history()
        if f_hist.size:
            self._force_plot.set_data(*f_hist.tolist())
        self._force_live_label.text = f"Force_Z: {f_now:.1f} N"
        peak = self._scenario.get_force_peak()
        if peak is not None:
            self._force_peak_label.text = f"Peak: {peak[0]:.1f} N @ {peak[1]:.2f}s"
        baseline = self._scenario.get_force_baseline()
        if baseline is not None:
            self._force_base_label.text = f"Baseline: {baseline:.2f} N"

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    
    def build_ui(self):
        # ---- World Controls ----
        world_frame = CollapsableFrame("World Controls", collapsed=False)
        with world_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button",
                    "LOAD",
                    setup_scene_fn=self._setup_scene,
                    setup_post_load_fn=self._setup_scenario,
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button",
                    "RESET",
                    pre_reset_fn=None,
                    post_reset_fn=self._on_post_reset_btn,
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        # ---- Run Scenario ----
        run_frame = CollapsableFrame("Run Scenario")
        with run_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        # ---- Force Feedback ----
        # Live Force(Z) vs Time trace recovered from the probe's contact
        # against the pad (see collide_capsule_sensed / force_sensor.py) --
        # meant to be visually comparable to the real suture-pad readings
        # (reading_1.txt / reading_2.txt): same axes (Force_Z in N vs
        # Time in s), calibrated against the real tissue-only resistance
        # (small ramp toward ~-3 to -4N) -- NOT the ~-40N dip in the raw
        # readings, which is the blade hitting the rigid table underneath
        # the pad once it's cut all the way through. See force_sensor.py's
        # find_tissue_only_region() for why that portion is excluded.
        force_frame = CollapsableFrame("Force Feedback", collapsed=False)
        with force_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._force_plot = ui.Plot(
                    ui.Type.LINE,
                    -8.0, 2.0,           # scale_min/scale_max -- tissue-only
                                          # range, not the ~-40N table dip
                    *[0.0] * 2,
                    height=160,
                    style={"color": 0xFF3355DD, "background_color": 0xFF1A1A1A},
                )
                with ui.HStack(spacing=8, height=0):
                    self._force_live_label = ui.Label("Force_Z: -- N", word_wrap=True)
                    self._force_peak_label = ui.Label("Peak: -- N", word_wrap=True)
                    self._force_base_label = ui.Label("Baseline: -- N", word_wrap=True)
                with ui.HStack(spacing=5, height=0):
                    ui.Button("Reset Trace", clicked_fn=self._on_reset_force_trace)
                    ui.Button("Export Trace (CSV + PNG)", clicked_fn=self._on_export_force_trace)
                self._force_export_label = ui.Label("", word_wrap=True)

        # ---- How to use ----
        info_frame = CollapsableFrame("How to use", collapsed=True)
        with info_frame:
            with ui.VStack(style=get_style(), spacing=4, height=0):
                ui.Label(
                    "1. Click LOAD -- builds the scene (green soft-body cube, "
                    "orange pipe, grey ground).",
                    word_wrap=True,
                )
                ui.Label(
                    "2. Click RUN -- cube falls and deforms against the pipe.",
                    word_wrap=True,
                )
                ui.Label(
                    "3. POKING with any rigid object:",
                    word_wrap=True,
                )
                ui.Label(
                    "   a. Create > Shapes > Sphere (or Cube, Capsule, Cone...).",
                    word_wrap=True,
                )
                ui.Label(
                    "   b. Right-click the new prim -> Add -> Physics -> Collider Preset.",
                    word_wrap=True,
                )
                ui.Label(
                    "   c. Move the prim into the cube while RUN is active -- it "
                    "will dent and deform the surface immediately.",
                    word_wrap=True,
                )
                ui.Label(
                    "   Supported shapes: Sphere, Cube/Box, Capsule, Cylinder, "
                    "Cone. Mesh prims fall back to a bounding-sphere collider.",
                    word_wrap=True,
                )
                ui.Label(
                    "4. Drag /World/WarpSoftBodyCube to throw the cube.",
                    word_wrap=True,
                )
                ui.Label(
                    "5. Click RESET -- re-spawns the cube above the pipe.",
                    word_wrap=True,
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _on_init(self):
        self._scenario = WarpSoftBodySim()

    def _add_light_to_stage(self):
        sphere_light = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphere_light.CreateRadiusAttr(2)
        sphere_light.CreateIntensityAttr(120_000)
        SingleXFormPrim(str(sphere_light.GetPath())).set_world_pose([0, 0, 7])

    # ---- LoadButton callbacks ----------------------------------------

    def _setup_scene(self):
        create_new_stage()
        self._add_light_to_stage()
        loaded_objects = self._scenario.load_example_assets()
        world = World.instance()
        for obj in loaded_objects:
            world.scene.add(obj)

    def _setup_scenario(self):
        self._scenario.setup()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    # ---- ResetButton callback ----------------------------------------

    def _on_post_reset_btn(self):
        self._scenario.reset()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    # ---- StateButton callbacks ---------------------------------------

    def _update_scenario(self, step: float):
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    # ---- Force Feedback panel buttons ---------------------------------

    def _on_reset_force_trace(self):
        self._scenario.reset_force_trace()
        self._force_plot.set_data(*[0.0, 0.0])
        self._force_live_label.text = "Force_Z: -- N"
        self._force_peak_label.text = "Peak: -- N"
        self._force_base_label.text = "Baseline: -- N"
        self._force_export_label.text = ""

    def _on_export_force_trace(self):
        os.makedirs(self._export_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self._export_dir, f"force_trace_{stamp}.csv")
        png_path = os.path.join(self._export_dir, f"force_trace_{stamp}.png")
        try:
            self._scenario.export_force_trace_csv(csv_path)
            t_hist, f_hist = self._scenario.get_force_history()
            save_force_plot(t_hist, f_hist, png_path,
                             title="Simulated Force (Z) vs. Time (suture pad)")
            self._force_export_label.text = f"Saved: {csv_path}"
        except Exception as e:
            carb.log_warn(f"[WarpSoftBody] force trace export failed: {e}")
            self._force_export_label.text = f"Export failed: {e}"

    # ---- Stage-open reset -------------------------------------------

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False
        if hasattr(self, "_force_plot"):
            self._on_reset_force_trace()
