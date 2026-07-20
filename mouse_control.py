"""
mouse_control.py
------------------
Run this ALONGSIDE isaac_cut_viewer.py (in a separate terminal). This
window is your control surface: click-and-drag across it to control
where the cut has progressed to. The actual physics and 3D rendering
happen in Isaac Sim, in the other process -- this window just captures
your mouse and writes the current target to a shared file every frame.

Why split into two processes: Isaac Sim's own viewport doesn't have a
reliable, well-documented way to convert a mouse click to a world
position (confirmed via multiple NVIDIA forum threads of people hitting
inconsistent/broken behavior on this exact thing). Matplotlib's mouse
events are simple and already proven working on your machine. Splitting
the "control input" from the "3D rendering" sidesteps the fragile part
entirely.

Run with:
    python3 mouse_control.py
"""

import json
import time
import matplotlib.pyplot as plt

SHARED_STATE_FILE = "/tmp/warp_cut_knife_state.json"

NX = 9  # must match isaac_cut_viewer.py's mesh resolution
SIZE_X = 0.09  # must match isaac_cut_viewer.py's sheet width (meters)


def main():
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.set_xlim(0, SIZE_X)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Click + hold, drag left->right to advance the cut\n"
                  "(actual sheet renders in the Isaac Sim window)")

    state = {"dragging": False, "x": None}

    def on_press(event):
        if event.inaxes == ax:
            state["dragging"] = True

    def on_release(event):
        state["dragging"] = False

    def on_move(event):
        if event.inaxes == ax:
            state["x"] = event.xdata

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.ion()
    plt.show()

    knife_col = -1
    dx = SIZE_X / (NX - 1)

    print(f"Writing knife target to {SHARED_STATE_FILE}")
    print("Make sure isaac_cut_viewer.py is running in another terminal.")

    while plt.fignum_exists(fig.number):
        if state["dragging"] and state["x"] is not None:
            target = int(round(state["x"] / dx))
            target = max(0, min(NX - 1, target))
            knife_col = max(knife_col, target)

        with open(SHARED_STATE_FILE, "w") as f:
            json.dump({"knife_col": knife_col, "dragging": state["dragging"]}, f)

        plt.pause(0.02)


if __name__ == "__main__":
    main()
