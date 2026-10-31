"""Mark the asked item on the LIBERO camera images with its box.

The simulator knows every object's geometry, so the box is exact: the
corners of the bounding box of each of the item's geoms are projected into
the camera, and the mark is a rectangle around them. This is the detection
module with a perfect detector; a learned detector can replace item_box.

Boxes are drawn on the raw renders (obs["agentview_image"] and
obs["robot0_eye_in_hand_image"]). The recorder and LeRobot's evaluation both
turn those images 180 degrees later, so the mark turns with the image.
"""

import mujoco
import numpy as np
from robosuite.utils import camera_utils

# Observation key -> MuJoCo camera name.
CAMERAS = {"agentview_image": "agentview", "robot0_eye_in_hand_image": "robot0_eye_in_hand"}
COLOR = (0, 255, 0)
WIDTH = 2
# How a robosuite projection (row 0 at the top of the upright view) maps
# onto the raw renders; set from datagen/check_marks.py.
FLIP_ROWS = True
FLIP_COLS = False
_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


def item_corners(sim, object_name: str) -> np.ndarray:
    """World coordinates of the bounding-box corners of every geom of an object."""
    # robosuite wraps mujoco's MjModel and MjData; the raw objects carry every field.
    m = getattr(sim.model, "_model", sim.model)
    d = getattr(sim.data, "_data", sim.data)
    bodies = {
        b for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(object_name)
    }
    corners = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies:
            center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            local = center + _SIGNS * half
            corners.append(d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T)
    if not corners:
        raise ValueError(f"No geoms for {object_name}")
    return np.concatenate(corners)


def item_box(sim, object_name: str, camera: str, height: int, width: int, flip_rows=None, flip_cols=None):
    """(top, left, bottom, right) of the object on the raw render, or None if out of view."""
    points = item_corners(sim, object_name)
    transform = camera_utils.get_camera_transform_matrix(sim, camera, height, width)
    p = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ transform.T
    p = p[p[:, 2] > 1e-6]  # corners in front of the camera
    if len(p) == 0:
        return None
    cols, rows = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    left, right, top, bottom = cols.min(), cols.max(), rows.min(), rows.max()
    if right < 0 or left > width - 1 or bottom < 0 or top > height - 1:
        return None
    left, right = max(left, 0), min(right, width - 1)
    top, bottom = max(top, 0), min(bottom, height - 1)
    if FLIP_ROWS if flip_rows is None else flip_rows:
        top, bottom = height - 1 - bottom, height - 1 - top
    if FLIP_COLS if flip_cols is None else flip_cols:
        left, right = width - 1 - right, width - 1 - left
    return int(round(top)), int(round(left)), int(round(bottom)), int(round(right))


def draw_box(image: np.ndarray, box, color=COLOR, width=WIDTH) -> np.ndarray:
    """A copy of image with a rectangle outline; image unchanged if box is None."""
    image = image.copy()
    if box is None:
        return image
    top, left, bottom, right = box
    image[top:top + width, left:right + 1] = color
    image[max(bottom - width + 1, 0):bottom + 1, left:right + 1] = color
    image[top:bottom + 1, left:left + width] = color
    image[top:bottom + 1, max(right - width + 1, 0):right + 1] = color
    return image


def mark(sim, observation: dict, object_name: str, keys=CAMERAS) -> dict:
    """observation with the object marked on each camera image (keys: obs key -> MuJoCo camera)."""
    marked = dict(observation)
    for key, camera in keys.items():
        image = observation[key]
        marked[key] = draw_box(image, item_box(sim, object_name, camera, image.shape[0], image.shape[1]))
    return marked
