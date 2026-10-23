"""Global paths, court dimensions (FIP rules 2026) and tunable thresholds."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"
VIDEOS_DIR = WORKSPACE / "videos"
CALIB_DIR = WORKSPACE / "calibrations"
RESULTS_DIR = WORKSPACE / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LABELS_DIR = PROJECT_ROOT / "labels"

# Folders scanned for match videos (project root keeps the original PadelTracker100 files).
VIDEO_SEARCH_DIRS = [VIDEOS_DIR, PROJECT_ROOT]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

for _d in (WORKSPACE, VIDEOS_DIR, CALIB_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ONNX model files (trained on Colab, see colab/). All optional: the pipeline
# degrades to classical fallbacks when a model is missing.
POSE_MODEL = MODELS_DIR / "padel_pose.onnx"          # YOLO-pose, 17 COCO keypoints
BALL_MODEL = MODELS_DIR / "padel_ball.onnx"          # YOLO detector, 1 class (ball)
SHOT_MODEL = MODELS_DIR / "padel_shot_classifier.onnx"  # temporal pose model
SHOT_MODEL_META = MODELS_DIR / "padel_shot_classifier.json"

# --------------------------------------------------------------------------
# Court geometry in metres. Origin = far-left floor corner as seen from the
# main broadcast camera. X: 0 (left wall) -> 10 (right wall).
# Y: 0 (far back wall) -> 10 (net) -> 20 (near back wall).
# --------------------------------------------------------------------------
COURT_W = 10.0
COURT_L = 20.0
NET_Y = 10.0
SERVICE_LINE_FROM_NET = 6.95
FAR_SERVICE_Y = NET_Y - SERVICE_LINE_FROM_NET    # 3.05
NEAR_SERVICE_Y = NET_Y + SERVICE_LINE_FROM_NET   # 16.95
CENTER_X = COURT_W / 2.0
NET_HEIGHT_CENTER = 0.88
NET_HEIGHT_POSTS = 0.92
# Side walls: glass for the first 4 m from each back wall, metal mesh in between.
SIDE_GLASS_LEN = 4.0

# Tactical zones by distance from the net (m).
# Boundaries sit in the valleys of the measured position distribution of WPT
# finalists (net peak 3.5-4 m, defence peak 7-8.5 m from the net).
ZONE_NET_MAX = 4.6          # attacking "net" zone
ZONE_TRANSITION_MAX = 6.4   # no-man's land between net and defence positions

# Standard court landmarks (metres) usable for calibration clicks.
COURT_LANDMARKS = {
    "far_left_corner": (0.0, 0.0),
    "far_right_corner": (COURT_W, 0.0),
    "far_service_left": (0.0, FAR_SERVICE_Y),
    "far_service_center": (CENTER_X, FAR_SERVICE_Y),
    "far_service_right": (COURT_W, FAR_SERVICE_Y),
    "net_left": (0.0, NET_Y),
    "net_center": (CENTER_X, NET_Y),
    "net_right": (COURT_W, NET_Y),
    "near_service_left": (0.0, NEAR_SERVICE_Y),
    "near_service_center": (CENTER_X, NEAR_SERVICE_Y),
    "near_service_right": (COURT_W, NEAR_SERVICE_Y),
    "near_left_corner": (0.0, COURT_L),
    "near_right_corner": (COURT_W, COURT_L),
}

# COCO-17 keypoints
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {n: i for i, n in enumerate(KEYPOINTS)}
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

# Base stroke labels (PadelTracker100 taxonomy) and refined shot types.
BASE_STROKES = ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"]
SHOT_TYPES = [
    "serve", "return", "forehand_drive", "backhand_drive", "forehand_volley",
    "backhand_volley", "forehand_wall", "backhand_wall", "lob", "smash",
    "bandeja", "vibora", "dropshot", "chiquita", "other",
]

# --------------------------------------------------------------------------
# Event detection thresholds (frames assume 30 fps; scaled by fps at runtime)
# --------------------------------------------------------------------------
MIN_FRAMES_BETWEEN_HITS = 9        # ~0.3 s
RALLY_GAP_SECONDS = 3.5            # no hit for this long -> rally over
MAX_RALLY_SHOT_GAP_SECONDS = 3.2
BALL_MAX_GAP_INTERP = 6            # frames
FOUL_MIN_CONFIDENCE_REPORT = 0.55  # below this a rule check is "review only"
