"""
Padel AI Coach Configuration
Constants, keypoint mappings, court dimensions, and dataset file paths.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
LABELS_DIR = PROJECT_ROOT / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASETS = {
    "womens_final": {
        "video": DATA_DIR / "2022_BCN_FinalF_1.mp4",
        "sample_video": DATA_DIR / "2022_BCN_FinalF_1_sample.mp4",
        "pose_json": LABELS_DIR / "2022_BCN_FinalF_1_pose.json",
        "ball_json": LABELS_DIR / "2022_BCN_FinalF_1_ball.json",
        "shots_csv": LABELS_DIR / "2022_BCN_FinalF_1_shots.csv",
    },
    "mens_final": {
        "video": DATA_DIR / "2022_BCN_FinalM_1.mp4",
        "pose_json": LABELS_DIR / "2022_BCN_FinalM_1_pose.json",
        "ball_json": LABELS_DIR / "2022_BCN_FinalM_1_ball.json",
        "shots_csv": LABELS_DIR / "2022_BCN_FinalM_1_shots.csv",
    }
}

# COCO Keypoint index mapping (17 keypoints)
KEYPOINTS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

KP_IDX = {name: i for i, name in enumerate(KEYPOINTS)}

# Shot Classes in Padel
SHOT_CLASSES = [
    "Serve",
    "Forehand",
    "Backhand",
    "Smash",
    "Dropshot",
    "Other",
]

# Standard International Padel Court Dimensions (Meters)
COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_POSITION_Y_M = 10.0
SERVICE_LINE_DIST_M = 3.0 # 3 meters from back glass (6.95m from net)
BACK_WALL_HEIGHT_M = 4.0
SIDE_WALL_HEIGHT_M = 3.0

# Court Zones for Tactical Analysis
ZONES = {
    "kitchen_net": (0.0, 4.0),       # 0m to 4m from net (offensive dominance)
    "transition": (4.0, 6.95),       # 4m to 6.95m ("No-man's land" - vulnerable zone)
    "defensive_back": (6.95, 10.0),  # Baseline / Glass zone (defensive recovery)
}
