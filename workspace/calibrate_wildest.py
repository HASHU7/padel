import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from padelcoach.court import CourtCalibration
from padelcoach import config as C

# Court dimensions (m):
# COURT_W = 10.0, COURT_L = 20.0, NET_Y = 10.0
# FAR_SERVICE_Y = 3.05, NEAR_SERVICE_Y = 16.95, CENTER_X = 5.0

# In the video the_wildest_match.mp4 (1280 x 720):
# 1. near_service_center: (CENTER_X, NEAR_SERVICE_Y) = (5.0, 16.95)
# 2. near_service_left: (0.0, 16.95)
# 3. net_center: (CENTER_X, NET_Y) = (5.0, 10.0)
# 4. net_left: (0.0, 10.0)
# 5. far_service_center: (CENTER_X, FAR_SERVICE_Y) = (5.0, 3.05)
# 6. far_service_left: (0.0, 3.05)
# 7. far_service_right: (COURT_W, FAR_SERVICE_Y) = (10.0, 3.05)

# Measured from frame lines:
# Center line passes (75, 675) -> (580, 510) -> (840, 425)
# Near service line runs from (0, 655) through (75, 675)
# Net on floor runs from (210, 560) through (580, 510)
# Far service line runs through (520, 440) -> (840, 425) -> (1080, 410)

dst = np.float32([
    [5.0, 16.95],
    [5.0, 10.0],
    [5.0, 3.05],
    [0.0, 10.0],
    [0.0, 3.05],
    [10.0, 3.05],
])

src = np.float32([
    [75.0, 675.0],
    [580.0, 510.0],
    [840.0, 425.0],
    [210.0, 560.0],
    [520.0, 440.0],
    [1080.0, 410.0],
])

# Find homography
H, _ = cv2.findHomography(src, dst, 0)
proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
err = np.mean(np.linalg.norm(proj - dst, axis=1))
print("Initial reprojection error:", err, "m")

# Now project pure 3D court rectangle corners into image space using inverse H
H_inv = np.linalg.inv(H)
corners_court = np.float32([[0, 0], [10, 0], [10, 20], [0, 20]])
corners_img = cv2.perspectiveTransform(corners_court.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
print("Projected court corners (px):")
for name, c in zip(["far_left", "far_right", "near_right", "near_left"], corners_img):
    print(f"  {name}: [{c[0]:.1f}, {c[1]:.1f}]")

# Let's verify and save the calibration dictionary for the_wildest_match
pts_dict = {
    "near_service_center": (float(src[0, 0]), float(src[0, 1])),
    "net_center": (float(src[1, 0]), float(src[1, 1])),
    "far_service_center": (float(src[2, 0]), float(src[2, 1])),
    "net_left": (float(src[3, 0]), float(src[3, 1])),
    "far_service_left": (float(src[4, 0]), float(src[4, 1])),
    "far_service_right": (float(src[5, 0]), float(src[5, 1])),
}

calib = CourtCalibration(pts_dict, 1280, 720, scoreboard_roi=[10, 10, 240, 130])
print("CourtCalibration error:", calib.reprojection_error_m, "m")

calib_path = C.CALIB_DIR / "the_wildest_match.json"
calib.save(calib_path)
print("Saved calibration to:", calib_path)
