"""
Padel Kinematic Feature Extractor
Extracts scale-normalized, camera-invariant biomechanical and trajectory features
for shot classification and technique coaching.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.config import KP_IDX


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculates the 2D angle (in degrees) at joint p2 formed by p1-p2-p3.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 180.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PadelFeatureExtractor:
    def __init__(self):
        pass

    def extract_features_from_event(self, shot_event: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Extracts a comprehensive kinematic feature vector for a matched shot event.
        """
        poses = shot_event.get("shooter_poses", {})
        if not poses:
            return None

        impact_fid = shot_event.get("impact_frame")
        start_fid = shot_event.get("start_frame")
        end_fid = shot_event.get("end_frame")

        # Fallback impact pose
        impact_ann = poses.get(impact_fid)
        if impact_ann is None:
            # Pick middle frame available
            sorted_fids = sorted(poses.keys())
            if not sorted_fids:
                return None
            impact_ann = poses[sorted_fids[len(sorted_fids) // 2]]

        kp_impact = np.array(impact_ann["keypoints"]).reshape(-1, 3) # (17, 3)

        # 1. Torso Normalization scale
        # Shoulders midpoint vs Hips midpoint
        l_sh = kp_impact[KP_IDX["left_shoulder"], :2]
        r_sh = kp_impact[KP_IDX["right_shoulder"], :2]
        l_hip = kp_impact[KP_IDX["left_hip"], :2]
        r_hip = kp_impact[KP_IDX["right_hip"], :2]

        sh_mid = (l_sh + r_sh) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0

        torso_len = float(np.linalg.norm(sh_mid - hip_mid))
        if torso_len < 10.0:
            # Fallback to bbox height
            bbox = impact_ann.get("bbox", [0, 0, 100, 200])
            torso_len = max(bbox[3] * 0.4, 20.0)

        # 2. Determine dominant/hitting arm by proximity to ball or wrist velocity
        r_wrist_impact = kp_impact[KP_IDX["right_wrist"], :2]
        l_wrist_impact = kp_impact[KP_IDX["left_wrist"], :2]
        
        # In padel, majority of players are right-handed, but we evaluate both
        ball_traj = shot_event.get("ball_trajectory", {})
        ball_impact = ball_traj.get(impact_fid)
        
        if ball_impact:
            bx, by = ball_impact[0], ball_impact[1]
            dist_r = np.hypot(r_wrist_impact[0] - bx, r_wrist_impact[1] - by)
            dist_l = np.hypot(l_wrist_impact[0] - bx, l_wrist_impact[1] - by)
            is_right_arm = dist_r <= dist_l
        else:
            is_right_arm = True # Default right

        hit_w_idx = KP_IDX["right_wrist"] if is_right_arm else KP_IDX["left_wrist"]
        hit_elb_idx = KP_IDX["right_elbow"] if is_right_arm else KP_IDX["left_elbow"]
        hit_sh_idx = KP_IDX["right_shoulder"] if is_right_arm else KP_IDX["left_shoulder"]

        # 3. Trajectory analysis of hitting wrist across shot window
        wrist_xs, wrist_ys = [], []
        frames_sorted = sorted(poses.keys())

        for fid in frames_sorted:
            ann = poses[fid]
            kp = np.array(ann["keypoints"]).reshape(-1, 3)
            w = kp[hit_w_idx, :2]
            v = kp[hit_w_idx, 2]
            if v > 0:
                # Normalize relative to hip_mid and torso_len
                # Note: negative y is upwards!
                norm_x = (w[0] - hip_mid[0]) / torso_len
                norm_y = (w[1] - hip_mid[1]) / torso_len
                wrist_xs.append(norm_x)
                wrist_ys.append(norm_y)

        if not wrist_ys:
            return None

        wrist_xs = np.array(wrist_xs)
        wrist_ys = np.array(wrist_ys)

        # Wrist Kinematics
        wrist_apex_height = float(np.min(wrist_ys)) # Min y is highest point in image
        wrist_lowest_height = float(np.max(wrist_ys))
        wrist_vertical_range = float(wrist_lowest_height - wrist_apex_height)
        wrist_horizontal_range = float(np.max(wrist_xs) - np.min(wrist_xs))

        # Velocities
        if len(wrist_xs) > 1:
            vx = np.diff(wrist_xs)
            vy = np.diff(wrist_ys)
            speeds = np.sqrt(vx**2 + vy**2)
            max_wrist_speed = float(np.max(speeds))
            mean_wrist_speed = float(np.mean(speeds))
            exit_vy = float(vy[-1]) # direction at end of swing
        else:
            max_wrist_speed = 0.0
            mean_wrist_speed = 0.0
            exit_vy = 0.0

        # 4. Joint Angles at Impact
        hit_sh = kp_impact[hit_sh_idx, :2]
        hit_elb = kp_impact[hit_elb_idx, :2]
        hit_w = kp_impact[hit_w_idx, :2]

        elbow_angle = calculate_angle(hit_sh, hit_elb, hit_w)

        # Knee Flexion (right and left knee angles)
        r_knee_angle = calculate_angle(
            kp_impact[KP_IDX["right_hip"], :2],
            kp_impact[KP_IDX["right_knee"], :2],
            kp_impact[KP_IDX["right_ankle"], :2]
        )
        l_knee_angle = calculate_angle(
            kp_impact[KP_IDX["left_hip"], :2],
            kp_impact[KP_IDX["left_knee"], :2],
            kp_impact[KP_IDX["left_ankle"], :2]
        )
        avg_knee_flexion = float((r_knee_angle + l_knee_angle) / 2.0)

        # 5. Court Side & Perspective Orientation
        # In broadcast view: y > 0.45 is near court (back to camera), y <= 0.45 is far court (facing camera)
        player_court_y = float(hip_mid[1] / 1080.0)
        is_near_court = 1.0 if player_court_y > 0.45 else -1.0

        # Relative wrist impact x (offset from center of body, perspective normalized)
        raw_wrist_impact_x = (hit_w[0] - hip_mid[0]) / torso_len
        # Aligned wrist impact x: Positive = Forehand side, Negative = Backhand side
        aligned_wrist_impact_x = float(raw_wrist_impact_x * is_near_court)

        # Horizontal swing vector (start to impact to follow-through)
        if len(wrist_xs) >= 2:
            raw_dx = float(wrist_xs[-1] - wrist_xs[0])
            aligned_swing_dx = float(raw_dx * is_near_court)
        else:
            raw_dx = 0.0
            aligned_swing_dx = 0.0

        # Ball relative impact side (is ball on the dominant or non-dominant side of torso?)
        if ball_impact:
            raw_ball_rel_x = (ball_impact[0] - hip_mid[0]) / torso_len
            aligned_ball_rel_x = float(raw_ball_rel_x * is_near_court)
        else:
            aligned_ball_rel_x = aligned_wrist_impact_x

        # Relative height to head and shoulder
        nose_y = (kp_impact[KP_IDX["nose"], 1] - hip_mid[1]) / torso_len
        sh_y = (sh_mid[1] - hip_mid[1]) / torso_len
        
        # Overhead indicator: 1.0 if wrist is above nose, 0.0 otherwise
        is_overhead = 1.0 if wrist_apex_height < nose_y else 0.0

        # 6. Ball trajectory features
        ball_fids = sorted(ball_traj.keys())
        ball_speed_pre = 0.0
        ball_speed_post = 0.0
        ball_dy_change = 0.0

        if len(ball_fids) >= 4:
            b_coords = [ball_traj[fid][:2] for fid in ball_fids]
            pre_fids = [f for f in ball_fids if f <= impact_fid]
            post_fids = [f for f in ball_fids if f > impact_fid]
            
            if len(pre_fids) >= 2:
                p1, p2 = ball_traj[pre_fids[0]][:2], ball_traj[pre_fids[-1]][:2]
                ball_speed_pre = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / max(len(pre_fids), 1))
            
            if len(post_fids) >= 2:
                p1, p2 = ball_traj[post_fids[0]][:2], ball_traj[post_fids[-1]][:2]
                ball_speed_post = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / max(len(post_fids), 1))
                ball_dy_change = float(p2[1] - p1[1])

        # 7. Court Location (Y coordinate in frame, normalized 0 to 1)
        player_court_y = float(hip_mid[1] / 1080.0) # 1080p standard height

        wrist_impact_y = float((hit_w[1] - hip_mid[1]) / torso_len)

        features = {
            "wrist_apex_height": wrist_apex_height,
            "wrist_lowest_height": wrist_lowest_height,
            "wrist_vertical_range": wrist_vertical_range,
            "wrist_horizontal_range": wrist_horizontal_range,
            "max_wrist_speed": max_wrist_speed,
            "mean_wrist_speed": mean_wrist_speed,
            "exit_vy": exit_vy,
            "elbow_angle": elbow_angle,
            "avg_knee_flexion": avg_knee_flexion,
            "is_overhead": is_overhead,
            "aligned_wrist_impact_x": aligned_wrist_impact_x,
            "aligned_swing_dx": aligned_swing_dx,
            "aligned_ball_rel_x": aligned_ball_rel_x,
            "wrist_impact_y": wrist_impact_y,
            "wrist_nose_height_diff": float(wrist_apex_height - nose_y),
            "wrist_shoulder_height_diff": float(wrist_apex_height - sh_y),
            "duration_frames": float(shot_event.get("duration", 15)),
            "ball_speed_pre": ball_speed_pre,
            "ball_speed_post": ball_speed_post,
            "ball_speed_delta": float(ball_speed_post - ball_speed_pre),
            "ball_dy_change": ball_dy_change,
            "player_court_y": player_court_y,
        }

        return features

    def extract_dataset_features(self, matched_events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Processes all matched shot events into a clean tabular DataFrame ready for training.
        """
        rows = []
        for ev in matched_events:
            feats = self.extract_features_from_event(ev)
            if feats is not None:
                feats["category"] = ev["category"]
                feats["dataset"] = ev["dataset"]
                feats["start_frame"] = ev["start_frame"]
                feats["impact_frame"] = ev["impact_frame"]
                rows.append(feats)
        return pd.DataFrame(rows)
