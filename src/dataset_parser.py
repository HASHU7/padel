"""
Padel Dataset Parser
Parses COCO video annotations for ball, player pose (17 keypoints), and shot ground truth.
Includes disk-caching to accelerate future loads.
"""
import json
import pickle
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.config import KP_IDX, DATASETS, LABELS_DIR


class PadelDatasetParser:
    def __init__(self, dataset_key: str = "womens_final"):
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset_key: {dataset_key}. Choose from {list(DATASETS.keys())}")
        self.dataset_key = dataset_key
        self.paths = DATASETS[dataset_key]
        self.cache_dir = LABELS_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_shots(self) -> pd.DataFrame:
        """Loads and formats the shot ground truth CSV."""
        df = pd.read_csv(self.paths["shots_csv"], sep=";", encoding="utf-8")
        df["frame_idx"] = df["file_name"].apply(
            lambda x: int(x.split("_")[1].split(".")[0])
        )
        return df

    def extract_shot_intervals(self) -> List[Dict[str, Any]]:
        """
        Groups continuous shot frames into distinct shot action events.
        Returns a list of event dictionaries with start_frame, end_frame, category, and duration.
        """
        df_shots = self.load_shots()
        shot_changes = (df_shots["category"] != df_shots["category"].shift()).cumsum()
        
        events = []
        for (cat, _), grp in df_shots.groupby(["category", shot_changes]):
            if cat != "0": # '0' represents no-shot rally frames
                start_f = int(grp["frame_idx"].min())
                end_f = int(grp["frame_idx"].max())
                events.append({
                    "dataset": self.dataset_key,
                    "category": str(cat),
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "duration": len(grp),
                    "impact_frame": (start_f + end_f) // 2 # initial estimate, refined by ball proximity
                })
        
        # Ensure chronological match sequence
        events.sort(key=lambda x: x["start_frame"])

        # Merge split annotation intervals (e.g. 1-6 frame drops to 0 during same shot action)
        merged = []
        for ev in events:
            if merged and merged[-1]["category"] == ev["category"] and (ev["start_frame"] - merged[-1]["end_frame"] <= 6):
                merged[-1]["end_frame"] = ev["end_frame"]
                merged[-1]["duration"] = merged[-1]["end_frame"] - merged[-1]["start_frame"] + 1
                merged[-1]["impact_frame"] = (merged[-1]["start_frame"] + merged[-1]["end_frame"]) // 2
            else:
                merged.append(ev)
        return merged

    def load_ball_positions(self) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Returns a dictionary mapping frame_idx -> (center_x, center_y, width, height).
        Cached to disk for instant subsequent loading.
        """
        cache_path = self.cache_dir / f"{self.dataset_key}_ball_cache.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"[{self.dataset_key}] Parsing ball JSON: {self.paths['ball_json'].name}...")
        with open(self.paths["ball_json"], "r", encoding="utf-8") as f:
            ball_data = json.load(f)

        ball_dict = {}
        for ann in ball_data.get("annotations", []):
            fid = ann["image_id"] - 1 # 0-indexed frame id
            bbox = ann.get("bbox", [])
            if bbox and len(bbox) == 4:
                cx = bbox[0] + bbox[2] / 2.0
                cy = bbox[1] + bbox[3] / 2.0
                ball_dict[fid] = (cx, cy, bbox[2], bbox[3])

        with open(cache_path, "wb") as f:
            pickle.dump(ball_dict, f)
        print(f"[{self.dataset_key}] Cached ball data for {len(ball_dict)} frames.")
        return ball_dict

    def load_player_poses(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Returns a dictionary mapping frame_idx -> list of player annotations (with keypoints and bbox).
        Cached to disk for fast loading.
        """
        cache_path = self.cache_dir / f"{self.dataset_key}_pose_cache.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"[{self.dataset_key}] Parsing pose JSON: {self.paths['pose_json'].name} (~110MB)...")
        with open(self.paths["pose_json"], "r", encoding="utf-8") as f:
            pose_data = json.load(f)

        poses_by_frame = {}
        for ann in pose_data.get("annotations", []):
            fid = ann["image_id"] - 1
            if fid not in poses_by_frame:
                poses_by_frame[fid] = []
            
            # Store compact representation
            poses_by_frame[fid].append({
                "id": ann.get("id"),
                "bbox": ann.get("bbox", []),
                "keypoints": ann.get("keypoints", []), # 51 values: x, y, v
                "num_keypoints": ann.get("num_keypoints", 0)
            })

        with open(cache_path, "wb") as f:
            pickle.dump(poses_by_frame, f)
        print(f"[{self.dataset_key}] Cached pose data for {len(poses_by_frame)} frames.")
        return poses_by_frame

    def match_shooter_and_impact(
        self,
        event: Dict[str, Any],
        ball_dict: Dict[int, Tuple[float, float, float, float]],
        poses_by_frame: Dict[int, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Identifies which player performed the shot and the exact impact frame
        by minimizing the distance between the ball and player's wrists.
        """
        start_f = event["start_frame"]
        end_f = event["end_frame"]
        
        best_player_idx = None
        min_overall_dist = float("inf")
        impact_frame = event["impact_frame"]

        # Track distances per player across frames
        player_min_dists = {}

        for fid in range(start_f, end_f + 1):
            b_info = ball_dict.get(fid)
            players = poses_by_frame.get(fid, [])
            if not b_info or not players:
                continue

            bx, by, _, _ = b_info

            for p_idx, player in enumerate(players):
                kp = player.get("keypoints", [])
                if len(kp) < 51:
                    continue

                # Right wrist (idx 10) & Left wrist (idx 9)
                rw_x, rw_y, rw_v = kp[KP_IDX["right_wrist"] * 3 : KP_IDX["right_wrist"] * 3 + 3]
                lw_x, lw_y, lw_v = kp[KP_IDX["left_wrist"] * 3 : KP_IDX["left_wrist"] * 3 + 3]

                dists = []
                if rw_v > 0:
                    dists.append(np.hypot(rw_x - bx, rw_y - by))
                if lw_v > 0:
                    dists.append(np.hypot(lw_x - bx, lw_y - by))

                if dists:
                    d = min(dists)
                    if p_idx not in player_min_dists or d < player_min_dists[p_idx]:
                        player_min_dists[p_idx] = d

                    if d < min_overall_dist:
                        min_overall_dist = d
                        best_player_idx = p_idx
                        impact_frame = fid

        # If ball wasn't close enough, fall back to center of bounding box closest to ball
        if best_player_idx is None and player_min_dists:
            best_player_idx = min(player_min_dists, key=player_min_dists.get)

        if best_player_idx is None:
            return None

        # Collect player pose sequence across shot window by tracking nearest player to shooter position
        shooter_poses = {}
        impact_players = poses_by_frame.get(impact_frame, [])
        ref_cx, ref_cy = 960.0, 540.0
        if best_player_idx < len(impact_players):
            ib = impact_players[best_player_idx].get("bbox", [])
            if len(ib) == 4:
                ref_cx, ref_cy = ib[0] + ib[2] / 2.0, ib[1] + ib[3] / 2.0

        for fid in range(start_f, end_f + 1):
            players = poses_by_frame.get(fid, [])
            if not players:
                continue
            best_p = None
            min_d = float("inf")
            for p in players:
                b = p.get("bbox", [])
                if len(b) == 4:
                    cx = b[0] + b[2] / 2.0
                    cy = b[1] + b[3] / 2.0
                    d = np.hypot(cx - ref_cx, cy - ref_cy)
                    if d < min_d:
                        min_d = d
                        best_p = p
            if best_p and min_d < 350.0:
                shooter_poses[fid] = best_p

        # Identify true player court quadrant (P0: Near Left, P3: Near Right, P1: Far Left, P2: Far Right)
        shooter_player_role = 0
        if shooter_poses and impact_frame in shooter_poses:
            p_ann = shooter_poses[impact_frame]
            kp = p_ann.get("keypoints", [])
            bbox = p_ann.get("bbox", [])
            if len(kp) >= 51 and kp[11 * 3 + 1] > 0 and kp[12 * 3 + 1] > 0:
                cx = (kp[11 * 3] + kp[12 * 3]) / 2.0
                cy = (kp[11 * 3 + 1] + kp[12 * 3 + 1]) / 2.0
            elif bbox and len(bbox) == 4:
                cx = bbox[0] + bbox[2] / 2.0
                cy = bbox[1] + bbox[3] / 2.0
            else:
                cx, cy = ref_cx, ref_cy

            if cy > 461.0:
                shooter_player_role = 0 if cx < 960.0 else 3
            else:
                shooter_player_role = 1 if cx < 960.0 else 2

        return {
            "category": event["category"],
            "dataset": self.dataset_key,
            "start_frame": start_f,
            "end_frame": end_f,
            "impact_frame": impact_frame,
            "duration": event["duration"],
            "detection_list_idx": best_player_idx,
            "shooter_player_idx": shooter_player_role,
            "min_wrist_ball_dist": min_overall_dist,
            "shooter_poses": shooter_poses,
            "ball_trajectory": {fid: ball_dict[fid] for fid in range(start_f, end_f + 1) if fid in ball_dict}
        }


if __name__ == "__main__":
    parser = PadelDatasetParser("womens_final")
    events = parser.extract_shot_intervals()
    print(f"Extracted {len(events)} shot intervals.")
    ball_dict = parser.load_ball_positions()
    poses_by_frame = parser.load_player_poses()
    sample_match = parser.match_shooter_and_impact(events[0], ball_dict, poses_by_frame)
    print("Sample matched shot:", sample_match["category"], "at frame", sample_match["impact_frame"])
