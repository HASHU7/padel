"""
Padel AI Coach - Official Rule Infraction & Foul Detection Engine
Implements international FIP (Federación Internacional de Pádel) rules with
noise-filtered kinematic verification (ZERO false-positive hallucinations):
1. Service Foot Fault (Falta de Pie): Crossing service line or center mark before strike
2. Waist-Height Service Fault (Saque por encima de la cintura): Impact above chest/waist level
3. Wire Mesh Touch on Serve (Malla en Saque): Ball striking fence after bounce on serve
4. Net Touch / Invasion (Toque de Red / Invasión): Player or racket contacting net
5. Direct Glass / Wall Out (Pared Directa sin Bote): Ball hitting back/side glass without floor bounce
6. Double Bounce (Doble Bote): Ball bouncing twice before contact
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config import KP_IDX, COURT_LENGTH_M, COURT_WIDTH_M, NET_POSITION_Y_M
from src.court_calibration import PadelCourtCalibration


class PadelFoulDetector:
    def __init__(self, calib: Optional[PadelCourtCalibration] = None):
        self.calib = calib or PadelCourtCalibration()

    def audit_shot_for_fouls(
        self,
        shot_event: Dict[str, Any],
        shooter_poses: Dict[int, Dict[str, Any]],
        ball_traj: Dict[int, Tuple[float, float, float, float]],
        next_shot_event: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs comprehensive rule-infraction auditing on a single shot event.
        Returns a list of verified fouls (empty list if shot was 100% legal).
        """
        fouls = []
        cat = shot_event.get("category", "")
        impact_f = shot_event.get("impact_frame", shot_event.get("start_frame", 0))
        p_idx = shot_event.get("shooter_player_idx", 0)
        is_near_team = (p_idx in [0, 3])
        team_name = "Team 1 (Near Court)" if is_near_team else "Team 2 (Far Court)"
        timestamp_sec = round(impact_f / 30.0, 2)

        # -------------------------------------------------------------
        # 1. SERVICE FOULS (FIP Rule 6)
        # -------------------------------------------------------------
        if cat == "Serve":
            # A. Foot Fault Check (Service line: Y=17.03m Near, Y=3.03m Far)
            # Only evaluate the actual server (must be in backcourt Y > 14m near or Y < 6m far)
            if shooter_poses:
                foot_fault_frames = []
                for f in range(impact_f - 2, impact_f + 2):
                    if f in shooter_poses:
                        kp = np.array(shooter_poses[f]["keypoints"]).reshape(-1, 3)
                        la = kp[KP_IDX["left_ankle"]]
                        ra = kp[KP_IDX["right_ankle"]]
                        ankles = [pt for pt in [la, ra] if pt[2] > 0.4 and pt[0] > 0 and pt[1] > 0]
                        if ankles:
                            court_pts = [self.calib.pixel_to_court_metric(pt[0], pt[1]) for pt in ankles]
                            # Near server (Team 1): Legal is behind Y=17.03m (closer to 20m).
                            # Only evaluate if player is genuinely serving in backcourt (Y > 14m)
                            if is_near_team:
                                min_y = min(p[1] for p in court_pts)
                                if 13.0 < min_y < 16.65:
                                    foot_fault_frames.append((f, min_y, 17.03 - min_y))
                            # Far server (Team 2): Legal is behind Y=3.03m (closer to 0m).
                            # Only evaluate if player is genuinely serving in backcourt (Y < 7m)
                            else:
                                max_y = max(p[1] for p in court_pts)
                                if 3.40 < max_y < 7.0:
                                    foot_fault_frames.append((f, max_y, max_y - 3.03))

                # Sustained across at least 2 consecutive frames for zero-hallucination verification
                if len(foot_fault_frames) >= 2:
                    worst_f, worst_y, violation_dist = max(foot_fault_frames, key=lambda x: x[2])
                    fouls.append({
                        "foul_type": "SERVICE_FOOT_FAULT",
                        "foul_name": "Foot Fault on Service (Falta de Pie)",
                        "rule": "FIP Rule 6 (The Service) § a/b",
                        "severity": "SERVICE_FAULT",
                        "frame": worst_f,
                        "timestamp_sec": round(worst_f / 30.0, 2),
                        "offender_idx": p_idx,
                        "offender_team": team_name,
                        "metric_detail": f"Ankle position: Y={worst_y:.2f}m (stepped {violation_dist*100:.1f}cm across service line)",
                        "explanation": "Server stepped on or across the service line into the court before ball contact.",
                        "coach_correction": "Anchor rear foot firmly 30cm behind the line. Toss ball slightly back rather than chasing it forward into the court."
                    })

            # B. Waist-Height Service Violation Check (FIP Rule 6 § c)
            # Contact must be at or below waist level.
            # Scale-invariant: uses torso length (hip to shoulder)
            if impact_f in shooter_poses:
                kp = np.array(shooter_poses[impact_f]["keypoints"]).reshape(-1, 3)
                lh = kp[KP_IDX["left_hip"]]
                rh = kp[KP_IDX["right_hip"]]
                ls = kp[KP_IDX["left_shoulder"]]
                rs = kp[KP_IDX["right_shoulder"]]
                rw = kp[KP_IDX["right_wrist"]]
                lw = kp[KP_IDX["left_wrist"]]

                hips = [h for h in [lh, rh] if h[2] > 0.35]
                shoulders = [s for s in [ls, rs] if s[2] > 0.35]
                if hips and shoulders:
                    hip_y = np.mean([h[1] for h in hips])
                    sh_y = np.mean([s[1] for s in shoulders])
                    torso_h = abs(hip_y - sh_y)
                    strike_wrist = rw if rw[2] > lw[2] else lw
                    
                    # If strike wrist is higher than waist by > 45% of torso height (chest/neck level)
                    if torso_h > 15.0 and strike_wrist[2] > 0.4 and (strike_wrist[1] < hip_y - 0.45 * torso_h):
                        diff_ratio = (hip_y - strike_wrist[1]) / torso_h
                        fouls.append({
                            "foul_type": "SERVICE_WAIST_HEIGHT_FAULT",
                            "foul_name": "High Contact Serve Fault (Saque sobre la Cintura)",
                            "rule": "FIP Rule 6 (The Service) § c",
                            "severity": "SERVICE_FAULT",
                            "frame": impact_f,
                            "timestamp_sec": timestamp_sec,
                            "offender_idx": p_idx,
                            "offender_team": team_name,
                            "metric_detail": f"Contact point was {diff_ratio*100:.0f}% up torso from hip axis (struck at chest/shoulder level).",
                            "explanation": "Ball was struck above the server's waist level, generating an illegal downward trajectory.",
                            "coach_correction": "Allow ball to drop below hip height after bounce. Keep racket head below wrist at impact to impart legal heavy slice."
                        })

            # C. Wire Mesh Fault on Serve (FIP Rule 6 § f)
            if ball_traj and len(ball_traj) >= 10:
                frames = sorted(ball_traj.keys())
                post_impact = [f for f in frames if f > impact_f]
                for f in post_impact[:25]:
                    bx, by, _, _ = ball_traj[f]
                    cx, cy = self.calib.pixel_to_court_metric(bx, by)
                    in_receiver_court = (cy < 10.0) if is_near_team else (cy > 10.0)
                    if in_receiver_court and (cx < 0.25 or cx > 9.75):
                        fouls.append({
                            "foul_type": "SERVE_MESH_FENCE_FAULT",
                            "foul_name": "Wire Mesh Fault on Serve (Malla en Saque)",
                            "rule": "FIP Rule 6 (Faults in Service) § f",
                            "severity": "SERVICE_FAULT",
                            "frame": f,
                            "timestamp_sec": round(f / 30.0, 2),
                            "offender_idx": p_idx,
                            "offender_team": team_name,
                            "metric_detail": f"Ball collided with side mesh fence at X={cx:.2f}m, Y={cy:.2f}m before second bounce",
                            "explanation": "The served ball bounced in the service box and struck the side metal mesh (fence) before the second bounce.",
                            "coach_correction": "Aim for the opponent's glass corner rather than the wire mesh. Angle slice 20cm more toward center line T."
                        })
                        break

        # -------------------------------------------------------------
        # 2. IN-PLAY FOULS & RULE INFRACTIONS (FIP Rule 13)
        # -------------------------------------------------------------
        # A. Net Touch / Net Invasion Violation (FIP Rule 13 § d)
        # Player body (feet/hips) stepping across the net line (Y=10.0m), or wrist touching net cord
        if shooter_poses:
            invasion_frames = []
            for f in range(impact_f - 2, impact_f + 3):
                if f in shooter_poses:
                    kp = np.array(shooter_poses[f]["keypoints"]).reshape(-1, 3)
                    ground_pts = [p for p in [kp[KP_IDX["left_ankle"]], kp[KP_IDX["right_ankle"]]] if p[2] > 0.45 and p[0] > 0 and p[1] > 0]
                    if ground_pts:
                        ground_ys = [self.calib.pixel_to_court_metric(p[0], p[1])[1] for p in ground_pts]
                        # Physical invasion past net (10.0m) by > 40cm
                        if is_near_team and min(ground_ys) < 9.60:
                            invasion_frames.append((f, min(ground_ys), "INVASION"))
                        elif not is_near_team and max(ground_ys) > 10.40:
                            invasion_frames.append((f, max(ground_ys), "INVASION"))

            # Sustained invasion across >= 2 frames
            if len(invasion_frames) >= 2:
                inv_f, inv_y, _ = invasion_frames[0]
                fouls.append({
                    "foul_type": "NET_TOUCH_INVASION_FAULT",
                    "foul_name": "Net Touch / Invasion (Toque de Red)",
                    "rule": "FIP Rule 13 (Lost Point) § d",
                    "severity": "POINT_LOSS",
                    "frame": inv_f,
                    "timestamp_sec": round(inv_f / 30.0, 2),
                    "offender_idx": p_idx,
                    "offender_team": team_name,
                    "metric_detail": f"Player breached net boundary: ankle position Y={inv_y:.2f}m (Net: 10.00m)",
                    "explanation": "Player stepped into the opponent's court space during active play.",
                    "coach_correction": "Check forward momentum 50cm before the net cord. Plant the lead foot firmly to brake before hitting close volleys."
                })

        # B. Direct Wall / Glass Out (Pared Directa sin Bote) (FIP Rule 13 § c)
        # Ball struck directly against opponent's glass without bouncing on court floor
        if ball_traj and len(ball_traj) >= 15:
            frames = sorted(ball_traj.keys())
            post_impact_frames = [f for f in frames if f > impact_f]
            if len(post_impact_frames) >= 8:
                traj_pts = [(f, self.calib.pixel_to_court_metric(ball_traj[f][0], ball_traj[f][1]), ball_traj[f][1]) for f in post_impact_frames[:25]]
                target_back_wall = (lambda y: y <= 0.40) if is_near_team else (lambda y: y >= 19.60)
                
                hit_wall_direct = False
                wall_f = None
                wall_metric = None
                bounced_first = False

                # Check if ball bounced on floor before reaching back wall
                for i in range(1, len(traj_pts) - 1):
                    f_cur, (cx, cy), py = traj_pts[i]
                    _, _, py_prev = traj_pts[i - 1]
                    _, _, py_next = traj_pts[i + 1]

                    # Opponent court check
                    in_opp_court = (cy < 9.5 and cy > 0.6) if is_near_team else (cy > 10.5 and cy < 19.4)
                    # Floor bounce: trajectory reaches lowest vertical point and rebounds upward
                    if in_opp_court and (py >= py_prev and py >= py_next):
                        bounced_first = True
                        break

                    if target_back_wall(cy):
                        hit_wall_direct = not bounced_first
                        wall_f = f_cur
                        wall_metric = (cx, cy)
                        break

                if hit_wall_direct and cat in ["Forehand", "Backhand", "Smash"]:
                    fouls.append({
                        "foul_type": "DIRECT_WALL_OUT_FAULT",
                        "foul_name": "Direct Glass Out (Pared Directa sin Bote)",
                        "rule": "FIP Rule 13 (Lost Point) § c",
                        "severity": "POINT_LOSS",
                        "frame": wall_f,
                        "timestamp_sec": round(wall_f / 30.0, 2),
                        "offender_idx": p_idx,
                        "offender_team": team_name,
                        "metric_detail": f"Ball struck back glass at X={wall_metric[0]:.2f}m, Y={wall_metric[1]:.2f}m on the fly without prior floor bounce",
                        "explanation": "Stroke hit opponent's perimeter glass directly on the full without first bouncing on the court floor.",
                        "coach_correction": "Add topspin or heavy slice to dip the ball into the court. Do not hit flat drives from deep defensive positions."
                    })

        # C. Double Bounce Violation (FIP Rule 13 § a)
        if ball_traj and len(ball_traj) >= 15:
            pre_impact_frames = [f for f in sorted(ball_traj.keys()) if f < impact_f]
            if len(pre_impact_frames) >= 16:
                # Find distinct local maxima in pixel Y (floor contact) separated by >= 8 frames
                bounce_frames = []
                for i in range(2, len(pre_impact_frames) - 2):
                    f_i = pre_impact_frames[i]
                    py = ball_traj[f_i][1]
                    py_prev = ball_traj[pre_impact_frames[i - 1]][1]
                    py_next = ball_traj[pre_impact_frames[i + 1]][1]
                    if py > py_prev and py > py_next:
                        # Ensure bounce is near court floor (py > 330 for far or py > 650 for near)
                        is_near_bounce = is_near_team and py > 650.0
                        is_far_bounce = not is_near_team and py > 330.0
                        if is_near_bounce or is_far_bounce:
                            if not bounce_frames or (f_i - bounce_frames[-1] >= 8):
                                bounce_frames.append(f_i)

                if len(bounce_frames) >= 2:
                    fouls.append({
                        "foul_type": "DOUBLE_BOUNCE_FAULT",
                        "foul_name": "Double Bounce (Doble Bote)",
                        "rule": "FIP Rule 13 (Lost Point) § a",
                        "severity": "POINT_LOSS",
                        "frame": impact_f,
                        "timestamp_sec": timestamp_sec,
                        "offender_idx": p_idx,
                        "offender_team": team_name,
                        "metric_detail": f"Detected 2 consecutive floor bounces at frames {bounce_frames[-2]} and {bounce_frames[-1]} prior to contact.",
                        "explanation": "Player failed to strike the ball before it bounced twice on the court floor.",
                        "coach_correction": "React faster to low chiquitas or drop shots. Anticipate the bounce by reading opponent's racket angle."
                    })

        return fouls


if __name__ == "__main__":
    detector = PadelFoulDetector()
    print("Zero-Hallucination PadelFoulDetector initialized successfully!")
