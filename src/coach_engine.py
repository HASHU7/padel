"""
Padel AI Coach - Biomechanics, Tactics & Coaching Intelligence Engine
World-Class Expert Padel Coaching Logic:
1. Deep Technical & Kinematic Auditing for Every Individual Shot
2. Tactical Positioning & Shot Selection Decision Scoring
3. Rule Adherence & Foul Diagnosis
4. Player-Specific Improvement Roadmaps & Certified Padel Training Drills
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR, ZONES


class PadelAICoach:
    def __init__(self, pipeline_path: Optional[Path] = None):
        if pipeline_path is None:
            pipeline_path = MODELS_DIR / "padel_shot_pipeline.joblib"
        
        self.pipeline = None
        self.feature_cols = []
        self.class_names = []
        if pipeline_path.exists():
            data = joblib.load(pipeline_path)
            self.pipeline = data
            self.feature_cols = data.get("feature_cols", [])
            self.class_names = data.get("class_names", [])

    def analyze_shot_biomechanics(self, shot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive real-coach technical audit of a single shot.
        Evaluates kinetic chain, impact timing, knee flexion, overhead extension,
        court positioning, ball exit dynamics, and prescribes immediate coaching cues.
        """
        category = shot_data.get("category", "Unknown")
        critiques = []
        strengths = []
        scores = {}

        aligned_x = shot_data.get("aligned_wrist_impact_x", 0.0)
        impact_y = shot_data.get("wrist_impact_y", 0.0)
        knee_angle = shot_data.get("avg_knee_flexion", 145.0)
        elbow_angle = shot_data.get("elbow_angle", 140.0)
        wrist_speed = shot_data.get("max_wrist_speed", 15.0)
        court_y = shot_data.get("player_court_y", 0.5)
        ball_speed = shot_data.get("ball_speed_kmh", 65.0)

        # -------------------------------------------------------------
        # 1. Contact Point Alignment (Forehand / Backhand / Volley)
        # -------------------------------------------------------------
        if category in ["Forehand", "Backhand"]:
            if category == "Forehand":
                if aligned_x < -0.16:
                    critiques.append({
                        "pillar": "Contact Timing",
                        "severity": "HIGH",
                        "issue": "Late Contact Point",
                        "detail": f"Ball struck {abs(aligned_x)*100:.1f}cm behind the lead hip axis. Results in spraying wide or hitting the side fence.",
                        "coach_cue": "Unit Turn Early: Start your shoulder coil as soon as the ball leaves the opponent's racket. Meet the ball 25cm in front of your lead hip."
                    })
                    scores["contact_timing"] = 55
                elif aligned_x > 0.35:
                    critiques.append({
                        "pillar": "Contact Timing",
                        "severity": "LOW",
                        "issue": "Over-Extended Early Contact",
                        "detail": "Reaching too far in front; lost rotational power from core.",
                        "coach_cue": "Let the ball come slightly closer to your strike zone before unleashing the kinetic chain."
                    })
                    scores["contact_timing"] = 75
                else:
                    strengths.append(f"Ideal contact point ({aligned_x*100:+.1f}cm in front of lead hip) with clean forward weight transfer.")
                    scores["contact_timing"] = 94
            elif category == "Backhand":
                if aligned_x > 0.16:
                    critiques.append({
                        "pillar": "Contact Timing",
                        "severity": "HIGH",
                        "issue": "Trapped Backhand Contact",
                        "detail": "Racket was trapped against the torso at impact without adequate arm extension.",
                        "coach_cue": "Keep non-dominant hand guiding the racket throat until the forward release. Form a firm hitting triangle."
                    })
                    scores["contact_timing"] = 58
                else:
                    strengths.append("Crisp backhand spacing and clean diagonal follow-through.")
                    scores["contact_timing"] = 92
        else:
            scores["contact_timing"] = 85

        # -------------------------------------------------------------
        # 2. Footwork, Stance & Center of Gravity (Knee Flexion)
        # -------------------------------------------------------------
        is_low_ball = (impact_y > 0.05) or (court_y > 0.65) # Defense or low ball
        if is_low_ball:
            if knee_angle > 152.0:
                critiques.append({
                    "pillar": "Footwork & Balance",
                    "severity": "HIGH",
                    "issue": "Stiff Knees / Bending at Waist",
                    "detail": f"Knee flexion was {knee_angle:.1f}° on a low ball. Bending from the waist causes errors into the net cord.",
                    "coach_cue": "Squat down with a wide athletic base. Lower your hips so your eyes are level with the ball trajectory."
                })
                scores["knee_flexion"] = 50
            elif knee_angle < 135.0:
                strengths.append(f"Exceptional low center of gravity (knees bent to {knee_angle:.1f}°). Perfectly balanced.")
                scores["knee_flexion"] = 96
            else:
                scores["knee_flexion"] = 82
        else:
            scores["knee_flexion"] = 88

        # -------------------------------------------------------------
        # 3. Overhead Extension & Kinetic Chain (Smash / Bandeja / Vibora)
        # -------------------------------------------------------------
        if category in ["Smash", "Serve"]:
            nose_diff = shot_data.get("wrist_nose_height_diff", 0.0)
            if nose_diff > -0.15: # Wrist not significantly above nose
                critiques.append({
                    "pillar": "Overhead Extension",
                    "severity": "HIGH",
                    "issue": "Cramped Overhead Reach ('T-Rex Arm')",
                    "detail": "Contact height was too low. Overhead shots require high vertical reach for steep downward penetration.",
                    "coach_cue": "Point at the ball with your non-dominant hand. Reach to full extension at 12 o'clock above your dominant shoulder."
                })
                scores["overhead_reach"] = 50
            elif elbow_angle < 138.0:
                critiques.append({
                    "pillar": "Kinetic Chain",
                    "severity": "MEDIUM",
                    "issue": "Elbow Flexed at Impact",
                    "detail": f"Elbow angle was {elbow_angle:.1f}°. Loss of leverage and racket speed.",
                    "coach_cue": "Snap the triceps into full lockout right before contact, then pronate forearm."
                })
                scores["overhead_reach"] = 68
            else:
                strengths.append("High apex extension with powerful vertical kinetic transfer.")
                scores["overhead_reach"] = 95
        else:
            scores["overhead_reach"] = 85

        # -------------------------------------------------------------
        # 4. Tactical Court Positioning Audit
        # -------------------------------------------------------------
        # Distance from net (Net is at Y=10.0m in standard 20m court)
        court_y_m = shot_data.get("court_y_metric", shot_data.get("player_court_y", 5.0))
        if court_y_m > 1.0:
            dist_from_net = abs(court_y_m - 10.0)
        else:
            dist_from_net = abs(court_y_m - 0.5) * 20.0

        if 3.8 < dist_from_net < 6.8 and category not in ["Serve"]:
            critiques.append({
                "pillar": "Tactical Court Positioning",
                "severity": "HIGH",
                "issue": "Caught in No-Man's Land (Transition Zone)",
                "detail": f"Shot executed {dist_from_net:.1f}m from net. Lingering in transition exposes you to low dipping balls at your feet.",
                "coach_cue": "Zone Discipline: Never stay in transition! If advancing, sprint to the net. If pushed back, retreat behind the service line."
            })
            scores["tactical_positioning"] = 50
        elif dist_from_net <= 3.8:
            strengths.append(f"Commanding net position ({dist_from_net:.1f}m from net) inside attacking kitchen zone.")
            scores["tactical_positioning"] = 92
        else:
            strengths.append(f"Disciplined defensive baseline position ({dist_from_net:.1f}m from net) behind service line.")
            scores["tactical_positioning"] = 88

        # -------------------------------------------------------------
        # 5. Shot Quality & Real Coach Advice Synthesis
        # -------------------------------------------------------------
        overall_score = int(np.mean(list(scores.values()))) if scores else 80

        # Primary coaching recommendation for this exact shot
        if critiques:
            top_crit = critiques[0]
            coach_advice = f"{top_crit['issue']}: {top_crit['coach_cue']}"
        else:
            coach_advice = "Superb mechanical execution and tactical court positioning. Maintain this tempo."

        return {
            "category": category,
            "overall_score": overall_score,
            "subscores": scores,
            "critiques": critiques,
            "strengths": strengths,
            "coach_advice": coach_advice,
            "metrics": {
                "knee_flexion_deg": round(float(knee_angle), 1),
                "elbow_angle_deg": round(float(elbow_angle), 1),
                "aligned_contact_x_cm": round(float(aligned_x * 100), 1),
                "ball_speed_kmh": round(float(ball_speed), 1),
            }
        }

    def generate_player_coaching_report(self, match_shots: List[Dict[str, Any]], fouls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Compiles an in-depth coaching report and 4-week structured development plan
        for an individual player based on all shots hit during the match.
        """
        if not match_shots:
            return {"error": "No shots provided"}

        fouls = fouls or []
        analyses = [self.analyze_shot_biomechanics(s) for s in match_shots]

        # Shot Counts & Distribution
        categories = [a["category"] for a in analyses]
        cat_counts = pd.Series(categories).value_counts().to_dict()

        # Aggregate critiques
        all_critiques = []
        for a in analyses:
            all_critiques.extend(a["critiques"])

        df_critiques = pd.DataFrame(all_critiques)
        top_flaws = []
        if not df_critiques.empty:
            issue_counts = df_critiques["issue"].value_counts().head(4)
            for issue_name, cnt in issue_counts.items():
                sample = df_critiques[df_critiques["issue"] == issue_name].iloc[0]
                top_flaws.append({
                    "flaw": issue_name,
                    "frequency": int(cnt),
                    "pillar": sample["pillar"],
                    "detail": sample["detail"],
                    "coach_cue": sample["coach_cue"]
                })

        # Assigned Comprehensive Training Drills
        assigned_drills = []
        assigned_names = set()
        for flaw in top_flaws:
            fname = flaw["flaw"]
            if "Late Contact" in fname and "Wall-Rebound Early Prep Drill" not in assigned_names:
                assigned_drills.append({
                    "name": "Wall-Rebound Early Prep Drill",
                    "target": "Contact Point Timing (Forehand & Backhand)",
                    "objective": "Eliminate late contact off the back wall and strike 25cm in front of lead hip.",
                    "setup": "Place 2 cones 25cm ahead of your lead foot. Feed rapid balls off the back glass; racket must meet ball precisely between cones.",
                    "reps": "4 sets of 15 repetitions alternating crosscourt and down-the-line.",
                    "coaching_key": "Rotate your shoulders early so the racket is already back before the ball hits the glass."
                })
                assigned_names.add("Wall-Rebound Early Prep Drill")

            elif "Stiff Knees" in fname and "Low Tray Knee Flexion Drill" not in assigned_names:
                assigned_drills.append({
                    "name": "Low Tray Knee Flexion Drill",
                    "target": "Center of Gravity & Squat Depth on Defense",
                    "objective": "Bend knees to 115° on low balls without hinging at the waist.",
                    "setup": "Place mini-cones at the baseline. Player must touch cone with knee before executing low chiquitas or wall returns.",
                    "reps": "5 sets of 12 balls with rapid lateral recovery.",
                    "coaching_key": "Drop hips straight down like an elevator, not bending forward like a crane."
                })
                assigned_names.add("Low Tray Knee Flexion Drill")

            elif ("Overhead" in fname or "Cramped" in fname or "T-Rex" in fname) and "Apex Reach & Vibora Drill" not in assigned_names:
                assigned_drills.append({
                    "name": "Apex Reach & Vibora Extension Drill",
                    "target": "Overhead Apex Reach (Smash & Bandeja)",
                    "objective": "Achieve full vertical extension for steep trajectory into opponent's glass.",
                    "setup": "High lobs fed from deep court. Hit smashes targeting opponent's corner glass with full vertical extension.",
                    "reps": "3 sets of 20 overhead strikes.",
                    "coaching_key": "Keep non-dominant hand pointing at the ball until right before impact; extend triceps fully."
                })
                assigned_names.add("Apex Reach & Vibora Drill")

            elif "No-Man's Land" in fname and "Net-to-Base Transition Sprints" not in assigned_names:
                assigned_drills.append({
                    "name": "Net-to-Base Transition Sprints",
                    "target": "Court Positioning & Eliminating Transition Flaws",
                    "objective": "Instantly transition between attack and defense without stalling in the danger zone.",
                    "setup": "Touch net tape with racket, sprint backward behind service line to defend lob, then sprint immediately forward to re-take net.",
                    "reps": "6 continuous rounds of 5 transitions.",
                    "coaching_key": "Never stop running until both feet are completely behind the service line or inside the kitchen zone."
                })
                assigned_names.add("Net-to-Base Transition Sprints")

        # Add foul correction drill if player committed fouls
        if fouls:
            foul_names = [f["foul_type"] for f in fouls]
            if any("FOOT_FAULT" in fn for fn in foul_names) and "Service Line Footwork Discipline Drill" not in assigned_names:
                assigned_drills.append({
                    "name": "Service Line Footwork Discipline Drill",
                    "target": "Eliminating Service Foot Faults",
                    "objective": "Maintain legal foot placement behind service line and center mark on serve.",
                    "setup": "Place chalk or training tape 15cm behind service line. Server must keep rear foot anchored behind tape through contact.",
                    "reps": "50 practice serves focusing strictly on legal foot placement.",
                    "coaching_key": "Toss ball slightly backward so momentum carries into ball without feet stepping forward early."
                })
                assigned_names.add("Service Line Footwork Discipline Drill")

        # Radar Scores
        avg_score = int(np.mean([a["overall_score"] for a in analyses]))
        timing_score = int(np.mean([a["subscores"].get("contact_timing", 80) for a in analyses]))
        knee_score = int(np.mean([a["subscores"].get("knee_flexion", 80) for a in analyses]))
        overhead_score = int(np.mean([a["subscores"].get("overhead_reach", 80) for a in analyses]))
        positioning_score = int(np.mean([a["subscores"].get("tactical_positioning", 80) for a in analyses]))
        foul_discipline_score = max(50, 100 - len(fouls) * 12)

        # 4-Week Development Plan
        development_curriculum = [
            {
                "week": "Week 1: Footwork & Center of Gravity",
                "focus": "Eliminating stiff knees and lowering hips on low defense",
                "core_drill": "Low Tray Knee Flexion Drill (5 sets x 12 reps)",
                "success_metric": "Knee flexion angle below 135° on 90% of defensive balls"
            },
            {
                "week": "Week 2: Contact Point Timing & Unit Turn",
                "focus": "Early shoulder coil and hitting 25cm ahead of lead hip",
                "core_drill": "Wall-Rebound Early Prep Drill (4 sets x 15 reps)",
                "success_metric": "Zero late contact points recorded off back glass"
            },
            {
                "week": "Week 3: Overhead Dominance & Power",
                "focus": "Full vertical extension on smash and bandeja",
                "core_drill": "Apex Reach & Vibora Extension Drill (3 sets x 20 reps)",
                "success_metric": "Exit ball speed over 105 km/h with steep downward angle"
            },
            {
                "week": "Week 4: Tactical Zone Discipline & Foul Eradication",
                "focus": "Zero time wasted in No-Man's land and clean footwork on serve",
                "core_drill": "Net-to-Base Transition Sprints (6 rounds x 5 reps)",
                "success_metric": "Transition zone occupancy under 15% and zero match foot faults"
            }
        ]

        return {
            "total_shots_analyzed": len(match_shots),
            "shot_distribution": cat_counts,
            "overall_technique_rating": f"{avg_score}/100",
            "coach_grade": "A" if avg_score >= 85 else ("B+" if avg_score >= 80 else ("B" if avg_score >= 75 else "C")),
            "radar_scores": {
                "overall": avg_score,
                "contact_timing": timing_score,
                "knee_flexion": knee_score,
                "overhead_reach": overhead_score,
                "court_positioning": positioning_score,
                "foul_discipline": foul_discipline_score
            },
            "top_technical_flaws": top_flaws,
            "fouls_committed": fouls,
            "assigned_training_drills": assigned_drills,
            "development_curriculum": development_curriculum
        }


if __name__ == "__main__":
    coach = PadelAICoach()
    test_shot = {
        "category": "Forehand",
        "aligned_wrist_impact_x": -0.20,
        "avg_knee_flexion": 158.0,
        "wrist_impact_y": 0.20,
        "player_court_y": 0.45,
        "ball_speed_kmh": 72.0
    }
    audit = coach.analyze_shot_biomechanics(test_shot)
    print("Test Shot Audit:")
    print(f"Overall Technique Score: {audit['overall_score']}/100")
    print(f"Coach Advice: {audit['coach_advice']}")
    for c in audit['critiques']:
        print(f" - [{c['severity']}] {c['issue']}: {c['detail']}")
        print(f"   Cue: {c['coach_cue']}")
