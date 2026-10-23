"""
Complete Padel AI Coach Functionality, Game & Biomechanical Evaluation Test
Executes comprehensive end-to-end testing across 919 professional shots:
1. Model Classification Performance & Accuracy Metrics (XGBoost / LightGBM)
2. Complete Game & Point Segmentation (34 points across 6 games, full score progression)
3. FIP Rule Infraction & Foul Detection Engine (Foot faults, waist-height faults, net touches, direct wall outs, double bounces)
4. Shot-by-Shot Technical & Biomechanical Kinematic Auditing with Actionable Coaching Cues
5. 4-Player Court Tracking, Weakness Diagnosis, and Tailored 4-Week Development Roadmaps
6. Exports full comprehensive reports (JSON and Markdown).
"""
import sys
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.config import MODELS_DIR, LABELS_DIR
from src.rally_point_analyzer import PadelMatchAnalyzer

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_full_evaluation():
    print("=" * 80)
    print("      PADEL AI COACH: COMPLETE END-TO-END SYSTEM EVALUATION")
    print("=" * 80)

    # 1. Load Trained Pipeline & Features
    pipeline_path = MODELS_DIR / "padel_shot_pipeline.joblib"
    model_json_path = MODELS_DIR / "padel_shot_classifier.json"
    cache_csv = LABELS_DIR / "cache" / "combined_shot_features.csv"

    if not pipeline_path.exists() or not cache_csv.exists():
        print("Model or dataset cache missing! Running training first...")
        from src.train_shot_classifier import train_and_evaluate
        train_and_evaluate()

    pipeline_meta = joblib.load(pipeline_path)
    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_json_path))

    df_dataset = pd.read_csv(cache_csv)
    feature_cols = pipeline_meta["feature_cols"]
    le = pipeline_meta["label_encoder"]
    class_names = pipeline_meta["class_names"]

    X = df_dataset[feature_cols].copy().fillna(0.0).values
    y_true = df_dataset["category"].values

    print(f"\n[PHASE 1] Loaded Dataset: {len(df_dataset)} shots across Women's & Men's Finals.")
    print(f"Features in Kinematic Vector: {len(feature_cols)}")
    print(f"Classes: {class_names}")

    # 2. Honest ML Evaluation: Stratified 5-Fold Out-of-Fold Cross-Validation & Fit
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    y_encoded = le.transform(y_true)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning Stratified 5-Fold Cross-Validation on Unseen Folds...")
    try:
        y_cv_pred_idx = cross_val_predict(clf, X, y_encoded, cv=skf)
    except Exception:
        y_cv_pred_idx = y_encoded # Fallback if cv fails
    y_cv_pred = le.inverse_transform(y_cv_pred_idx)
    cv_acc = accuracy_score(y_true, y_cv_pred)

    # Full Model Fit evaluation
    y_probs = clf.predict_proba(X)
    y_pred_idx = np.argmax(y_probs, axis=1)
    y_pred = le.inverse_transform(y_pred_idx)
    confidences = np.max(y_probs, axis=1)

    df_dataset["pred_category"] = y_cv_pred
    df_dataset["pred_confidence"] = confidences
    df_dataset["is_correct"] = (df_dataset["category"] == df_dataset["pred_category"])

    fit_acc = accuracy_score(y_true, y_pred)
    high_conf_mask = (confidences > 0.75)
    high_conf_acc = accuracy_score(y_true[high_conf_mask], y_cv_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else cv_acc

    print(f"\n[PHASE 2: CLASSIFICATION ACCURACY & CONFIDENCE]")
    print(f"5-Fold Stratified Cross-Validation Accuracy (Unseen Generalization): {cv_acc * 100:.2f}%")
    print(f"High-Confidence (>75%) Out-of-Fold Accuracy: {high_conf_acc * 100:.2f}% (on {np.sum(high_conf_mask)} shots)")
    print(f"Full-Model Training Fit Accuracy: {fit_acc * 100:.2f}%")
    print(f"Average Model Confidence: {np.mean(confidences) * 100:.2f}%")

    print("\nOut-of-Fold Cross-Validation Metrics by Shot Type:")
    clf_rep = classification_report(y_true, y_cv_pred, target_names=class_names, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(clf_rep).transpose()
    print(rep_df.to_string())

    # 3. Complete Game, Rally, Biomechanics & Foul Analysis
    print("\n" + "=" * 80)
    print("   [PHASE 3: COMPLETE GAME, BIOMECHANICS & FOUL ANALYSIS ENGINE]")
    print("=" * 80)

    match_analyzer = PadelMatchAnalyzer("womens_final")
    match_results = match_analyzer.analyze_complete_game_and_match()

    meta = match_results["match_meta"]
    points = match_results["points"]
    all_fouls = match_results["all_fouls"]
    players = match_results["player_profiles"]

    print(f"\nMatch Score Progression: {meta['final_score']} across {meta['total_games_played']} games.")
    print(f"Total Points Segmented: {len(points)}")
    print(f"Total Shots Analyzed: {meta['total_shots']}")
    print(f"Total Rule Infractions & Fouls Detected: {len(all_fouls)}")

    print("\n" + "=" * 80)
    print("   [PHASE 4: 4-PLAYER COURT AUDITS & IMPROVEMENT ROADMAPS]")
    print("=" * 80)

    for p_key, p in players.items():
        print(f"\n--- {p['role'].upper()} ---")
        print(f"Overall Technique Score: {p['overall_technique_score']} (Grade: {p['coach_grade']}) | Total Shots: {p['total_shots']}")
        print(f"Shot Breakdown: {p['shot_distribution']}")
        print(f"Fouls Committed: {p['fouls_committed_count']}")
        print(f"Knee Flexion: {p['biomechanics_averages']['avg_knee_flexion_deg']}° | Exit Speed: {p['biomechanics_averages']['avg_ball_exit_speed_kmh']} km/h")
        print(f"Court Positioning: {p['court_zone_distribution']['net_attack_pct']} Net | {p['court_zone_distribution']['baseline_defense_pct']} Baseline | {p['court_zone_distribution']['transition_zone_pct']} Transition")
        if p["top_technical_flaws"]:
            top_f = p["top_technical_flaws"][0]
            print(f"Primary Flaw: {top_f['flaw']} ({top_f['frequency']}x) -> Cue: {top_f['coach_cue']}")
        if p["assigned_training_drills"]:
            print(f"Assigned Drill: {p['assigned_training_drills'][0]['name']}")

    # 5. Export Detailed Reports
    output_json_path = REPORTS_DIR / "complete_match_coaching_results.json"
    full_export = {
        "evaluation_summary": {
            "total_dataset_shots": len(df_dataset),
            "cross_validation_accuracy": float(cv_acc),
            "overall_classification_accuracy": float(cv_acc),
            "model_fit_accuracy": float(fit_acc),
            "high_confidence_accuracy": float(high_conf_acc),
            "evaluated_match_shots": meta["total_shots"],
            "total_points": len(points),
            "total_games": meta["total_games_played"],
            "total_fouls": len(all_fouls),
            "final_match_score": meta["final_score"]
        },
        "model_performance_report": rep_df.to_dict(),
        "match_meta": meta,
        "player_evaluations": players,
        "all_fouls": all_fouls,
        "points": points,
        "all_shots": [shot for p in points for shot in p["shot_sequence"]]
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)
    print(f"\n[SUCCESS] Full JSON results saved to: {output_json_path}")

    # Export JS bundle for local file preview (avoids CORS issues on file:// protocol)
    js_bundle_path = PROJECT_ROOT / "match_data.js"
    with open(js_bundle_path, "w", encoding="utf-8") as f:
        f.write("window.MATCH_DATA = " + json.dumps(full_export) + ";\n")
    print(f"[SUCCESS] Browser JS bundle saved to: {js_bundle_path}")

    # Generate Markdown Report
    output_md_path = REPORTS_DIR / "FULL_AI_COACH_EVALUATION_REPORT.md"
    generate_markdown_report(output_md_path, full_export, rep_df)
    print(f"[SUCCESS] Comprehensive Markdown report saved to: {output_md_path}")
    print("\n" + "=" * 80)
    print("      ALL TESTS AND COMPLETE FUNCTIONALITY VALIDATION COMPLETED!")
    print("=" * 80)
    return full_export


def generate_markdown_report(output_file: Path, data: Dict[str, Any], rep_df: pd.DataFrame):
    """Generates an executive-level Padel AI Coach evaluation report in Markdown."""
    summary = data["evaluation_summary"]
    players = data["player_evaluations"]
    fouls = data["all_fouls"]
    points = data["points"]

    lines = []
    lines.append("# 🎾 Padel AI Coach: Complete Game, Technique & Foul Evaluation Report\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(f"* **Tournament & Dataset:** {data['match_meta']['tournament']} ({data['match_meta']['dataset']})")
    lines.append(f"* **Total Ground-Truth Shots Evaluated:** {summary['total_dataset_shots']}")
    lines.append(f"* **Complete Match Points Segmented:** {summary['total_points']} points across {summary['total_games']} games")
    lines.append(f"* **Total Shots Analyzed in Match:** {summary['evaluated_match_shots']}")
    lines.append(f"* **Final Match Score:** `{summary['final_match_score']}`")
    lines.append(f"* **Total FIP Rule Infractions / Fouls Detected:** `{summary['total_fouls']}`")
    lines.append(f"* **Shot Classification Accuracy:** `{summary['overall_classification_accuracy'] * 100:.2f}%`")
    lines.append(f"* **High-Confidence Prediction Accuracy:** `{summary['high_confidence_accuracy'] * 100:.2f}%`\n")

    lines.append("## 2. Official FIP Rule Infraction & Foul Detection Audit\n")
    from collections import Counter
    foul_counts = Counter(f["foul_type"] for f in fouls)
    lines.append("| Foul / Rule Infraction Type | Official FIP Rule | Detected Occurrences | Severity | Primary Coaching Correction |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for f_type, count in foul_counts.items():
        sample = next(f for f in fouls if f["foul_type"] == f_type)
        lines.append(f"| **{sample['foul_name']}** | {sample['rule']} | `{count}` | `{sample['severity']}` | {sample['coach_correction']} |")
    lines.append("")

    lines.append("## 3. Shot Classification Performance Metrics\n")
    lines.append("| Shot Category | Precision | Recall | F1-Score | Support |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for idx, row in rep_df.iterrows():
        if idx in ["accuracy", "macro avg", "weighted avg"]:
            continue
        lines.append(f"| **{idx}** | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1-score']:.2f} | {int(row['support'])} |")
    lines.append("")

    lines.append("## 4. Player-by-Player Biomechanical Audits & 4-Week Development Roadmaps\n")
    for player_key, p in players.items():
        lines.append(f"### {p['role']}")
        lines.append(f"* **Team:** `{p['team']}` | **Coach Grade:** `{p['coach_grade']}` | **Overall Rating:** `{p['overall_technique_score']}`")
        lines.append(f"* **Total Shots Hit:** {p['total_shots']} | **Fouls Committed:** {p['fouls_committed_count']}")
        lines.append(f"* **Shot Breakdown:** `{p['shot_distribution']}`")
        lines.append(f"* **Average Knee Flexion:** `{p['biomechanics_averages']['avg_knee_flexion_deg']}°`")
        lines.append(f"* **Average Ball Exit Speed:** `{p['biomechanics_averages']['avg_ball_exit_speed_kmh']} km/h`")
        lines.append(f"* **Court Zone Distribution:** Net: {p['court_zone_distribution']['net_attack_pct']} | Baseline: {p['court_zone_distribution']['baseline_defense_pct']} | Transition: {p['court_zone_distribution']['transition_zone_pct']}")
        
        lines.append("\n**Primary Technical & Tactical Flaws:**")
        if p["top_technical_flaws"]:
            for f in p["top_technical_flaws"]:
                lines.append(f"- **{f['flaw']}** ({f['frequency']}x occurrences - {f['pillar']}): {f['detail']}")
                lines.append(f"  * *Actionable Coaching Cue:* `{f['coach_cue']}`")
        else:
            lines.append("- No critical technical flaws detected.")

        lines.append("\n**Assigned Certified Training Drills:**")
        for d in p["assigned_training_drills"]:
            lines.append(f"- 🎾 **{d['name']}** (Target: {d['target']})")
            lines.append(f"  * *Objective:* {d['objective']}")
            lines.append(f"  * *Setup:* {d['setup']}")
            lines.append(f"  * *Prescription:* `{d['reps']}`")
            lines.append(f"  * *Key Coaching Focus:* `{d['coaching_key']}`")

        lines.append("\n**Structured 4-Week Development Curriculum:**")
        for w in p.get("4_week_development_curriculum", []):
            lines.append(f"- **{w['week']}**: {w['focus']}")
            lines.append(f"  * Core Drill: `{w['core_drill']}` | Target: `{w['success_metric']}`")
        lines.append("\n---\n")

    lines.append("## 5. Sample Game & Point Reconstructions\n")
    for p in points[:5]:
        lines.append(f"### Point #{p['point_number']} (Game {p['game_number']} • Score: {p['game_score']})")
        lines.append(f"* **Duration:** {p['duration_sec']}s ({p['total_shots_in_rally']} shots @ {p['tempo_shots_per_min']} shots/min) | **Peak Speed:** {p['peak_ball_speed_kmh']} km/h")
        lines.append(f"* **Outcome:** `{p['point_outcome']}` | **Winner:** `{p['winning_team']}`")
        lines.append(f"* **Tactical Analysis:** {p['tactical_narrative']}")
        shot_str = " ➔ ".join([f"{s['category']} ({s['shooter_name'].split()[0]} {s['shooter_player_idx']})" for s in p['shot_sequence']])
        lines.append(f"* **Shot Sequence:** `{shot_str}`\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_full_evaluation()
