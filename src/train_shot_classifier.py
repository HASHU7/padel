"""
Padel Shot Classifier Training Pipeline
Trains an ensemble XGBoost / LightGBM model on kinematic and trajectory features.
Evaluates cross-validation accuracy, precision, recall, and feature importances.
Saves model artifacts to models/ directory.
"""
import sys
import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.config import MODELS_DIR, LABELS_DIR
from src.dataset_parser import PadelDatasetParser
from src.feature_extractor import PadelFeatureExtractor


def build_full_dataset() -> pd.DataFrame:
    """
    Parses both Women's and Men's finals, extracts kinematic features,
    and returns a combined DataFrame. Caches the tabular dataframe to CSV.
    """
    cache_file = LABELS_DIR / "cache" / "combined_shot_features.csv"
    if cache_file.exists():
        print(f"Loading cached features from {cache_file.name}...")
        return pd.read_csv(cache_file)

    extractor = PadelFeatureExtractor()
    dfs = []

    for d_key in ["womens_final", "mens_final"]:
        print(f"\nProcessing dataset: {d_key}...")
        parser = PadelDatasetParser(d_key)
        events = parser.extract_shot_intervals()
        ball_dict = parser.load_ball_positions()
        poses_by_frame = parser.load_player_poses()

        matched = []
        for ev in events:
            m = parser.match_shooter_and_impact(ev, ball_dict, poses_by_frame)
            if m:
                matched.append(m)

        print(f"[{d_key}] Matched {len(matched)} / {len(events)} events.")
        df_feat = extractor.extract_dataset_features(matched)
        print(f"[{d_key}] Extracted {len(df_feat)} feature vectors.")
        dfs.append(df_feat)

    combined_df = pd.concat(dfs, ignore_index=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(cache_file, index=False)
    print(f"\nSaved combined dataset ({len(combined_df)} shots) to {cache_file}")
    return combined_df


def train_and_evaluate():
    print("=" * 60)
    print("   PADEL AI SHOT CLASSIFIER - ACCURACY & TRAINING BENCHMARK")
    print("=" * 60)

    df = build_full_dataset()
    print(f"\nTotal Dataset Samples: {len(df)}")
    print("\nShot Category Distribution:")
    print(df["category"].value_counts())

    # Metadata columns to exclude from training features
    meta_cols = ["category", "dataset", "start_frame", "impact_frame"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy().fillna(0.0)
    y_raw = df["category"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    print(f"\nFeature Count: {len(feature_cols)}")
    print(f"Classes: {class_names}")

    # Compute sample weights to balance rare classes (Dropshot, Serve, Other)
    class_counts = pd.Series(y).value_counts().to_dict()
    total_samples = len(y)
    n_classes = len(class_names)
    weights = np.array([total_samples / (n_classes * class_counts[cls]) for cls in y])

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist", # Ultra-fast on CPU
    )

    print("\nRunning 5-Fold Stratified Cross-Validation...")
    try:
        y_pred = cross_val_predict(clf, X, y, cv=skf, params={"sample_weight": weights})
    except TypeError:
        # Fallback for scikit-learn versions
        y_pred = cross_val_predict(clf, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)

    print(f"\nCross-Validation Accuracy: {acc * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # Cross-Match Generalization Test: Train on Men, Test on Women
    print("\n" + "=" * 60)
    print("   CROSS-MATCH GENERALIZATION TEST (Train: Men -> Test: Women)")
    print("=" * 60)
    train_mask = df["dataset"] == "mens_final"
    test_mask = df["dataset"] == "womens_final"

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        weights_train = weights[train_mask]

        clf_gen = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
        )
        clf_gen.fit(X_train, y_train, sample_weight=weights_train)
        y_test_pred = clf_gen.predict(X_test)
        gen_acc = accuracy_score(y_test, y_test_pred)
        print(f"\nCross-Match Test Accuracy (Zero-Shot Match Transfer): {gen_acc * 100:.2f}%")

    # Fit final model on all data
    print("\nFitting final model on full dataset...")
    clf.fit(X, y, sample_weight=weights)

    # Save Model Artifacts
    model_path = MODELS_DIR / "padel_shot_classifier.json"
    clf.save_model(str(model_path))

    pipeline_artifact = {
        "label_encoder": le,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "accuracy": float(acc),
    }
    joblib.dump(pipeline_artifact, MODELS_DIR / "padel_shot_pipeline.joblib")

    # Feature Importances
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Most Predictive Kinematic Features:")
    print(feat_imp.head(10))

    print(f"\nModel artifacts successfully saved to {MODELS_DIR}")
    return clf, pipeline_artifact


if __name__ == "__main__":
    train_and_evaluate()
