"""
MantiQ-Auth: Data Leakage & Overfitting Analysis

This script rigorously tests the current TamperingClassifier to detect:
1. "Same-Original" Data Leakage: where tampered images from the same patient/original
   are split across train and validation sets, allowing the model to cheat.
2. Overfitting: by comparing Train vs Validation performance.
3. Label Shuffling Leakage: by training on randomized labels to ensure AUC drops to ~0.5.

Usage:
    python scripts/test_leakage_and_overfitting.py
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_extraction import extract_enriched_resnet_features, load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("LeakageTest")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Max rows to process (to save time)")
    args = parser.parse_args()

    data_root = Path("data")
    tamper_csv = data_root / "tampered_dataset" / "metadata.csv"

    if not tamper_csv.exists():
        logger.error(f"Cannot find metadata CSV at {tamper_csv}")
        sys.exit(1)

    # 1. Load Data
    logger.info("Loading dataset and extracting features...")
    features, labels, groups = [], [], []
    
    with open(tamper_csv, "r") as f:
        rows = list(csv.DictReader(f))
        
    # Limit for speed in testing
    rows = rows[:args.limit]
    
    # Track groups
    group_map = {}
    group_counter = 0

    for i, row in enumerate(rows):
        fpath = data_root / row["image_path"]
        if not fpath.exists(): continue
        
        try:
            tensor = load_and_preprocess_image(fpath)
            feat = extract_enriched_resnet_features(tensor)
            features.append(feat)
            labels.append(int(row["label"]))
            
            # Map source_image to a unique group ID
            src = row["source_image"]
            if src not in group_map:
                group_map[src] = group_counter
                group_counter += 1
            groups.append(group_map[src])
            
        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

        if (i+1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(rows)}")

    X = np.array(features)
    y = np.array(labels)
    groups = np.array(groups)
    
    logger.info(f"Dataset loaded: {len(X)} samples, {len(np.unique(groups))} unique original images.")
    if len(X) < 20:
        logger.error("Not enough data to run tests.")
        sys.exit(1)

    # Base Classifier
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42))
    ])

    # --------------------------------------------------------------------------------
    # TEST 1: Current Split (StratifiedKFold) - Likely Leaking
    # --------------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Current Splitting Strategy (StratifiedKFold)")
    logger.info("This does NOT group by original image. It may suffer from 'same-original' leakage.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_leaked = cross_validate(clf, X, y, cv=skf, scoring='roc_auc', return_train_score=True)
    
    train_auc_leaked = np.mean(cv_results_leaked['train_score'])
    val_auc_leaked = np.mean(cv_results_leaked['test_score'])
    
    logger.info(f"  Train AUC: {train_auc_leaked:.4f}")
    logger.info(f"  Val AUC:   {val_auc_leaked:.4f}")
    if val_auc_leaked > 0.99:
        logger.warning("  Val AUC is practically 1.0! Overfitting/Leakage is highly probable.")

    # --------------------------------------------------------------------------------
    # TEST 2: Label Shuffling (Randomized Labels)
    # --------------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Label Shuffling Test")
    logger.info("Training on completely random labels. Expected Val AUC is ~0.50.")
    
    np.random.seed(42)
    y_shuffled = np.random.permutation(y)
    cv_results_shuffled = cross_validate(clf, X, y_shuffled, cv=skf, scoring='roc_auc', return_train_score=True)
    
    val_auc_shuffled = np.mean(cv_results_shuffled['test_score'])
    logger.info(f"  Train AUC (Shuffled): {np.mean(cv_results_shuffled['train_score']):.4f}")
    logger.info(f"  Val AUC (Shuffled):   {val_auc_shuffled:.4f}")
    if val_auc_shuffled > 0.7:
        logger.error("  Val AUC on shuffled labels is HIGH. There is a serious bug in data preprocessing/feature extraction.")
    else:
        logger.info("  Val AUC is near 0.5. Features themselves are not leaking the labels.")

    # --------------------------------------------------------------------------------
    # TEST 3: GroupKFold (Preventing Same-Original Leakage)
    # --------------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Leakage-Free Splitting (GroupKFold)")
    logger.info("Ensures tampered versions of an image are NEVER in the training set if the original is in validation.")
    
    gkf = GroupKFold(n_splits=5)
    cv_results_grouped = cross_validate(clf, X, y, groups=groups, cv=gkf, scoring='roc_auc', return_train_score=True)
    
    train_auc_grouped = np.mean(cv_results_grouped['train_score'])
    val_auc_grouped = np.mean(cv_results_grouped['test_score'])
    
    logger.info(f"  Train AUC: {train_auc_grouped:.4f}")
    logger.info(f"  Val AUC:   {val_auc_grouped:.4f}")
    
    auc_drop = val_auc_leaked - val_auc_grouped
    if auc_drop > 0.1:
        logger.warning(f"  Massive AUC drop ({auc_drop:.4f}) detected when using GroupKFold!")
        logger.warning("  CONCLUSION: The previous perfect AUC (1.000) was largely due to 'Same-Original' Data Leakage.")
        logger.warning("  The model was memorizing the backgrounds of the images rather than learning generic tampering signatures.")
    else:
        logger.info("  No significant AUC drop. The model's performance is legitimate and robust to new images!")

    # --------------------------------------------------------------------------------
    # TEST 4: Generate Learning Curves
    # --------------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Generating Learning Curves (GroupKFold)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, groups=groups, cv=gkf, scoring='roc_auc', 
        n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training AUC")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation AUC (Leakage-Free)")
    plt.title("Learning Curves (GroupKFold)")
    plt.xlabel("Training Examples")
    plt.ylabel("AUC Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path = Path("output/q1_results/learning_curves.png")
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=300)
    logger.info(f"  Saved learning curves to {plot_path}")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE.")
    logger.info("Check the output above for leakage severity and the saved learning curves plot.")

if __name__ == "__main__":
    main()
