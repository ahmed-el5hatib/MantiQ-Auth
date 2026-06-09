"""
MantiQ-Auth: Cross-Source Generalization Evaluation
Evaluates the model trained on LIDC-IDRI (CT) on a new dataset (e.g. COVID-19-AR X-ray).
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_extraction import extract_enriched_resnet_features, load_and_preprocess_image
from src.tampering_classifier import TamperingClassifier
from src.utils import setup_logging

logger = logging.getLogger("mantiq.eval_cross")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=str, default="data/tampered_dataset_xray/metadata.csv")
    parser.add_argument("--model", type=str, default="models/tampering_classifier.pkl")
    args = parser.parse_args()

    setup_logging("INFO")
    
    data_root = Path("data")
    tamper_csv = Path(args.test_csv)
    model_path = Path(args.model)

    if not tamper_csv.exists():
        logger.error(f"Cannot find metadata CSV at {tamper_csv}")
        sys.exit(1)
        
    if not model_path.exists():
        logger.error(f"Cannot find trained model at {model_path}")
        sys.exit(1)

    # 1. Load Pre-trained Model (Trained on CT)
    logger.info(f"Loading pre-trained classifier from {model_path}...")
    clf = TamperingClassifier.load(str(model_path))

    # 2. Load Data & Extract Features
    logger.info(f"Loading unseen cross-source dataset from {tamper_csv} (ResNet-18)...")
    features, labels = [], []
    
    with open(tamper_csv, "r") as f:
        rows = list(csv.DictReader(f))
        
    for i, row in enumerate(rows):
        fpath = data_root.parent / row["image_path"] # image_path might be absolute or relative
        if not fpath.exists():
            fpath = data_root / row["image_path"]
            if not fpath.exists(): continue
            
        try:
            tensor = load_and_preprocess_image(fpath)
            feat = extract_enriched_resnet_features(tensor)
            features.append(feat)
            labels.append(int(row["label"]))
            
        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

        if (i+1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(rows)}")

    X = np.array(features)
    y_true = np.array(labels)
    
    logger.info(f"Cross-source dataset loaded: {len(X)} samples ({np.sum(y_true==0)} original, {np.sum(y_true==1)} tampered).")

    # 3. Predict & Evaluate (NO RETRAINING)
    logger.info("Evaluating cross-source performance (Zero-Shot on new modality)...")
    
    y_pred = clf.pipeline.predict(X)
    y_proba = clf.pipeline.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y_true, y_proba)
    
    # Calculate TPR @ 0.1% FPR based on thresholds of THIS test set
    fprs, tprs, thresholds = roc_curve(y_true, y_proba)
    try:
        idx_001 = np.where(fprs <= 0.001)[0][-1]
        tpr_001 = tprs[idx_001]
    except IndexError:
        tpr_001 = 0.0
        
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info("=" * 60)
    logger.info("CROSS-SOURCE EVALUATION RESULTS (Trained: CT -> Tested: X-Ray)")
    logger.info("=" * 60)
    logger.info(f"AUC Score: {auc:.4f}")
    logger.info(f"TPR @ 0.1% FPR: {tpr_001:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"[[TN, FP]\n [FN, TP]]\n{cm}")
    logger.info("=" * 60)

    # Bootstrapped 95% CI for AUC
    n_bootstraps = 1000
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_proba), len(y_proba))
        if len(np.unique(y_true[indices])) < 2: continue
        score = roc_auc_score(y_true[indices], y_proba[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    conf_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    conf_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    logger.info(f"AUC 95% CI: {conf_lower:.4f} - {conf_upper:.4f}")

if __name__ == "__main__":
    main()
