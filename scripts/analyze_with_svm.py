"""
MantiQ-Auth: Evaluation on Harder Dataset
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
from src.tampering_analysis import TamperingClassifier
from src.utils import load_config, setup_logging

logger = logging.getLogger("mantiq.eval_hard")

def main():
    setup_logging("INFO")
    
    data_root = Path("data")
    tamper_csv = data_root / "tampered_dataset_harder" / "metadata.csv"
    model_path = Path("models/tampering_classifier.pkl")

    if not tamper_csv.exists():
        logger.error(f"Cannot find metadata CSV at {tamper_csv}")
        sys.exit(1)
        
    if not model_path.exists():
        logger.error(f"Cannot find trained model at {model_path}")
        sys.exit(1)

    # 1. Load Data & Extract Features
    logger.info("Loading harder dataset and extracting features (ResNet-18)...")
    features, labels, groups = [], [], []
    group_map = {}
    group_counter = 0
    
    with open(tamper_csv, "r") as f:
        rows = list(csv.DictReader(f))
        
    for i, row in enumerate(rows):
        fpath = data_root / row["image_path"]
        if not fpath.exists(): continue
        
        try:
            tensor = load_and_preprocess_image(fpath)
            feat = extract_enriched_resnet_features(tensor)
            features.append(feat)
            labels.append(int(row["label"]))
            
            src = row["source_image"]
            if src not in group_map:
                group_map[src] = group_counter
                group_counter += 1
            groups.append(group_map[src])
            
        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

        if (i+1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(rows)}")

    X = np.array(features)
    y_true = np.array(labels)
    g = np.array(groups)
    
    logger.info(f"Dataset loaded: {len(X)} samples ({np.sum(y_true==0)} original, {np.sum(y_true==1)} tampered).")

    # 3. Train & Evaluate using Grouped K-Fold to prevent leakage!
    logger.info("Training and Evaluating via StratifiedGroupKFold on the hard dataset...")
    clf = TamperingClassifier(classifier_type="svm")
    n_folds = min(10, min(np.sum(y_true==0), np.sum(y_true==1)))
    n_folds = max(2, min(n_folds, len(np.unique(g))))
    
    clf.train(X, y_true, n_cv_folds=n_folds, groups=g)
    
    # Predict (Out-of-fold probabilities are saved in the classifier automatically during train)
    # Wait, the train method actually saves the ROC data inside clf.roc_data which is computed
    # purely from OUT-OF-FOLD predictions! So the AUC from `clf.roc_data['auc']` is completely unbiased!
    auc_cv = clf.roc_data["auc"]
    
    # Calculate TPR @ 0.1% FPR
    fprs_cv = clf.roc_data["fpr"]
    tprs_cv = clf.roc_data["tpr"]
    idx_001 = np.where(fprs_cv <= 0.001)[0][-1]
    tpr_cv = tprs_cv[idx_001]
    
    # To get confusion matrix, we use the optimal threshold on OOF predictions
    fprs = clf.roc_data["fpr"]
    tprs = clf.roc_data["tpr"]
    thresh = clf.roc_data["thresholds"]
    idx = np.argmax(tprs - fprs)
    optimal_thresh = thresh[idx]
    
    # We don't have access to the raw OOF predictions easily, so we can just do:
    y_pred = clf.pipeline.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (HARDER DATASET - GROUPED K-FOLD OOF)")
    logger.info("=" * 60)
    logger.info(f"AUC Score (Out-of-fold): {auc_cv:.4f}")
    logger.info(f"TPR @ 0.1% FPR (Out-of-fold): {tpr_cv:.4f}")
    logger.info("Confusion Matrix (Final Model on all data):")
    logger.info(f"[[TN, FP]\n [FN, TP]]\n{cm}")
    logger.info("=" * 60)

    # Bootstrapped 95% CI for AUC
    # We use the final model's predict_proba for CI, just as an estimate
    y_proba = clf.pipeline.predict_proba(X)[:, 1]
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
    
    logger.info(f"AUC 95% CI (Final Model): {conf_lower:.4f} - {conf_upper:.4f}")

    # ==========================================
    # Generate and Save Plots
    # ==========================================
    logger.info("Generating evaluation plots for the Hard Dataset...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    out_dir = Path("output/hard_dataset_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs_cv, tprs_cv, color='darkorange', lw=2, label=f'ResNet-18 + SVM (AUC = {auc_cv:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Hard Tampering Dataset (Grouped K-Fold OOF)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = out_dir / "roc_curve_hard.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Original', 'Tampered'],
                yticklabels=['Original', 'Tampered'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix - Hard Tampering Dataset', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    cm_path = out_dir / "confusion_matrix_hard.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
