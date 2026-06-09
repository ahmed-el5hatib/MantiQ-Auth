"""
MantiQ-Auth: Tampering Detection Classifier (Improvement 2)

Implements a machine-learning-based tampering classifier to raise AUC from 0.74 to >0.90:
  - SVM with RBF kernel (primary) and optional MLP fallback
  - 10-fold cross-validation with stratified splits
  - ROC curve, precision-recall curve, optimal threshold (Youden's index)
  - Adaptive threshold mechanism with configurable FPR target (<0.001)
  - Trained on enriched feature vectors (ViT CLS + variance + entropy + L2 norm)

Usage:
    from src.tampering_classifier import TamperingClassifier
    clf = TamperingClassifier()
    clf.train(features, labels)
    prediction = clf.predict(feature_vector)
    probability = clf.predict_proba(feature_vector)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger("mantiq.classifier")


# ─── Feature Enrichment ─────────────────────────────────────────────────────

def compute_enriched_features(
    cls_features: np.ndarray,
    patch_features: Optional[np.ndarray] = None,
    attention_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Enrich the base CLS token features with additional discriminative signals.

    Appends:
        1. Feature variance across the vector (scalar)
        2. Feature entropy (Shannon entropy of softmaxed features)
        3. L2 norm of the raw feature vector (before normalization)

    This enrichment raises the feature dimension from 384 → 387 for ViT-S.

    Args:
        cls_features: Base CLS token features (384-dim for ViT-S).
        patch_features: Optional patch-level features for variance computation.
        attention_map: Optional attention weights for entropy computation.

    Returns:
        Enriched feature vector of shape (387,) or (original_dim + 3,).
    """
    base_dim = len(cls_features)

    # 1. Feature variance across dimensions
    feat_variance = float(np.var(cls_features))

    # 2. Feature entropy (Shannon entropy of probability-like distribution)
    # Convert features to probability distribution via softmax
    feat_shifted = cls_features - np.max(cls_features)  # Numerical stability
    exp_feat = np.exp(feat_shifted)
    probs = exp_feat / (np.sum(exp_feat) + 1e-10)
    # Shannon entropy
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    # 3. L2 norm of the feature vector
    l2_norm = float(np.linalg.norm(cls_features))

    # If attention map is provided, use it for more informative entropy
    if attention_map is not None:
        attn_probs = attention_map / (np.sum(attention_map) + 1e-10)
        entropy = -float(np.sum(attn_probs * np.log(attn_probs + 1e-10)))

    # If patch features are provided, compute inter-patch variance
    if patch_features is not None and len(patch_features) > 1:
        feat_variance = float(np.mean(np.var(patch_features, axis=0)))

    enriched = np.concatenate([
        cls_features,
        np.array([feat_variance, entropy, l2_norm]),
    ])

    return enriched


def extract_enriched_features_from_image(
    image_path: Union[str, Path],
    extractor: str = "vit",
) -> np.ndarray:
    """
    Full pipeline: load image → extract ViT features → enrich with extra signals.

    Args:
        image_path: Path to the image file.
        extractor: Feature extractor ("vit" or "resnet").

    Returns:
        Enriched feature vector.
    """
    from src.feature_extraction import (
        extract_features,
        load_and_preprocess_image,
        load_and_preprocess_dicom,
    )

    image_path = Path(image_path)
    if image_path.suffix.lower() == ".dcm":
        tensor = load_and_preprocess_dicom(image_path)
    else:
        tensor = load_and_preprocess_image(image_path)

    features = extract_features(tensor, extractor=extractor)
    return compute_enriched_features(features)


# ─── Tampering Classifier ───────────────────────────────────────────────────

class TamperingClassifier:
    """
    Binary classifier for medical image tampering detection.

    Supports SVM (RBF kernel) and MLP architectures with automatic
    threshold optimization for clinical FPR requirements.
    """

    def __init__(
        self,
        classifier_type: str = "svm",
        fpr_target: float = 0.001,
        random_state: int = 42,
    ):
        """
        Args:
            classifier_type: "svm" for SVM-RBF or "mlp" for MLP.
            fpr_target: Target false positive rate for threshold selection.
            random_state: Random seed for reproducibility.
        """
        self.classifier_type = classifier_type
        self.fpr_target = fpr_target
        self.random_state = random_state

        # Build the sklearn pipeline with scaling
        if classifier_type == "svm":
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                )),
            ])
        elif classifier_type == "mlp":
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=random_state,
                )),
            ])
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.optimal_threshold: float = 0.5
        self.roc_data: Optional[Dict] = None
        self.pr_data: Optional[Dict] = None
        self.cv_results: Optional[Dict] = None
        self._is_trained: bool = False

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_cv_folds: int = 10,
        groups: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Train the classifier with stratified k-fold cross-validation.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Binary labels (0=original, 1=tampered).
            n_cv_folds: Number of cross-validation folds (default 10).
            groups: Optional group labels for GroupKFold to prevent leakage.

        Returns:
            Dictionary with cross-validation results, ROC data, and metrics.
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        n_samples, n_features = features.shape
        logger.info(
            "Training %s classifier: %d samples × %d features, %d folds",
            self.classifier_type.upper(), n_samples, n_features, n_cv_folds,
        )

        # ── 10-fold cross-validation ──────────────────────────
        if groups is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            skf = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=self.random_state)
            splitter = skf.split(features, labels, groups=groups)
        else:
            skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=self.random_state)
            splitter = skf.split(features, labels)

        # Collect out-of-fold predictions for ROC/PR curves
        oof_probas = np.zeros(n_samples)
        oof_preds = np.zeros(n_samples, dtype=int)
        fold_aucs = []
        fold_accuracies = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            self.pipeline.fit(X_train, y_train)

            val_proba = self.pipeline.predict_proba(X_val)[:, 1]
            val_pred = self.pipeline.predict(X_val)

            oof_probas[val_idx] = val_proba
            oof_preds[val_idx] = val_pred

            fold_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5
            fold_acc = accuracy_score(y_val, val_pred)
            fold_aucs.append(fold_auc)
            fold_accuracies.append(fold_acc)

            logger.debug(
                "  Fold %d/%d: AUC=%.4f, Accuracy=%.4f",
                fold_idx + 1, n_cv_folds, fold_auc, fold_acc,
            )

        # ── Final model: train on ALL data ───────────────────────────────
        self.pipeline.fit(features, labels)
        self._is_trained = True

        # ── Compute ROC curve from out-of-fold predictions ───────────────
        fpr, tpr, roc_thresholds = roc_curve(labels, oof_probas)
        overall_auc = auc(fpr, tpr)

        self.roc_data = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
            "auc": overall_auc,
        }

        # ── Precision-Recall curve ───────────────────────────────────────
        precision, recall, pr_thresholds = precision_recall_curve(labels, oof_probas)
        pr_auc = auc(recall, precision)

        self.pr_data = {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
            "auc": pr_auc,
        }

        # ── Optimal threshold via Youden's index ────────────────────────
        youden_index = tpr - fpr
        best_youden_idx = np.argmax(youden_index)
        youden_threshold = float(roc_thresholds[best_youden_idx])

        # ── Clinical threshold: maximize TPR at FPR < target ────────────
        clinical_threshold = self._find_clinical_threshold(fpr, tpr, roc_thresholds)

        self.optimal_threshold = youden_threshold

        # ── Confusion matrix at optimal threshold ────────────────────────
        optimal_preds = (oof_probas >= youden_threshold).astype(int)
        cm = confusion_matrix(labels, optimal_preds)

        # ── TPR at specified FPR ─────────────────────────────────────────
        tpr_at_fpr_target = self._get_tpr_at_fpr(fpr, tpr, self.fpr_target)

        # ── Compile results ──────────────────────────────────────────────
        self.cv_results = {
            "cv_auc_mean": float(np.mean(fold_aucs)),
            "cv_auc_std": float(np.std(fold_aucs)),
            "cv_auc_per_fold": fold_aucs,
            "cv_accuracy_mean": float(np.mean(fold_accuracies)),
            "cv_accuracy_std": float(np.std(fold_accuracies)),
            "overall_auc": overall_auc,
            "pr_auc": pr_auc,
            "youden_threshold": youden_threshold,
            "clinical_threshold": clinical_threshold,
            "tpr_at_target_fpr": tpr_at_fpr_target,
            "target_fpr": self.fpr_target,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                labels, optimal_preds, target_names=["Original", "Tampered"],
            ),
        }

        logger.info("Training complete!")
        logger.info("  Cross-validated AUC: %.4f ± %.4f", np.mean(fold_aucs), np.std(fold_aucs))
        logger.info("  Overall AUC (OOF): %.4f", overall_auc)
        logger.info("  Youden's optimal threshold: %.4f", youden_threshold)
        logger.info("  Clinical threshold (FPR<%.3f): %.4f", self.fpr_target, clinical_threshold)
        logger.info("  TPR @ FPR=%.3f: %.4f", self.fpr_target, tpr_at_fpr_target)

        return self.cv_results

    def predict(self, features: np.ndarray, use_clinical_threshold: bool = False) -> np.ndarray:
        """Predict binary labels using the trained classifier."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        probas = self.predict_proba(features)
        threshold = (
            self.cv_results["clinical_threshold"]
            if use_clinical_threshold
            else self.optimal_threshold
        )
        return (probas >= threshold).astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of tampering (class 1)."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        return self.pipeline.predict_proba(features)[:, 1]

    def save(self, filepath: Union[str, Path] = "models/tampering_classifier.pkl") -> None:
        """Save the trained classifier to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "pipeline": self.pipeline,
            "optimal_threshold": self.optimal_threshold,
            "cv_results": self.cv_results,
            "roc_data": {
                "fpr": self.roc_data["fpr"].tolist() if self.roc_data else [],
                "tpr": self.roc_data["tpr"].tolist() if self.roc_data else [],
                "auc": self.roc_data["auc"] if self.roc_data else 0.0,
            },
            "classifier_type": self.classifier_type,
            "fpr_target": self.fpr_target,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info("Saved classifier to %s", filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/tampering_classifier.pkl") -> "TamperingClassifier":
        """Load a trained classifier from disk."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        obj = cls(
            classifier_type=save_data["classifier_type"],
            fpr_target=save_data.get("fpr_target", 0.001),
        )
        obj.pipeline = save_data["pipeline"]
        obj.optimal_threshold = save_data["optimal_threshold"]
        obj.cv_results = save_data["cv_results"]
        obj._is_trained = True

        # Reconstruct numpy arrays for roc_data
        roc = save_data.get("roc_data", {})
        if roc:
            obj.roc_data = {
                "fpr": np.array(roc["fpr"]),
                "tpr": np.array(roc["tpr"]),
                "auc": roc["auc"],
            }

        logger.info("Loaded classifier from %s", filepath)
        return obj

    # ── Internal helpers ─────────────────────────────────────────────────

    def _find_clinical_threshold(
        self, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
    ) -> float:
        """Find threshold that maximizes TPR while keeping FPR < target."""
        # Find indices where FPR is below the target
        valid_idx = np.where(fpr <= self.fpr_target)[0]
        if len(valid_idx) == 0:
            logger.warning(
                "No threshold achieves FPR < %.4f. Using Youden's threshold.",
                self.fpr_target,
            )
            youden_idx = np.argmax(tpr - fpr)
            return float(thresholds[youden_idx])

        # Among valid thresholds, pick the one with the highest TPR
        best_idx = valid_idx[np.argmax(tpr[valid_idx])]
        return float(thresholds[min(best_idx, len(thresholds) - 1)])

    @staticmethod
    def _get_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
        """Interpolate TPR at a specific FPR value."""
        if len(fpr) < 2:
            return 0.0
        return float(np.interp(target_fpr, fpr, tpr))


# ─── Convenience Function ───────────────────────────────────────────────────

def train_tampering_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str = "models/tampering_classifier.pkl",
    classifier_type: str = "svm",
    n_cv_folds: int = 10,
) -> Tuple[TamperingClassifier, Dict]:
    """
    Train and save a tampering detection classifier.

    Args:
        features: Feature matrix (n_samples × n_features).
        labels: Binary labels (0=original, 1=tampered).
        save_path: Path to save the trained model.
        classifier_type: "svm" or "mlp".
        n_cv_folds: Cross-validation folds.

    Returns:
        Tuple of (trained_classifier, results_dict).
    """
    clf = TamperingClassifier(classifier_type=classifier_type)
    results = clf.train(features, labels, n_cv_folds=n_cv_folds)
    clf.save(save_path)
    return clf, results
