# evaluation.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

def evaluate_model(model, X_train, y_train, X_test, y_test, recall_target=0.8, use_smote=True):
    """
    Evaluate a classification model with optional SMOTE balancing.

    Parameters
    ----------
    model : sklearn classifier
        The model to train and evaluate.
    X_train, y_train : training data
    X_test, y_test : test data
    recall_target : float
        Desired recall level for threshold selection.
    use_smote : bool
        If True, apply SMOTE oversampling to training data.
        If False, rely on class_weight or raw imbalance.

    Returns
    -------
    dict : results with AUC, precision, recall
    """

    # Optionally apply SMOTE
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Fit model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # ROC AUC
    auc = roc_auc_score(y_test, y_probs)

    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    # Find threshold closest to recall_target
    idx = np.argmin(np.abs(recalls - recall_target))
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    # Apply threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics
    report = classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"])
    cm = confusion_matrix(y_test, y_pred)

    print("="*70)
    print(f"Model: {model.__class__.__name__}")
    print(f"AUC: {auc:.4f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    return {"auc": auc, "precision": precisions[idx], "recall": recalls[idx]}
