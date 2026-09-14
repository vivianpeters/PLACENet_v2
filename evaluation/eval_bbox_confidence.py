import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score

def count_valid_sources(boxes, confidences=None):
    """Count number of valid sources in each sample."""
    if confidences is not None:
        return np.sum(confidences >= 0.5, axis=-1)
    batch_size = boxes.shape[0]
    counts = []
    for b in range(batch_size):
        valid_mask = ~np.all(boxes[b] == 0.0, axis=-1)
        counts.append(np.sum(valid_mask))
    return np.array(counts)

def empty_slot_accuracy(true_boxes, pred_boxes):
    """Calculate accuracy of predicting empty slots."""
    batch_size = true_boxes.shape[0]
    empty_slot_correct = []
    for b in range(batch_size):
        true_empty_mask = np.all(true_boxes[b] == 0.0, axis=-1)
        if np.any(true_empty_mask):
            pred_empty_slots = pred_boxes[b][true_empty_mask]
            pred_is_pad = np.all(pred_empty_slots == 0.0, axis=-1)
            empty_slot_correct.append(np.mean(pred_is_pad))
    return np.mean(empty_slot_correct) if empty_slot_correct else np.nan

def compute_classification_metrics(y_true, y_pred, labels, strategy_name=""):
    """Calculate precision, recall, F1 (both macro and weighted), and confusion matrix for source counts."""
    y_pred_clipped = np.clip(y_pred, 0, max(labels))
    
    precision_weighted = precision_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
    
    precision_per_class = precision_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_clipped, labels=labels)
    
    return {
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
    }

def compute_slot_confidence_metrics(labels_test, pred_confidences_matched):
    """Binary classification metrics for confidence at the slot level."""
    if pred_confidences_matched is None:
        return {}
        
    true_confidences = np.zeros((labels_test.shape[0], labels_test.shape[1]), dtype=np.float32)
    for b in range(labels_test.shape[0]):
        for s in range(labels_test.shape[1]):
            is_valid = not np.all(labels_test[b, s] == 0.0)
            true_confidences[b, s] = 1.0 if is_valid else 0.0
            
    true_conf_flat = true_confidences.flatten()
    pred_conf_flat = pred_confidences_matched.flatten()
    pred_conf_binary = (pred_conf_flat >= 0.5).astype(int)
    true_conf_binary = true_conf_flat.astype(int)
    
    conf_cm = confusion_matrix(true_conf_binary, pred_conf_binary, labels=[0, 1])
    
    return {
        'confidence_accuracy': accuracy_score(true_conf_binary, pred_conf_binary),
        'confidence_precision': precision_score(true_conf_binary, pred_conf_binary, zero_division=0),
        'confidence_recall': recall_score(true_conf_binary, pred_conf_binary, zero_division=0),
        'confidence_f1': f1_score(true_conf_binary, pred_conf_binary, zero_division=0),
        'confidence_mean_pred': float(pred_conf_flat.mean()),
        'confidence_mean_true': float(true_conf_flat.mean()),
        'confidence_auc': roc_auc_score(true_conf_binary, pred_conf_flat) if len(np.unique(true_conf_binary)) > 1 else np.nan,
        'confidence_confusion_matrix': conf_cm.tolist(),
        'true_confidences_flat': true_conf_flat,
        'pred_confidences_flat': pred_conf_flat,
        'true_confidences': true_confidences,
    }
