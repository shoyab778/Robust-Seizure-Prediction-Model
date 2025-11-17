import os, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model
from src.data_loader import load_predictive_files
from src.utils import to_categorical
from src.config import MODEL_DIR

def compute_fp_per_hour(y_true, y_pred, stride_sec=1, window_sec=1):
    inter_idx = np.where(y_true == 0)[0]
    if len(inter_idx) == 0:
        return float('nan')
    false_alarms = np.sum((y_pred[inter_idx] == 1))
    hours = len(inter_idx) * stride_sec / 3600.0
    return false_alarms / max(1e-6, hours)

def evaluate():
    (Xtr, ytr), (Xv, yv), (Xt, yt) = load_predictive_files()
    model_path = os.path.join(MODEL_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found at " + model_path)
    model = load_model(model_path)
    preds = model.predict(Xt, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(yt, y_pred)
    prec = precision_score(yt, y_pred, zero_division=0)
    rec = recall_score(yt, y_pred, zero_division=0)
    f1 = f1_score(yt, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(to_categorical(yt, 2), preds)
    except:
        auc = float('nan')
    cm = confusion_matrix(yt, y_pred)
    fp_per_h = compute_fp_per_hour(yt, y_pred)
    print("acc", acc, "prec", prec, "rec", rec, "f1", f1, "auc", auc, "fp/hr", fp_per_h)
    print("confusion\n", cm)

if __name__ == "__main__":
    evaluate()
