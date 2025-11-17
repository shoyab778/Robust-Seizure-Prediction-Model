import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import PRED_DIR, INPUT_SHAPE

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    if "X" in data and "y" in data:
        X, y = data["X"], data["y"]
    else:
        keys = list(data.keys())
        if len(keys) >= 2:
            X, y = data[keys[0]], data[keys[1]]
        else:
            raise ValueError("NPZ format not recognized: expected X and y arrays")
    # Ensure shape (n, time, channels)
    if X.ndim == 3 and X.shape[1] == INPUT_SHAPE[0] and X.shape[2] == INPUT_SHAPE[1]:
        # already (n, time, channels)
        pass
    elif X.ndim == 3 and X.shape[1] == INPUT_SHAPE[1] and X.shape[2] == INPUT_SHAPE[0]:
        # (n, channels, time) -> transpose
        X = X.transpose(0,2,1)
    return X.astype("float32"), y.astype("int32")

def load_predictive_files(pred_dir=PRED_DIR):
    files = os.listdir(pred_dir)
    train = next((f for f in files if "predictive" in f and "train" in f), None)
    val = next((f for f in files if "predictive" in f and "val" in f and "balanced" not in f), None)
    test = next((f for f in files if "predictive" in f and "test" in f), None)

    if train:
        Xtr, ytr = load_npz(os.path.join(pred_dir, train))
        if val:
            Xv, yv = load_npz(os.path.join(pred_dir, val))
        else:
            Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.1, random_state=42, stratify=ytr)
        if test:
            Xt, yt = load_npz(os.path.join(pred_dir, test))
        else:
            Xv, Xt, yv, yt = train_test_split(Xv, yv, test_size=0.5, random_state=42, stratify=yv)
        return (Xtr, ytr), (Xv, yv), (Xt, yt)
    else:
        # fallback: pick first npz
        npzs = [os.path.join(pred_dir, f) for f in files if f.endswith(".npz")]
        if not npzs:
            raise FileNotFoundError("No .npz found in predictive dir")
        X, y = load_npz(npzs[0])
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xv, Xt, yv, yt = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp)
        return (Xtr, ytr), (Xv, yv), (Xt, yt)
