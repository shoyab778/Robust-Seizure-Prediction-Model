import os, json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def to_categorical(y, n_classes=2):
    return np.eye(n_classes)[y]

def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def save_history(history_dict, path):
    with open(path, "w") as f:
        json.dump(history_dict, f)
