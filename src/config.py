import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset")
PRED_DIR = os.path.join(DATA_DIR, "predictive")
MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR = os.path.join(ROOT, "logs")
TEMP_DIR = os.path.join(ROOT, "tmp")

for d in (DATA_DIR, PRED_DIR, MODEL_DIR, LOG_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

# Data specs (Kaggle predictive dataset)
N_CHANNELS = 23
SAMPLES = 256  # timepoints (1-second windows)
INPUT_SHAPE = (SAMPLES, N_CHANNELS)  # (time, channels)

# Training
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4
SEED = 42
NUM_CLASSES = 2

# Continual learning buffer
REPLAY_PATH = os.path.join(MODEL_DIR, "replay_buffer.npz")
REPLAY_SIZE = 2000
CL_EPOCHS = 2
CL_LR = 1e-5
