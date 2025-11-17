import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from src.config import MODEL_DIR, REPLAY_PATH, REPLAY_SIZE, CL_EPOCHS, CL_LR
from src.utils import to_categorical

def load_replay():
    if os.path.exists(REPLAY_PATH):
        d = np.load(REPLAY_PATH)
        return d["X"], d["y"]
    return np.zeros((0,)), np.zeros((0,))

def save_replay(X, y):
    np.savez_compressed(REPLAY_PATH, X=X, y=y)

def continual_update(new_X, new_y, epochs=CL_EPOCHS, lr=CL_LR, replay_size=REPLAY_SIZE):
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("No trained model found in models/")
    model_path = os.path.join(MODEL_DIR, sorted(model_files)[0])
    model = load_model(model_path)

    bufX, bufy = load_replay()
    if bufX.size == 0:
        X_comb = new_X
        y_comb = to_categorical(new_y, 2)
    else:
        n = min(len(bufX), replay_size)
        idx = np.random.choice(len(bufX), n, replace=False)
        X_comb = np.concatenate([new_X, bufX[idx]], axis=0)
        y_comb = np.concatenate([to_categorical(new_y, 2), bufy[idx]], axis=0)

    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_comb, y_comb, epochs=epochs, batch_size=16, verbose=2)
    model.save(model_path)

    # update buffer
    keep = min(replay_size, len(new_X))
    newbufX = new_X[:keep]
    newbufY = to_categorical(new_y, 2)[:keep]
    if bufX.size != 0:
        rem = replay_size - keep
        if rem > 0:
            idx_old = np.random.choice(len(bufX), rem, replace=False)
            newbufX = np.concatenate([newbufX, bufX[idx_old]], axis=0)
            newbufY = np.concatenate([newbufY, bufy[idx_old]], axis=0)
    save_replay(newbufX, newbufY)
    print("Continual learning done. Buffer size:", len(newbufX))
