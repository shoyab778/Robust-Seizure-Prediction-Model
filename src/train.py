import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.config import MODEL_DIR, LOG_DIR, EPOCHS, BATCH_SIZE, LR, INPUT_SHAPE, NUM_CLASSES, SEED
from src.utils import set_seed, to_categorical, compute_class_weights, save_history
from src.data_loader import load_predictive_files
from src.model import build_cnn_bilstm

set_seed(SEED)

def train():
    (Xtr,ytr),(Xv,yv),(Xt,yt) = load_predictive_files()
    print("Shapes -> train:", Xtr.shape, ytr.shape, "val:", Xv.shape, yv.shape, "test:", Xt.shape, yt.shape)

    ytr_cat = to_categorical(ytr, NUM_CLASSES)
    yv_cat = to_categorical(yv, NUM_CLASSES)

    class_weight = compute_class_weights(ytr)

    model = build_cnn_bilstm(input_shape=INPUT_SHAPE, n_classes=NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_fname = os.path.join(MODEL_DIR, "best_model.h5")
    ck = ModelCheckpoint(model_fname, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=1)

    history = model.fit(Xtr, ytr_cat,
                        validation_data=(Xv, yv_cat),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        class_weight=class_weight,
                        callbacks=[ck, es, rl], verbose=2)

    # Save training history
    hist_path = os.path.join(LOG_DIR, "history.json")
    save_history(history.history, hist_path)
    print("Training complete. Best model saved to:", model_fname)

if __name__ == "__main__":
    train()
