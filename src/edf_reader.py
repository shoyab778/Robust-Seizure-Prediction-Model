import mne
import numpy as np
from scipy.signal import resample
from src.config import N_CHANNELS, SAMPLES

def read_edf_file(path, target_fs=256):
    """
    Read EDF and convert to sliding windows of shape (n_windows, time, channels).
    If channels < N_CHANNELS, pad with zeros.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    data = raw.get_data()  # shape (n_channels, n_samples)
    fs = int(raw.info.get("sfreq", target_fs))

    if data.shape[0] < N_CHANNELS:
        pad = N_CHANNELS - data.shape[0]
        data = np.vstack([data, np.zeros((pad, data.shape[1]))])

    if fs != target_fs:
        ns = int(data.shape[1] * target_fs / fs)
        data = resample(data, ns, axis=1)
        fs = target_fs

    win_len = SAMPLES
    stride = SAMPLES  # non-overlapping 1s windows
    windows = []
    for st in range(0, data.shape[1] - win_len + 1, stride):
        w = data[:, st:st+win_len]  # (channels, time)
        windows.append(w.T)  # (time, channels)
    if len(windows) == 0:
        return np.zeros((0, win_len, N_CHANNELS), dtype="float32")
    return np.stack(windows, axis=0).astype("float32")
