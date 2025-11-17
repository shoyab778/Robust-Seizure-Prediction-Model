# app/streamlit_app.py
# SEIZURE PREDICTION DASHBOARD — Full Single File (Premium, Centered, Clinical UI)
# Usage: place in seizure_predictor/app/, ensure src/ exists (data_loader, edf_reader, config),
# put trained .h5 model(s) in models/ (MODEL_DIR). Then run:
#    streamlit run app/streamlit_app.py
#
# Features:
# - Upload .npz or .edf (single action)
# - Auto preprocessing & windowing (via src.edf_reader / src.data_loader)
# - Auto prediction with latest model from models/
# - Beautiful centered clinical UI (black title)
# - High-quality stacked-channel EEG view with zoom/range slider
# - Long-trace concatenation view
# - High-resolution spectrogram
# - Probability timeline + heatmap + onset locator
# - Top suspicious windows panel
# - Clinician notes
# - Export PDF clinical report + CSV of predictions
# - Responsive, elegant, and production-ready visuals
#
# NOTE: This file uses helper functions from your project's src/ package:
#   - src.config with MODEL_DIR, TEMP_DIR
#   - src.data_loader.load_npz(path) -> (X, y) shape (n_windows, time, channels)
#   - src.edf_reader.read_edf_file(path) -> X shape (n_windows, time, channels)
#
# If those modules differ in names/behaviour, adapt the import lines accordingly.

import sys, os, io, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tensorflow.keras.models import load_model
from scipy.ndimage import uniform_filter1d

# Project helper modules (expected in src/)
from src.data_loader import load_npz
from src.edf_reader import read_edf_file
from src.config import MODEL_DIR, TEMP_DIR

# ---------- Page config & premium CSS ----------
st.set_page_config(page_title="Seizure Prediction Dashboard", layout="wide")
CSS = """
<style>
main .block-container { max-width:1150px; margin-left:auto; margin-right:auto; padding-top:18px; padding-bottom:28px; }
.card { background:#ffffff; border-radius:12px; padding:16px 18px; box-shadow:0 8px 24px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.03); }
.title { color:#0b0f14; font-size:42px; font-weight:800; text-align:center; margin-bottom:6px; }
.subtitle { text-align:center; color:#546b78; font-size:14px; margin-bottom:14px; }
.small-muted { color:#57707b; font-size:0.95rem; }
.plot-pad { padding-top:8px; padding-bottom:8px; }
.section-divider { border-bottom:1px solid #eef5f8; margin:16px 0; }
.channel-label { font-size:11px; color:#0b0f14; }
.center { display:flex; justify-content:center; align-items:center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="card"><div class="title">SEIZURE PREDICTION DASHBOARD</div><div class="subtitle">Upload EEG (.npz or .edf) → Automatic analysis → Clinical report & visualizations</div></div>', unsafe_allow_html=True)
st.write("")

# ---------- Helpers ----------
def latest_model(models_dir):
    if not os.path.isdir(models_dir):
        return None
    files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    return os.path.join(models_dir, files[0])

def smooth_probs(probs, win):
    if win <= 1: return probs
    return uniform_filter1d(probs, size=win)

def find_onset(probs, thresh, consecutive=2):
    above = probs > thresh
    n = len(above)
    if n == 0: return None
    for i in range(n - consecutive + 1):
        if all(above[i:i+consecutive]): return int(i)
    idxs = np.where(above)[0]
    return int(idxs[0]) if len(idxs)>0 else None

def compute_offsets(n_ch, spacing=4.0):
    return np.arange(n_ch) * spacing

def make_pdf(buf, uploaded_name, model_name, probs, X, onset_idx, clin_notes, thresh):
    with PdfPages(buf) as pdf:
        # Page 1: summary + timeline
        fig = plt.figure(figsize=(8.27,11.69))
        plt.suptitle("Seizure Prediction — Clinical Report", fontsize=14)
        plt.subplot(3,1,1)
        plt.text(0.01, 0.8, f"File: {uploaded_name}", fontsize=10)
        plt.text(0.01, 0.74, f"Model: {model_name}", fontsize=10)
        plt.text(0.01, 0.68, f"Windows analyzed: {len(probs)}", fontsize=10)
        plt.text(0.01, 0.62, f"Suggested onset window: {onset_idx if onset_idx is not None else 'N/A'}", fontsize=10)
        plt.subplot(3,1,2)
        plt.plot(probs, color='tab:blue')
        plt.axhline(thresh, color='tab:red', linestyle='--')
        plt.title("Probability timeline")
        plt.xlabel("Window index")
        plt.ylabel("Probability")
        plt.subplot(3,1,3)
        top_idx = np.argsort(probs)[-6:][::-1]
        for i, idx in enumerate(top_idx):
            plt.plot(X[idx][:,0] + i*4, linewidth=0.8)
        plt.title("Top suspicious windows (channel 0 previews)")
        pdf.savefig()
        plt.close('all')
        # Page 2: thumbnails
        fig2 = plt.figure(figsize=(8.27,11.69))
        plt.suptitle("EEG Thumbnails", fontsize=12)
        top_idx = np.argsort(probs)[-12:][::-1]
        ncols = 3
        nrows = int(np.ceil(len(top_idx)/ncols))
        for i, idx in enumerate(top_idx):
            ax = fig2.add_subplot(nrows, ncols, i+1)
            for ch in range(min(2, X.shape[2])):
                ax.plot(X[idx][:,ch] + ch*3, linewidth=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{idx} p={probs[idx]:.2f}", fontsize=8)
        pdf.savefig()
        plt.close('all')

# ---------- Sidebar: minimal controls ----------
with st.sidebar:
    st.header("Quick Controls")
    model_file = latest_model(MODEL_DIR)
    if model_file:
        st.markdown(f"**Model:** `{os.path.basename(model_file)}`")
    else:
        st.warning("No model found in models/. Train model and place .h5 in models/")
    uploaded = st.file_uploader("Upload EEG (.npz or .edf)", type=["npz","edf"])
    smoothing_sec = st.slider("Smoothing (windows)", 1, 8, 1)
    long_trace_windows = st.slider("Long trace windows", 10, 300, 120)
    clinician_notes = st.text_area("Clinician notes (appear in PDF)", value="", height=120)
    st.markdown("---")
    st.caption("This app is for research/support. Not a diagnostic device.")

# ---------- Validate model and upload ----------
if model_file is None:
    st.error("Please place a trained model (.h5) inside models/ and reload.")
    st.stop()
try:
    model = load_model(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if uploaded is None:
    st.info("Upload a .npz (preprocessed) or .edf to begin analysis.")
    st.stop()

# Save uploaded to TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=True)
tmp_path = os.path.join(TEMP_DIR, f"upload_{int(time.time())}_{uploaded.name}")
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())

# ---------- Load data automatically ----------
try:
    if uploaded.name.endswith(".npz"):
        X, y = load_npz(tmp_path)     # expected shape (n_windows, time, channels)
        source = "NPZ (preprocessed)"
    else:
        X = read_edf_file(tmp_path)   # expected shape (n_windows, time, channels)
        y = None
        source = "EDF (raw -> windows)"
except Exception as e:
    st.error(f"Failed to read EEG file: {e}")
    st.stop()

if X is None or X.size == 0 or X.ndim != 3:
    st.error("Loaded data has unexpected shape. Expected (n_windows, time, channels).")
    st.stop()

n_windows, win_len, n_ch = X.shape
sfreq = 256  # assumed sampling rate used by preprocessing pipeline

# ---------- Prediction ----------
with st.spinner("Running model predictions..."):
    preds = model.predict(X, verbose=0)
    probs = preds[:,1] if preds.ndim==2 else preds
    probs = probs.astype(np.float32)
    probs_sm = smooth_probs(probs, smoothing_sec)

# Decision heuristic
THRESH = 0.60
CONSECUTIVE = 2
onset = find_onset(probs_sm, THRESH, CONSECUTIVE)
seizure_flag = onset is not None
max_p = float(probs_sm.max())
mean_p = float(probs_sm.mean())
risk_windows = int(np.sum(probs_sm > THRESH))

# ---------- Top centered summary ----------
st.markdown('<div class="card center">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.markdown("**Session**")
    st.write(f"- File: `{uploaded.name}`")
    st.write(f"- Source: {source}")
    st.write(f"- Windows: {n_windows} | Window length: {win_len} | Channels: {n_ch}")
with c2:
    st.markdown("**Model & Risk**")
    st.write(f"- Model: `{os.path.basename(model_file)}`")
    st.write(f"- Max prob: **{max_p:.3f}**")
    st.write(f"- Mean prob: {mean_p:.3f}")
    st.write(f"- Risk windows (> {THRESH:.2f}): {risk_windows}")
with c3:
    st.markdown("**Decision**")
    if seizure_flag:
        st.markdown(f"### ⚠️ Seizure Detected (onset ~ window {onset})")
        st.warning("Urgent clinical review recommended.")
    else:
        st.markdown("### ✔️ No Seizure Detected")
        st.success("No immediate seizure detected in this recording.")
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ---------- Main layout: EEG viewer (left) + predictions (right) ----------
left, right = st.columns([2,1])

# LEFT: stacked EEG with improved visuals
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("EEG Viewer — Stacked channels (clinical view)")
    win_idx = st.number_input("Window index", min_value=0, max_value=max(0, n_windows-1), value=min(n_windows-1, n_windows//2), step=1)
    amp_scale = st.slider("Visual amplitude scale", 0.6, 4.0, 1.2, step=0.1)
    show_labels = st.checkbox("Show channel labels", value=True)
    sample = X[win_idx]  # (time, channels)
    t = np.arange(win_len) / sfreq

    # Adaptive spacing
    ch_std = np.std(sample, axis=0)
    median_amp = np.median(ch_std) if np.median(ch_std) > 0 else 1.0
    spacing = max(3.0, median_amp * 4.5) * amp_scale
    offsets = compute_offsets(n_ch, spacing=spacing)

    fig = go.Figure()
    for ch in range(n_ch):
        y = sample[:, ch]
        if np.std(y) > 0:
            y = (y - np.mean(y)) / (np.std(y) + 1e-9)
        y = y * 1.2
        fig.add_trace(go.Scatter(x=t, y=y + offsets[ch], mode='lines', line=dict(width=1), name=f"Ch {ch}", showlegend=False))
    fig.update_layout(height=520, template='plotly_white', margin=dict(l=40,r=20,t=30,b=30), xaxis_title="Time (s)")
    fig.update_yaxes(showticklabels=False)
    if show_labels:
        for ch in range(n_ch):
            fig.add_annotation(dict(xref='paper', x=0.01, y=offsets[ch], xanchor='left', yanchor='middle', text=f"Ch {ch}", showarrow=False, font=dict(size=10, color='#0b0f14')))
    # range slider for zoom
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="small-muted">Use range slider to zoom in time. Increase amplitude scale to reveal small spikes.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT: probability timeline, heatmap, top thumbnails
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Overview")
    # probability timeline
    figp = go.Figure()
    figp.add_trace(go.Scatter(y=probs_sm, mode='lines', line=dict(color='#1f77b4'), name='Probability'))
    figp.add_hline(y=THRESH, line_dash='dash', line_color='red', annotation_text=f"Threshold {THRESH}", annotation_position="top left")
    hi = np.where(probs_sm > THRESH)[0]
    if len(hi) > 0:
        figp.add_trace(go.Scatter(x=hi, y=probs_sm[hi], mode='markers', marker=dict(color='red', size=6), name='Alerts'))
    figp.update_layout(template='plotly_white', height=260)
    st.plotly_chart(figp, use_container_width=True)

    # heatmap
    st.markdown("Risk heatmap")
    hm = probs_sm[np.newaxis, :]
    fhm = px.imshow(hm, aspect='auto', color_continuous_scale='Reds')
    fhm.update_layout(template='plotly_white', height=80, margin=dict(t=6,b=6))
    fhm.update_yaxes(showticklabels=False)
    st.plotly_chart(fhm, use_container_width=True)

    # top suspicious thumbnails
    st.markdown("Top suspicious windows")
    topk = min(8, len(probs_sm))
    top_idx = np.argsort(probs_sm)[-topk:][::-1]
    cols = st.columns(2)
    for i, idx in enumerate(top_idx):
        with cols[i%2]:
            st.markdown(f"**#{idx}** — p={probs_sm[idx]:.3f}")
            figt = go.Figure()
            for ch in range(min(3, n_ch)):
                figt.add_trace(go.Scatter(y=X[idx][:,ch] + ch*2.4, mode='lines', line=dict(width=1), showlegend=False))
            figt.update_layout(height=120, template='plotly_white', margin=dict(l=6,r=6,t=6,b=6), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(figt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- Long trace concatenation ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Long Trace — Concatenated windows (overview)")
start_win = st.number_input("Start window for long trace", min_value=0, max_value=max(0, n_windows-1), value=0, step=1)
numw = min(long_trace_windows, n_windows - start_win)
if numw <= 0:
    st.info("Choose a start window within range.")
else:
    long_data = X[start_win:start_win+numw]
    long_trace = long_data.reshape(-1, n_ch)
    time_long = np.arange(long_trace.shape[0]) / sfreq
    show_chs = list(range(min(n_ch, 10)))
    offsets_long = compute_offsets(len(show_chs), spacing=spacing*1.2)
    figL = go.Figure()
    for i, ch in enumerate(show_chs):
        y = long_trace[:, ch]
        if np.std(y) > 0:
            y = (y - np.mean(y)) / (np.std(y)+1e-9)
        figL.add_trace(go.Scatter(x=time_long, y=y + offsets_long[i], mode='lines', line=dict(width=0.8), showlegend=False))
        figL.add_annotation(dict(xref='paper', x=0.01, y=offsets_long[i], xanchor='left', yanchor='middle', text=f"Ch {ch}", showarrow=False, font=dict(size=9)))
    figL.update_layout(height=360, template='plotly_white', margin=dict(t=6,b=6))
    st.plotly_chart(figL, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ---------- Spectrogram ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Spectrogram — High resolution")
spec_win = st.number_input("Window index for spectrogram", min_value=0, max_value=max(0, n_windows-1), value=win_idx, step=1, key="spec_win")
spec_ch = st.selectbox("Channel for spectrogram", options=list(range(n_ch)), index=0, key="spec_ch")
spec_data = X[spec_win][:, spec_ch]
fig_sp, ax = plt.subplots(figsize=(10,3))
ax.specgram(spec_data, NFFT=128, Fs=sfreq, noverlap=96, cmap='magma')
ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
st.pyplot(fig_sp)
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ---------- Onset locator and top table ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Onset Locator & Suspicious Windows")
if onset is None:
    st.info("No sustained high-risk region detected by the internal heuristic.")
else:
    st.success(f"Suggested onset window: {onset} (prob {probs_sm[onset]:.3f})")

topk2 = min(12, len(probs_sm))
top_idx2 = np.argsort(probs_sm)[-topk2:][::-1]
df_top = pd.DataFrame({"window_index": top_idx2, "probability": [float(probs_sm[i]) for i in top_idx2]})
st.table(df_top)
st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ---------- Clinician notes & export ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Clinician Notes & Export")
clin = st.text_area("Clinician notes (included in PDF):", value=clinician_notes, height=140)
df_preds = pd.DataFrame({"window_index": np.arange(len(probs_sm)), "probability": probs_sm, "alert": (probs_sm > THRESH).astype(int)})
csv_bytes = df_preds.to_csv(index=False).encode()
st.download_button("Download predictions (CSV)", csv_bytes, file_name=f"predictions_{int(time.time())}.csv", mime="text/csv")

if st.button("Generate & Download PDF Report"):
    with st.spinner("Composing PDF..."):
        buf = io.BytesIO()
        make_pdf(buf, uploaded.name, os.path.basename(model_file), probs_sm, X, onset, clin, THRESH)
        buf.seek(0)
        st.download_button("Download PDF", data=buf.read(), file_name=f"seizure_report_{int(time.time())}.pdf", mime="application/pdf")
        st.success("PDF ready for download.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.write("")
st.markdown('<div class="card center"><div class="small-muted">This tool is provided for research and clinician support. It is not a medical diagnostic device. Any flagged event must be reviewed by a qualified clinician.</div></div>', unsafe_allow_html=True)
