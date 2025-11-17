project:
  name: "Robust Seizure Prediction Model"
  subtitle: "Using EEG Signals + Deep Learning (CNN–BiLSTM)"
  description: |
    This project implements a clinical-grade seizure prediction and detection
    system using EEG signals. It provides a premium Streamlit dashboard with 
    EEG waveform visualization, spectrogram analysis, probability timeline, 
    seizure onset detection, and medical-style PDF report generation.

dataset:
  name: "CHB-MIT Seizure Dataset"
  source: "Kaggle"
  link: "https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset"
  note: |
    Dataset is NOT included in this repository due to GitHub size limits.
    Place downloaded dataset files inside: seizure_predictor/dataset/

project_structure:
  root: "seizure_predictor/"
  folders:
    - app/streamlit_app.py
    - src/config.py
    - src/data_loader.py
    - src/edf_reader.py
    - src/model.py
    - src/train.py
    - src/evaluate.py
    - src/utils.py
    - models/best_model.h5
    - dataset/raw/
    - dataset/processed/
    - requirements.txt
    - README.md

features:
  eeg_support:
    - ".edf"
    - ".npz"
  seizure_detection:
    - "Automatic feature extraction"
    - "Seizure vs Non-Seizure classification"
    - "Probability timeline"
    - "Seizure onset detection"
  premium_dashboard:
    - "Clinical-grade white UI"
    - "Multi-channel EEG viewer"
    - "Spectrogram visualization"
    - "Probability graph"
    - "Suspicious region highlight"
    - "Session notes"
    - "Responsive modern layout"
  report_generation:
    - "Medical-style PDF"
    - "CSV export"

model_architecture:
  cnn:
    description: "Extract spatial EEG channel patterns"
  bilstm:
    description: "Capture long-range temporal dependencies"
  dense_layers:
    description: "Dense + Dropout for generalization"
  output_layer:
    activation: "Sigmoid"
    purpose: "Binary seizure classification"
  advantage: |
    Combines spatial + temporal learning, ideal for EEG-based seizure detection.

installation:
  steps:
    - step: "Clone repository"
      command: |
        git clone https://github.com/shoyab778/Robust-Seizure-Prediction-Model.git
        cd Robust-Seizure-Prediction-Model

    - step: "Create & activate virtual environment"
      command: |
        python -m venv env
        env\Scripts\activate   # For Windows

    - step: "Install dependencies"
      command: |
        pip install -r requirements.txt

    - step: "Place dataset in project folder"
      path: "seizure_predictor/dataset/"

    - step: "Train model (Optional)"
      command: |
        python src/train.py

    - step: "Run Streamlit dashboard"
      command: |
        streamlit run app/streamlit_app.py
      url: "http://localhost:8501"

dashboard_features:
  upload: "Upload .edf or .npz EEG files with auto processing"
  eeg_viewer:
    - "Stacked clinical EEG visualization"
    - "Grid lines"
    - "Zoom"
    - "Navigation"
  spectrogram:
    - "Time-frequency analysis"
    - "High-intensity seizure region detection"
  probability_timeline:
    - "Per-second seizure probability"
    - "Colored risk pattern"
  onset_locator:
    - "Automatic marking of predicted seizure onset"
  pdf_report:
    includes:
      - "Waveforms"
      - "Spectrogram"
      - "Probability graph"
      - "Detection results"
      - "Precautions and recommendations"

medical_interpretation:
  provided:
    - "Possible risks"
    - "Meaning of results"
    - "Precautions"
    - "When to seek medical help"
  note: "Tool is NOT a medical diagnosis. For research & academic use only."

future_enhancements:
  - "Live EEG streaming"
  - "Early seizure prediction (pre-ictal modeling)"
  - "Transformer-based EEG models"
  - "Edge deployment (Jetson Nano, Raspberry Pi)"

credits:
  - "CHB-MIT EEG Epilepsy Dataset"
  - "MIT PhysioNet"
  - "Kaggle contributors"

contact:
  email: "smdshoyab07@gmail.com"
