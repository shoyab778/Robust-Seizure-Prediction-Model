# 🧠 Robust Seizure Prediction Model  
### Using EEG Signals + Deep Learning (CNN–BiLSTM)

This project implements a **clinical-grade seizure prediction and detection system** using EEG signals.  
It includes:

- A premium Streamlit dashboard  
- EEG waveform visualization  
- Spectrogram analysis  
- Probability timeline  
- Seizure onset detection  
- Medical-style PDF report generation  
- Automatic EEG preprocessing  
- CHB-MIT dataset support  
- Clean, modern, professional UI  

---

# 📥 Dataset

The CHB-MIT Seizure Dataset used in this project can be downloaded from Kaggle:

➡️ **[CHB-MIT Seizure Dataset (Kaggle)](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset)**

> ⚠️ Dataset is **not included in this repo** because of GitHub's size restriction.  
> Place downloaded dataset files inside:  
seizure_predictor/dataset/

yaml
Copy code

---

# 📁 Project Structure

seizure_predictor/
│
├── app/
│ └── streamlit_app.py
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── edf_reader.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── models/
│ └── best_model.h5
│
├── dataset/
│ └── raw/
│ └── processed/
│
├── requirements.txt
└── README.md

yaml
Copy code

---

# 🚀 Features

## ✔ EEG File Support  
- `.edf`  
- `.npz`

## ✔ Seizure Detection  
- Automatic feature extraction  
- Seizure vs Non-Seizure classification  
- Probability timeline  
- Seizure onset detection  

## ✔ Premium Dashboard  
- Clean, clinical-grade white UI  
- Multi-channel EEG waveform viewer  
- Spectrogram visualization  
- Probability graph viewer  
- Suspicious region highlight  
- Session notes  
- Responsive modern layout  

## ✔ Report Generation  
- PDF medical report (waveforms + spectrogram + conclusions)  
- CSV export for EEG features  

---

# 🧠 Model Architecture

The prediction model uses:

### 📌 **1. Convolution Blocks (1D CNN)**
- Extract low-level EEG channel patterns  
- Detect spatial filters  

### 📌 **2. BiLSTM Layers**
- Learn long-range temporal dependencies  
- Identify pre-ictal → ictal transitions  
- Bidirectional processing enhances accuracy  

### 📌 **3. Fully Connected Layers**
- Dense + Dropout for generalization  

### 📌 **4. Output Layer**
- **Sigmoid activation** for binary seizure classification  

This hybrid architecture gives **temporal awareness + spatial understanding**, ideal for EEG-based detection.

---

# 🔧 Installation & Setup

Follow these steps to run the project locally.

---

## ✔ 1. Clone the repository

```bash
git clone https://github.com/shoyab778/Robust-Seizure-Prediction-Model.git
cd Robust-Seizure-Prediction-Model
✔ 2. Create & activate virtual environment (Recommended)
bash
Copy code
python -m venv env
env\Scripts\activate   # For Windows
✔ 3. Install dependencies
bash
Copy code
pip install -r requirements.txt
✔ 4. Place dataset in project folder
Copy dataset files to:

bash
Copy code
seizure_predictor/dataset/
✔ 5. Train the model (Optional)
bash
Copy code
python src/train.py
✔ 6. Run the Streamlit app (Main Dashboard)
bash
Copy code
streamlit run app/streamlit_app.py
Then open:

arduino
Copy code
http://localhost:8501
📊 Dashboard Features (Premium)
🔹 Upload EEG File
Upload .edf or .npz → automatic processing.

🔹 EEG Waveform Viewer
Multi-channel clinical-style view

Clean stacked waveform design

Interactive zoom & navigation

🔹 Spectrogram Viewer
Time–frequency representation

Highlights high-energy seizure regions

🔹 Probability Timeline
Per-second seizure probability

Graphical risk pattern

🔹 Seizure Onset Locator
Automatic marking of predicted onset

🔹 Medical-Style PDF Report
Includes:

Waveforms

Spectrogram

Probability graph

Detection results

Suggested precautions

🩺 Medical Interpretation (Auto-generated)
If seizure is detected, the dashboard provides:

Possible risks

What this means clinically

Precautionary steps

When to seek urgent medical help

⚠ This tool is NOT a medical diagnosis.
It is for research & academic purposes.

🔮 Future Enhancements
Live EEG data streaming

Improved early prediction (pre-ictal modeling)

Multi-channel transformer models

Portable edge-device deployment

❤️ Credits
CHB-MIT EEG Epilepsy Dataset

MIT PhysioNet

Kaggle contributors

📧 Contact
For help, queries, or collaboration:
