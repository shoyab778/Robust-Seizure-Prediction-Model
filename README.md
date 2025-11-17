# 🧠 Robust Seizure Prediction Model  
### Using EEG Signals + Deep Learning (CNN–BiLSTM)

This project implements a **clinical-grade seizure prediction and detection system** using EEG signals.  
It includes:

- A Streamlit dashboard  
- EEG waveform visualization  
- Spectrogram analysis  
- Probability timeline  
- Seizure onset detection  
- Medical-style PDF report generation  
- Automatic pre-processing  
- CHB-MIT dataset support  


# 📦 Dataset

The CHB-MIT Seizure Dataset used in this project can be downloaded from Kaggle:

➡️ **[CHB-MIT Seizure Dataset (Kaggle)](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset)**

> ⚠️ The dataset is **not included in this GitHub repo** due to size limits.  
> Place all downloaded dataset files inside:

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


## ✔ Report Generation  
- PDF medical report (waveforms + spectrogram + conclusions)  
- CSV export for EEG features  

---

# 🧠 Model Architecture
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

## 1. Clone the repository

```bash
git clone https://github.com/shoyab778/Robust-Seizure-Prediction-Model.git
cd Robust-Seizure-Prediction-Model
```

## 2. Create & activate virtual environment (Recommended)

```bash
python -m venv env
env\Scripts\activate   # For Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Place dataset in project folder

```bash
seizure_predictor/dataset/
```

## 5. Train the model

```bash
python src/train.py
```

## 6. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Credits

CHB-MIT EEG Epilepsy Dataset

MIT PhysioNet

Kaggle contributors
