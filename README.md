# 🧪 smiles2smell

> **AI-Powered Digital Olfaction Engine:** Predict 138 odor descriptors from molecular SMILES strings using Graph Neural Networks (MPNN-POM).

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepChem](https://img.shields.io/badge/DeepChem-OpenPOM-brightgreen)](https://deepchem.io/)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-0.875-blueviolet)](#results)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Overview

**`smiles2smell`** bridges chemistry and sensory perception by teaching machine learning models to "smell" molecules directly from their chemical graphs. Built on the **Principal Odor Map (POM)** architecture using Message Passing Neural Networks (MPNN), this project predicts probabilities across **138 distinct odor descriptors** (such as *vanilla*, *mint*, *fruity*, *woody*, *floral*, *sweaty*, *pungent*, and *citrus*).

### Key Features
- 🌐 **Modern Web Interface (`app.py`)**: Interactive Streamlit web app with 2D chemical structure rendering (RDKit), responsive Altair probability bar charts, and CSV/JSON export.
- 🪟 **Desktop GUI (`odor_ui.py`)**: Single-window Tkinter desktop app featuring live RDKit preview while typing SMILES strings, background model loading, and instant results.
- ⚡ **Headless CLI (`predict_smiles.py`)**: Fast command-line inference tool designed for batch pipelines and remote server/SSH workflows.
- 📓 **End-to-End Notebooks**: Step-by-step Jupyter notebooks covering environment verification, exploratory data analysis, and multi-label stratified training achieving **~0.875 ROC-AUC**.
- 📡 **E-Nose Hardware Pipeline (Part B)**: Guide and workflow for interfacing physical gas sensors (Bosch BME688 + Raspberry Pi) with BME AI-Studio.

---

## 📁 Repository Structure

```text
smiles2smell/
├── app.py                     # Streamlit web frontend
├── odor_ui.py                 # Desktop Tkinter GUI application
├── predict_smiles.py          # Command-line predictor
├── 01_setup_check.ipynb       # Step 1: CUDA & environment sanity check
├── 02_explore_data.ipynb      # Step 2: Odor dataset & class imbalance EDA
├── 03_train.ipynb             # Step 3: MPNN-POM training & evaluation
├── mpnn_pom_train.ipynb       # Upstream training reference notebook
├── models/
│   └── 03_train/
│       └── test_score.txt     # Benchmark results from 150-epoch run
├── .streamlit/
│   └── config.toml            # UI theme configuration
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Preconfigured Git ignore rules
└── LICENSE                    # MIT License
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

Python 3.10 is recommended:

```bash
git clone https://github.com/<your-username>/smiles2smell.git
cd smiles2smell

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Core Dependencies

#### With GPU (CUDA 12.4):
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install --pre deepchem
pip install -r requirements.txt
```

#### On CPU:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install --pre deepchem
pip install -r requirements.txt
```

### 3. Clone OpenPOM (Data & Base Models)

Clone the curated OpenPOM repository into the workspace root:

```bash
git clone https://github.com/ARY2260/openpom.git
```

---

## 💻 Usage

### 🌐 Streamlit Web UI
Launch the interactive web application:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Enter any SMILES string (e.g. `CCO` for ethanol or `COc1cc(C=O)ccc1O` for vanillin) to inspect 2D molecular graphs and predicted odor probabilities.

### 🪟 Desktop GUI
Run the native desktop GUI:

```bash
python odor_ui.py
```

### ⚡ Command Line (CLI)
Predict smells directly in the terminal:

```bash
# Predict top odors for Vanillin
python predict_smiles.py "COc1cc(C=O)ccc1O" --top 10

# Predict top odors for Ethanol on GPU
python predict_smiles.py CCO --device cuda
```

---

## 📊 Results & Model Performance

Trained using a Message Passing Neural Network with edge featurization and a multi-label stratified 80/10/10 split on 4,983 curated molecules:

| Model Setup | Epochs | Valid ROC-AUC | Test ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| Single Model (baseline) | 5 | 0.806 | 0.806 | ~30s |
| **Single Model (this run)** | **150** | **0.880** | **0.875** | **12m 30s (RTX 4050)** |
| 10-Model Ensemble (paper) | 150 | — | ~0.890 | — |

---

## 📡 Part B: Physical Electronic Nose (Optional)

The repository also includes notes for integrating physical odor sensors:
- **Hardware**: Bosch BME688 8-in-1 environmental gas sensor breakout + Raspberry Pi.
- **Workflow**:
  1. Record gas resistance and temperature/humidity profiles in a sealed container using the Raspberry Pi.
  2. Train gas classification models in Bosch **BME AI-Studio**.
  3. Deploy classification configs directly to the sensor firmware.

---

## 📜 References & Acknowledgments

- **OpenPOM**: [ARY2260/openpom](https://github.com/ARY2260/openpom)
- **Principal Odor Map (POM)**: Lee et al., *"A Principal Odor Map Unifies Diverse Tasks in Olfaction"*, Science (2023).
- **DeepChem**: Molecule featurization and multi-task graph network architecture.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
