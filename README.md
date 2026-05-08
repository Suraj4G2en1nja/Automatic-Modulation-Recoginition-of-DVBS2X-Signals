# Automatic-Modulation-Recoginition-of-DVBS2X-Signals
AFR-Net + S-GBC for Robust Automatic Modulation Recognition (AMR)
📌 Overview

This project presents a Physics-Informed Hybrid Fusion Framework for Automatic Modulation Recognition (AMR) in DVB-S2X satellite communication systems.
# demo video is provided
The framework combines:

AFR-Net (Adaptive Fusion-Residual Network)
→ Deep learning architecture for robust low-SNR modulation recognition.
S-GBC (Statistical Gradient Boosting Classifier)
→ Lightweight statistical classifier optimized for high-speed and high-SNR environments.

The proposed dual-path system achieves a balance between:

High classification accuracy
Low-SNR robustness
Computational efficiency
Real-time SDR deployment capability
🚀 Key Features
Dual-path AMR architecture
Physics-informed statistical feature engineering
Hybrid fusion of raw IQ + statistical descriptors
Robust classification under noisy channels
Optimized for DVB-S2X satellite signals
Real-time inference capability
Explainable AI (XAI) based feature analysis
Lightweight deployment support for SDR systems
📊 Performance Highlights
Model	Peak Accuracy	Low-SNR Robustness (-20 dB)	Inference Latency
AFR-Net	85.60%	15.55%	4.5 ms
S-GBC	93.41%	10.32%	120 μs
Standard CNN	81.40%	9.80%	3.8 ms
ResNet-18	83.20%	11.20%	12.4 ms
🧠 Proposed Architecture
1️⃣ AFR-Net (Adaptive Fusion-Residual Network)

AFR-Net uses two parallel branches:

Temporal Branch
Processes raw IQ samples
Uses 1D CNN layers
Extracts temporal signal structures
Statistical Branch
Processes engineered statistical features
Uses dense embedding layers
Captures higher-order signal characteristics
Fusion Layer
Combines both representations
Produces a unified 256-dimensional latent representation
Classification Head
Softmax-based modulation prediction
2️⃣ S-GBC (Statistical Gradient Boosting Classifier)

A lightweight classifier using:

LightGBM-based boosting
Higher-order statistical features
Bayesian hyperparameter optimization

Optimized for:

Fast inference
Low memory usage
Embedded SDR deployment
📡 Supported Modulation Schemes

The framework supports recognition of:

BPSK
QPSK
8PSK
PAM4
GFSK
CPFSK
QAM16
QAM64
AM-DSB
AM-SSB
WBFM
📂 Dataset

The project uses radio modulation datasets inspired by:

RadioML2016.10a
GNU Radio generated SDR datasets
SNR Range

-20 dB → +18 dB

Input Representations
Raw IQ samples
Amplitude
Phase
Higher-order cumulants
Energy features
Spectral features
🛠️ Technologies Used
Languages
Python
Libraries & Frameworks
PyTorch
NumPy
SciPy
Scikit-learn
LightGBM
Matplotlib
Pandas
Signal Processing
GNU Radio
SDR-based preprocessing
📈 Feature Engineering

The framework extracts:

4th-order cumulants
6th-order cumulants
8th-order cumulants
Amplitude features
Phase features
Energy statistics
Kurtosis
Spectral representations

Total feature dimension:
391-dimensional statistical vector

📊 Visualizations Included
SNR distribution analysis
Modulation class distribution
Amplitude vs phase plots
Energy variation graphs
t-SNE feature clustering
Confusion matrices
Feature importance analysis
Modulation DNA profiling
🧪 Research Contributions
Major Contributions

✔ Dual-path AMR framework for DVB-S2X
✔ Hybrid fusion of deep learning + expert statistical features
✔ Low-SNR robust modulation recognition
✔ Lightweight high-speed classifier for SDR deployment
✔ Explainable statistical feature analysis
✔ Real-time compatible architecture
