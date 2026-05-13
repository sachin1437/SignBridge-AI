<div align="center">

# 🤟 SignBridge

### Real-Time Indian Sign Language Recognition for Assistive Communication

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.6%25-brightgreen?style=for-the-badge)]()
[![LPU](https://img.shields.io/badge/LPU-Research-orange?style=for-the-badge)]()

<br/>

> *A software-only, privacy-preserving ISL recognition framework that bridges the communication gap for 63 million hearing-impaired Indians.*


</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Gesture Classes](#gesture-classes)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Research Paper](#research-paper)
- [Tech Stack](#tech-stack)
- [Authors](#authors)

---

## 🧠 Overview

**SignBridge** is a real-time Indian Sign Language (ISL) recognition system built for assistive human–computer interaction. It uses **MediaPipe Holistic** for skeletal landmark extraction and a **Bidirectional LSTM** network for temporal gesture classification, followed by offline **text-to-speech** output.

Unlike most SLR systems that rely on expensive hardware or raw video processing, SignBridge:
- Works entirely on a **standard RGB webcam**
- Discards raw video — processes only **258-dimensional skeletal vectors**
- Runs in **real-time on a laptop CPU** without GPU support
- Produces **spoken output** via offline TTS engine

This makes it **privacy-preserving, hardware-light, and deployable** in real-world assistive scenarios — hospitals, schools, and public services.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎥 Webcam-only | No depth sensors, gloves, or special hardware |
| 🔒 Privacy-preserving | Raw video never stored; only landmarks processed |
| ⚡ Real-time | 24+ FPS inference on standard laptop |
| 🔇 Offline TTS | Works without internet via pyttsx3 |
| 🧠 BiLSTM | Captures temporal motion dynamics |
| 📊 High accuracy | 90.6% test accuracy across 19 classes |
| 🔁 Augmented training | 8× dataset expansion via landmark augmentation |
| 🛡️ Robust inference | Temporal voting + confidence gating |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SignBridge Pipeline                  │
│                                                         │
│  📷 Webcam Input (720p @ 30 FPS)                        | 
│          │                                              │
│          ▼                                              │
│  🦴 MediaPipe Holistic                                  │
│     • 21 hand landmarks (×2)                           │
│     • 33 pose landmarks                                 │
│     → 258-dim feature vector per frame                  │
│          │                                              │
│          ▼                                              │
│  📦 Sequence Buffer (30 frames)                         │
│          │                                              │
│          ▼                                              │
│  🧠 BiLSTM Classifier (808K params)                     │
│     • 2× Bidirectional LSTM (128 units)                 │
│     • Dropout (0.3) + Dense layers                      │
│          │                                              │
│          ▼                                              │
│  🎯 Confidence Gate (τ = 0.7)                           │
│          │                                              │
│          ▼                                              │
│  🗳️ Temporal Voting Buffer (5 frames)                   │
│          │                                              │
│          ▼                                              │
│  🔊 Text-to-Speech Output (pyttsx3)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🤙 Gesture Classes (19)

| Category | Gestures |
|---|---|
| 👋 Greetings | `Hello` `Greetings` |
| 😊 Emotions | `Happy` `Hurts` `I_Love_You` `I_Hate_You` `Perfect` |
| 🫵 Commands | `Sit` `Stand_Up` `Stop` `Call_me` |
| 🗣️ Social | `Me` `You` `Yes` `No` `Please` `Help` `Ok` |
| ⏸️ System | `No_Gesture` |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/sachin1437/SignBridgeAI
cd SignBridgeAI

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.10
mediapipe==0.10
opencv-python
numpy
pyttsx3
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Usage

### Run Real-Time Inference
```bash
python app.py
```

### Collect Dataset
```bash
python collect_data.py --class_name Hello --num_samples 200
```

### Train Model
```bash
python train.py
```

### Evaluate Model
```bash
python evaluate.py
```

**Controls during inference:**
- `Q` — Quit
- `S` — Save current frame to dataset

---

## 📂 Dataset

Custom ISL dataset recorded with a standard 720p RGB webcam across multiple signers and lighting conditions.

| Parameter | Value |
|---|---|
| Gesture classes | 19 |
| Raw clips per class | 200 |
| Total raw clips | 3,800 |
| Frames per sequence | 30 |
| Feature dimensions | 258 |
| Augmentation factor | ~8× |
| Train / Val / Test | 70% / 15% / 15% |
| Recording FPS | 30 |
| Resolution | 1280 × 720 |

### Data Augmentation Strategy

| Technique | Description |
|---|---|
| Gaussian Jitter | Noise injection (σ=0.01) to simulate MediaPipe estimation noise |
| Horizontal Flip | Mirror landmarks to synthesize left-handed signers |
| Temporal Speed Warp | Random resampling at 0.8×–1.2× to simulate signing speed variation |
| Spatial Scaling | Uniform scaling 0.9×–1.1× around centroid |

---

## 📊 Model Performance

### Comparative Results

| Model | Accuracy | Macro-F1 | Params |
|---|---|---|---|
| MLP (baseline) | 81.2% | 0.798 | 1.05M |
| LSTM (unidirectional) | 87.4% | 0.866 | 0.42M |
| **BiLSTM (proposed)** | **90.6%** | **0.901** | **0.81M** |

### Training Curves

![Training History](training_history.png)

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Per-Class Highlights

✅ **Perfect recall (100%):** `Call_me` `I_Hate_You` `I_Love_You` `Me` `Perfect` `You`

⚠️ **Challenging classes:** `Help` (60.6%) `Ok` (68.2%) `Yes` (71.8%) — confused due to similar closed-fist configurations

---

## 🗂️ Project Structure

```
SignBridgeAI/
├── app.py                    # Real-time inference entry point
├── train.py                  # Model training script
├── evaluate.py               # Evaluation and metrics
├── collect_data.py           # Dataset collection script
├── requirements.txt
│
├── model/
│   ├── bilstm_model.h5       # Trained BiLSTM model
│   └── label_encoder.npy     # Class label encoder
│
├── dataset/
│   └── all_images/           # Collected gesture sequences
│       ├── Hello/
│       ├── Help/
│       └── ...
│
├── utils/
│   ├── landmark_utils.py     # MediaPipe extraction + normalization
│   ├── augmentation.py       # Landmark-level augmentation
│   └── tts_utils.py          # TTS output handler
│
├── results/
│   ├── confusion_matrix.png
│   └── training_history.png
│
└── assets/
    └── banner.png
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x / Keras |
| Pose Estimation | MediaPipe Holistic 0.10 |
| Video Capture | OpenCV 4.x |
| Text-to-Speech | pyttsx3 (offline) |
| Data Processing | NumPy / scikit-learn |
| Visualization | Matplotlib / Seaborn |

</div>

---

## 📄 Research Paper

> **SignBridge: A Real-Time Visual Computing Framework for Indian Sign Language Recognition Using MediaPipe Holistic and Bidirectional LSTM for Assistive Human–Computer Interaction**
>
> Sachin Gupta, Devendar Kumar
> *School of Computer Applications, Lovely Professional University, Jalandhar, Punjab, India*
>
> 📌 Submitted to **Visual Computing for Industry, Biomedicine, and Art (VCIBA)**, Springer Nature
>
> 🧪 Plagiarism Score: **5% (Grade A)** — DrillBit

---

## 👨‍💻 Authors

<div align="center">

| | |
|---|---|
| **Sachin Gupta** | MCA Student, School of Computer Applications, LPU |
| GitHub | [@sachin1437](https://github.com/sachin1437) |
| LinkedIn | [sachin-gupta1420](https://linkedin.com/in/sachin-gupta1420) |
| Email | sachingupta1437@gmail.com |

| | |
|---|---|
| **Devendar Kumar** | Assistant Professor, School of Computer Applications, LPU |
| Email | devender.kumar2k7@gmail.com |

</div>

---

## 🏢 About

SignBridge is a product of **[NetraaLabs](https://netraalabs.netlify.app)** — building AI-powered visual intelligence systems for real-world impact.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- School of Computer Applications, Lovely Professional University for institutional support
- All volunteers who contributed gesture recordings to the dataset
- Google MediaPipe team for the open-source pose estimation framework

---

<div align="center">

Made with ❤️ for the deaf and hard-of-hearing community

⭐ Star this repo if you find it useful!

</div>