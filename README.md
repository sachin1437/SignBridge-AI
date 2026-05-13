# SignBridge 🤟

**Real-Time Indian Sign Language Recognition for Assistive Communication**

SignBridge is a software-only, real-time ISL recognition system that translates Indian Sign Language gestures into spoken audio using a standard RGB webcam. No specialized hardware required.

---

## Demo
Webcam → MediaPipe Holistic → BiLSTM Classifier → Text-to-Speech

---

## Features

- 🎥 **Webcam-only** — no depth sensors, gloves, or special hardware
- 🔒 **Privacy-preserving** — raw video never stored; only skeletal landmarks processed
- ⚡ **Real-time** — runs at 24+ FPS on standard laptop hardware
- 🔇 **Offline TTS** — works without internet connectivity
- 🧠 **BiLSTM temporal modeling** — captures motion dynamics, not just static poses
- 📊 **90.6% test accuracy** across 19 ISL gesture classes

---

## Gesture Classes (19)

| Category | Gestures |
|---|---|
| Greetings | Hello, Greetings |
| Emotions | Happy, Hurts, I_Love_You, I_Hate_You, Perfect |
| Commands | Sit, Stand_Up, Stop, Call_me |
| Social | Me, You, Yes, No, Please, Help, Ok |
| System | No_Gesture |

---

## System Architecture
Input Frame (720p RGB)
↓
MediaPipe Holistic
(21 hand + 33 pose landmarks)
↓
Feature Vector (258-dim per frame)
↓
Sequence Buffer (30 frames)
↓
BiLSTM Classifier (808K params)
↓
Confidence Gate (τ = 0.7)
↓
Temporal Voting Buffer (5 frames)
↓
Text-to-Speech Output

---

## Installation

```bash
git clone https://github.com/sachin1437/SignBridgeAI
cd SignBridgeAI
pip install -r requirements.txt
```

### Requirements
tensorflow>=2.10
mediapipe==0.10
opencv-python
numpy
pyttsx3

---

## Usage

### Run real-time inference
```bash
python app.py
```

### Collect dataset
```bash
python collect_data.py --class_name Hello --num_samples 200
```

### Train model
```bash
python train.py
```

---

## Dataset

Custom ISL dataset collected using standard 720p RGB webcam:

| Parameter | Value |
|---|---|
| Gesture classes | 19 |
| Raw clips per class | 200 |
| Frames per sequence | 30 |
| Feature dimensions | 258 |
| Augmentation factor | ~8× |
| Train/Val/Test split | 70/15/15 |

### Data Augmentation
- Gaussian jitter (σ = 0.01)
- Horizontal flip (left-handed signer synthesis)
- Temporal speed warp (0.8× – 1.2×)
- Spatial scaling (0.9× – 1.1×)

---

## Model Performance

| Model | Accuracy | Macro-F1 |
|---|---|---|
| MLP (baseline) | 81.2% | 0.798 |
| LSTM (unidirectional) | 87.4% | 0.866 |
| **BiLSTM (proposed)** | **90.6%** | **0.901** |

Classes with perfect recall: `Call_me`, `I_Hate_You`, `I_Love_You`, `Me`, `Perfect`, `You`

---

## Project Structure
SignBridgeAI/
├── app.py                  # Real-time inference
├── train.py                # Model training
├── collect_data.py         # Dataset collection
├── model/
│   └── bilstm_model.h5     # Trained model
├── dataset/
│   └── all_images/         # Collected gesture data
├── utils/
│   ├── landmark_utils.py   # MediaPipe extraction
│   └── augmentation.py     # Data augmentation
└── requirements.txt

---

## Tech Stack

- **Python** 3.10
- **TensorFlow / Keras** — BiLSTM model
- **MediaPipe** 0.10 — Skeletal landmark extraction
- **OpenCV** — Webcam capture
- **pyttsx3** — Offline text-to-speech

---

## Research Paper

> *SignBridge: A Real-Time Visual Computing Framework for Indian Sign Language Recognition Using MediaPipe Holistic and Bidirectional LSTM for Assistive Human–Computer Interaction*
>
> Sachin Gupta, Devendar Kumar
> School of Computer Applications, Lovely Professional University
> Submitted to Visual Computing for Industry, Biomedicine, and Art (VCIBA), Springer

---

## Authors

**Sachin Gupta** — [@sachin1437](https://github.com/sachin1437) | [LinkedIn](https://linkedin.com/in/sachin-gupta1420)

**Devendar Kumar** — Assistant Professor, School of Computer Applications, LPU

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

School of Computer Applications, Lovely Professional University, for institutional support. All volunteers who contributed gesture recordings to the dataset.
```