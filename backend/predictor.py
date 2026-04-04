import numpy as np
import cv2
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(os.path.dirname(__file__))
from utils.helpers import normalize_landmarks, landmarks_to_array

# ── CONFIG ──────────────────────────────────────────────
MODEL_DIR            = os.path.join(os.path.dirname(__file__), 'model')
CLASSES_PATH         = os.path.join(MODEL_DIR, 'classes.npy')
HAND_MODEL_PATH      = os.path.join(MODEL_DIR, 'hand_landmarker.task')
CONFIDENCE_THRESHOLD = 0.75
# ────────────────────────────────────────────────────────

keras_model = None
classes     = None


def load_model():
    global keras_model, classes

    if os.path.exists(CLASSES_PATH):
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        print(f"[Predictor] Classes loaded: {list(classes)}")
    else:
        classes = np.array([
            'Call_me', 'Greetings', 'Happy', 'Hello', 'Help',
            'Hurts', 'I_Hate_You', 'I_Love_You', 'Me', 'No',
            'No_Gesture', 'Ok', 'Perfect', 'Please', 'Sit',
            'Stand_Up', 'Stop', 'Yes', 'You'
        ])
        print("[Predictor] Model not trained yet — using placeholder classes")

    # TODO: uncomment once model.h5 is ready
    # from tensorflow.keras.models import load_model as keras_load
    # keras_model = keras_load(os.path.join(MODEL_DIR, 'model.h5'))


def get_landmarker():
    base_options = mp_python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def predict(frame):
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    landmarker = get_landmarker()
    result     = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, 0.0, frame

    hand = result.hand_landmarks[0]
    h, w = frame.shape[:2]
    for lm in hand:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 170), -1)

    raw        = landmarks_to_array(hand)
    normalized = normalize_landmarks(raw)
    input_data = normalized.reshape(1, -1)

    # ── PLACEHOLDER ──
    gesture    = "Hello"
    confidence = 0.99
    # ─────────────────

    # TODO: uncomment once model.h5 is ready
    # predictions = keras_model.predict(input_data, verbose=0)[0]
    # idx         = np.argmax(predictions)
    # confidence  = float(predictions[idx])
    # gesture     = classes[idx] if confidence >= CONFIDENCE_THRESHOLD else "No_Gesture"

    return gesture, confidence, frame