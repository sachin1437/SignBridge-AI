import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import os
import numpy as np
import urllib.request

# Download hand landmarker model
model_path = '/kaggle/working/hand_landmarker.task'
if not os.path.exists(model_path):
    print('Downloading hand landmarker model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print('Downloaded!')

# Config
GESTURES = [
    'Call_me', 'Greetings', 'Happy', 'Hello', 'Help',
    'Hurts', 'I_Hate_You', 'I_Love_You', 'Me', 'No',
    'No_Gesture', 'Ok', 'Perfect', 'Please', 'Sit',
    'Stand_Up', 'Stop', 'Yes', 'You'
]

DATASET_DIR = '/kaggle/input/hand-gestures/augmented_dataset'
OUTPUT_CSV  = '/kaggle/working/landmarks.csv'

# Check dataset path
if not os.path.exists(DATASET_DIR):
    # Try alternate path
    base = '/kaggle/input'
    for d in os.listdir(base):
        print(os.path.join(base, d))

def get_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options      = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    pts   = landmarks.reshape(21, 3)
    wrist = pts[0].copy()
    pts   = pts - wrist
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()

def extract():
    landmarker = get_landmarker()
    saved      = 0
    skipped    = 0

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{ax}{i}' for i in range(21) for ax in ['x','y','z']] + ['label']
        writer.writerow(header)

        for gesture in GESTURES:
            folder = os.path.join(DATASET_DIR, gesture)
            if not os.path.exists(folder):
                print(f'[WARNING] Not found: {folder}')
                continue

            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
            print(f'Processing {gesture}: {len(imgs)} images...')

            for fname in imgs:
                path  = os.path.join(folder, fname)
                image = cv2.imread(path)
                if image is None:
                    skipped += 1
                    continue

                rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    row  = []
                    for lm in hand:
                        row.extend([lm.x, lm.y, lm.z])
                    raw        = np.array(row)
                    normalized = normalize_landmarks(raw)
                    row_final  = list(normalized) + [gesture]
                    writer.writerow(row_final)
                    saved += 1
                else:
                    skipped += 1

    print(f'\nExtraction done!')
    print(f'Saved  : {saved}')
    print(f'Skipped: {skipped}')
    print(f'Output : {OUTPUT_CSV}')

extract()