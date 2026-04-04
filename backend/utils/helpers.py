import numpy as np
import cv2
import base64


def decode_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def landmarks_to_array(hand_landmarks):
    result = []
    for point in hand_landmarks:
        result.extend([point.x, point.y, point.z])
    return np.array(result)


def normalize_landmarks(landmarks):
    pts = landmarks.reshape(21, 3)
    pts = pts - pts[0]  # subtract wrist
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()