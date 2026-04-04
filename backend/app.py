from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import os
import sys

sys.path.append(os.path.dirname(__file__))
from predictor import predict, load_model
from tts import speak
from utils.helpers import decode_base64_image

# Tell Flask where to find templates and static files
# since our frontend folder is outside the backend folder
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')
)

# This allows the browser to make requests to our Flask server
# Without this the browser blocks the requests due to CORS policy
CORS(app)

@app.route('/')
def index():
    # Serves the main frontend page when you open localhost:5000
    return render_template('index.html')


@app.route('/app')
def app_page():
    # Serves the sign language recognition tool page
    return render_template('app.html')


@app.route('/health', methods=['GET'])
def health():
    # Simple check to confirm the server is running
    # The frontend uses this to show "Server online" or "Server offline"
    return jsonify({'status': 'ok', 'message': 'Server is running'})


@app.route('/predict', methods=['POST'])
def predict_gesture():
    # Receives a webcam frame from the browser as a base64 image
    # Runs hand detection and returns the predicted gesture

    data = request.get_json()

    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    # Decode the base64 image into something OpenCV can read
    frame = decode_base64_image(data['frame'])
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    gesture, confidence, _ = predict(frame)

    if gesture is None:
        return jsonify({'gesture': None, 'confidence': 0.0, 'detected': False})

    return jsonify({
        'gesture':    gesture,
        'confidence': round(confidence, 4),
        'detected':   True
    })


@app.route('/speak', methods=['POST'])
def speak_text():
    # Receives a text string and speaks it out loud
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text is empty'}), 400

    speak(text)
    return jsonify({'status': 'speaking', 'text': text})

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)