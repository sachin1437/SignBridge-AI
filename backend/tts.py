import pyttsx3
import threading

# Initialize the TTS engine once when this file is imported
# pyttsx3 uses the system's built-in voice engine
# On Windows it uses SAPI5 which is already installed
engine = pyttsx3.init()

# Rate is words per minute — 150 is natural conversational speed
engine.setProperty('rate', 150)

# Volume is 0.0 to 1.0
engine.setProperty('volume', 1.0)

# This lock prevents two speak() calls from running at the same time
# For example if the user clicks speak twice quickly
_lock = threading.Lock()


def speak(text):
    # We run speaking in a separate thread so it doesn't freeze the Flask server
    # While the server is speaking, it should still be able to handle other requests

    def _speak():
        with _lock:
            engine.say(text)
            engine.runAndWait()

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


def set_rate(rate):
    # Allows changing speed later if needed
    # 100 = slow, 150 = normal, 200 = fast
    engine.setProperty('rate', rate)


def set_volume(volume):
    # Allows changing volume later if needed
    # 0.0 = mute, 1.0 = full volume
    engine.setProperty('volume', volume)