import pyttsx3
import threading

_lock   = threading.Lock()
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 150)
        _engine.setProperty('volume', 1.0)
    return _engine

def speak(text):
    def _speak():
        with _lock:
            try:
                engine = get_engine()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error: {e}")
                # Reinitialize engine on failure
                global _engine
                _engine = None

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

def set_rate(rate):
    get_engine().setProperty('rate', rate)

def set_volume(volume):
    get_engine().setProperty('volume', volume)