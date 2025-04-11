import pyttsx3

def speak_text(text):
    """Speaks the given text using pyttsx3."""
    engine = pyttsx3.init()  # Initialize the TTS engine
    engine.say(text)
    engine.runAndWait()  # Wait for speech to complete

if __name__ == "__main__":
    test_text = "This is a test of text to speech on Raspberry Pi."
    speak_text(test_text)
