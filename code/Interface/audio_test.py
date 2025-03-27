import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Available voices:")
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name}")

engine.setProperty('rate', 150)  # Adjust speed
engine.say("Hello, testing audio output")
engine.runAndWait()

