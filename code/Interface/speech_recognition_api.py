import sys
import os

from enum import Enum
import speech_recognition as sr
import collections
import threading
import time

# Add the Log directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Logs')))

from Logging import Log

#== Global Variables ==#


#== Enums ==#
class Recognizer(Enum):
    """
    Enum for different speech recognizers.
    """
    GOOGLE = 1
    SPHINX = 2
    HOUNDIFY = 3

#== Classes ==#

class Listen:
    """
        Class to handle speech recognition using various APIs.
    """
    def __init__(self, wake_word: str = "lane", sleep_word: str = "stop", recognizer_name: str = Recognizer.GOOGLE, device_name: str = "USB PnP Sound Device:"):
        """
        Initialize the Speech class with a wake word.
        """
        self.wake_word = wake_word # "Hey Lain"
        self.sleep_word = sleep_word # "Stop" or "Sleep"
        self.recognizer_name = recognizer_name  # Default recognizer name
        self.device_name = device_name
        self.MIC = sr.Microphone(device_index=find_microphone(self.device_name))
        self.recognizer = sr.Recognizer()


    def listen(self) -> str:
        
        """
        Function to recognize speech from audio input.
        """

        try:

            self.recognizer.pause_threshold = 2.0 # allow longer pauses
            
            # Use the microphone as the audio source
            with self.MIC as source:
                print("Please say something...")
                # Adjust for ambient noise and record audio
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
        
            #TODO: Get a API key as the default is only good for personal/testing purposes 

            try: 
                if self.recognizer_name == Recognizer.GOOGLE:
                    # Measure time for Google Web Speech API
                    print("Using Google Web Speech API...") 
                    start_time = time.time()
                    text = self.recognizer.recognize_google(audio)
                    end_time = time.time()
                    # print(f"Your Question: {text} (Time taken: {(end_time - start_time):.2f} s)")

                elif self.recognizer_name == Recognizer.SPHINX:
                    # Measure time for Sphinx
                    print("Using Sphinx...")
                    start_time = time.time()
                    text = self.recognizer.recognize_sphinx(audio)
                    end_time = time.time()
                    # print(f"Your Question: {text} (Time taken: {(end_time - start_time):.2f} s)")

                elif self.recognizer_name == Recognizer.HOUNDIFY:
                    #Measure time for Houndify
                    print("Using Houndify...")
                    start_time = time.time()
                    text = self.recognizer.recognize_houndify(audio, client_id="QGbfTnsp6zpB8m6yFC4Cfg==", client_key="xsISiTIHIsCKckTShaOay6sBX8zduFibr3v3DhKYDN3UvOMpZTCa66NDw6tFMLRlDW9KGkjtVCWC0l-uX5h_eg==")
                    end_time = time.time()
                    # print(f"Your Question: {text} (Time taken: {(end_time - start_time):.2f} s)")
            

            except sr.UnknownValueError:
                Log.log("ERROR", f"{self.recognizer_name} Recognition could not understand audio")
                return None
        except sr.RequestError as e:
            Log.log("ERROR", f"Could not request results from {self.recognizer_name} Recognition service; {e}")
            return None

        return text

    def listen_for_action_word(self):
        """
        Continuously listen for the wake word.
        """
        self.recognizer.pause_threshold = 0.8 # default

        text = ""
        with self.MIC as source:
            self.recognizer.adjust_for_ambient_noise(source)
            Log.log("SYSTEM", "Listening for action...")

            while True:
                try:
                    audio = self.recognizer.listen(source, steam=True, phrase_time_limit=1)

                    if self.recognizer_name == Recognizer.GOOGLE:
                        text = self.recognizer.recognize_google(audio).lower()
                        

                    elif self.recognizer_name == Recognizer.SPHINX:
                        text = self.recognizer.recognize_sphinx(audio, keyword_entries=[("lane", 1.0)] ).lower()
                        
                        
                    elif self.recognizer_name == Recognizer.HOUNDIFY:
                        text = self.recognizer.recognize_houndify(audio).lower()
                        
                    print(f"Heard: {text}")
                    
                    if self.wake_word in text:
                        print(f"Wake word '{self.wake_word}' detected!")
                        return True
                    elif self.sleep_word in text:
                        print(f"Sleep word '{self.sleep_word}' detected, stopping...")
                        return False
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    Log.log("ERROR", f"Could not request results from {self.recognizer_name} Recognition service; {e}")


    def listen_with_background_recognition(self, wake_word):
        """
        Continuously record audio and process it in the background to detect the wake word.
        """
        buffer = collections.deque(maxlen=2)  # Circular buffer to store the last 2 seconds of audio
        stop_event = threading.Event()  # Event to signal the background thread to stop

        def record_audio():
            """Continuously record audio and update the buffer."""
            with self.MIC as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening for audio...")
                while not stop_event.is_set():
                    try:
                        # Record 1 second of audio
                        audio_chunk = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
                        buffer.append(audio_chunk)  # Add the chunk to the buffer
                    except sr.WaitTimeoutError:
                        # Handle timeout if no audio is detected
                        continue

        def process_audio():
            """Continuously process audio from the buffer to detect the wake word."""
            while not stop_event.is_set():
                if len(buffer) == 2:  # Ensure we have 2 seconds of audio
                    # Combine the last 2 chunks into one audio segment
                    combined_audio = sr.AudioData(
                        b"".join([chunk.get_raw_data() for chunk in buffer]),
                        buffer[0].sample_rate,
                        buffer[0].sample_width,
                    )

                    try:
                        # Recognize the combined audio
                        if self.recognizer_name == Recognizer.GOOGLE:
                            text = self.recognizer.recognize_google(combined_audio).lower()
                        elif self.recognizer_name == Recognizer.SPHINX:
                            text = self.recognizer.recognize_sphinx(
                                combined_audio, keyword_entries=[(wake_word, 1.0)]
                            ).lower()
                        elif self.recognizer_name == Recognizer.HOUNDIFY:
                            text = self.recognizer.recognize_houndify(combined_audio).lower()

                        # Check if the wake word is in the recognized text
                        print(f"Recognized: {text}")
                        if wake_word in text:
                            print(f"Wake word '{wake_word}' detected!")
                            stop_event.set()  # Signal to stop recording and processing
                            break

                    except sr.UnknownValueError:
                        # Handle cases where the recognizer doesn't understand the audio
                        continue
                    except sr.RequestError as e:
                        # Handle API errors
                        print(f"Could not request results; {e}")
                        stop_event.set()
                        break

        # Start the recording and processing threads
        record_thread = threading.Thread(target=record_audio)
        process_thread = threading.Thread(target=process_audio)

        record_thread.start()
        process_thread.start()

        # Wait for both threads to finish
        record_thread.join()
        process_thread.join()


#== Methods ==#

def find_microphone(device_name: str) -> int:
    # Find Device Name using this
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if device_name in name:
                 return index
            

