import sys
import os

from enum import Enum
import speech_recognition as sr
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
    def __init__(self, wake_word: str = "hey lane", sleep_word: str = "stop", recognizer_name: str = Recognizer.GOOGLE, device_name: str = "USB PnP Sound Device:"):
        """
        Initialize the Speech class with a wake word.
        """
        self.wake_word = wake_word # "Hey Lain"
        self.sleep_word = sleep_word # "Stop" or "Sleep"
        self.recognizer_name = recognizer_name  # Default recognizer name
        self.device_name = device_name


    def listen(self) -> str:
        
        """
        Function to recognize speech from audio input.
        """

        try:
            # Initialize the recognizer
            recognizer = sr.Recognizer()
            
            # Use the microphone as the audio source
            with sr.Microphone(device_index=find_microphone(self.device_name)) as source:
                print("Please say something...")
                # Adjust for ambient noise and record audio
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
        
            #TODO: Get a API key as the default is only good for personal/testing purposes 

            try: 
                if self.recognizer_name == Recognizer.GOOGLE:
                    # Measure time for Google Web Speech API
                    print("Using Google Web Speech API...") 
                    start_time = time.time()
                    text = recognizer.recognize_google(audio)
                    end_time = time.time()
                    print(f"Your Question: {text} (Time taken: {(end_time - start_time):.2f} s)")

                elif self.recognizer_name == Recognizer.SPHINX:
                    # Measure time for Sphinx
                    print("Using Sphinx...")
                    start_time = time.time()
                    sphinx_text = recognizer.recognize_sphinx(audio)
                    end_time = time.time()
                    print(f"Your Question: {sphinx_text} (Time taken: {(end_time - start_time):.2f} s)")

                elif self.recognizer_name == Recognizer.HOUNDIFY:
                    #Measure time for Houndify
                    print("Using Houndify...")
                    start_time = time.time()
                    houndify_text = recognizer.recognize_houndify(audio, client_id="QGbfTnsp6zpB8m6yFC4Cfg==", client_key="xsISiTIHIsCKckTShaOay6sBX8zduFibr3v3DhKYDN3UvOMpZTCa66NDw6tFMLRlDW9KGkjtVCWC0l-uX5h_eg==")
                    end_time = time.time()
                    print(f"Your Question: {houndify_text} (Time taken: {(end_time - start_time):.2f} s)")
            

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
        recognizer = sr.Recognizer()
        mic = sr.Microphone(device_index=find_microphone(self.device_name))

        text = ""
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            Log.log("SYSTEM", "Listening for action...")

            while True:
                try:
                    audio = recognizer.listen(source)

                    if self.recognizer_name == Recognizer.GOOGLE:
                        text = recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                    elif self.recognizer_name == Recognizer.SPHINX:
                        text = recognizer.recognize_sphinx(audio).lower()
                        print(f"Heard: {text}")
                        
                    elif self.recognizer_name == Recognizer.HOUNDIFY:
                        text = recognizer.recognize_houndify(audio).lower()
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


#== Methods ==#

def find_microphone(device_name: str) -> int:
    # Find Device Name using this
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if device_name in name:
                 return index
            

