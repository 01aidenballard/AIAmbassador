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

    #TODO: Fix the recording so that it doesn just send audio whenver and records specifically 0.5 seconds.
    def listen_with_background_recognition(self):
        """
        Use the `listen_in_background` method to continuously record audio and process it
        in the background to detect the wake word.
        """
        def callback(recognizer, audio):
            """
            Callback function to process audio in the background.
            """
            global actively_listening

            try:
                # Recognize the audio
                if self.recognizer_name == Recognizer.GOOGLE:
                    text = recognizer.recognize_google(audio).lower()
                elif self.recognizer_name == Recognizer.SPHINX:
                    text = recognizer.recognize_sphinx(
                        audio, keyword_entries=[(self.wake_word, 1.0)]
                    ).lower()
                elif self.recognizer_name == Recognizer.HOUNDIFY:
                    text = recognizer.recognize_houndify(audio).lower()

                # Check if the wake word is in the recognized text
                print(f"Recognized: {text}")
                if self.wake_word in text:
                    Log.log("INFO", "Wake Word '{self.wake_word}' detected!")
                    stop_listening(wait_for_stop=False)  # Stop background listening
                    actively_listening = False # stop the main loop
                
            except sr.UnknownValueError:
                # Handle cases where the recognizer doesn't understand the audio
                pass
            except sr.RequestError as e:
                # Handle API errors
                Log.log("ERROR", "Callback function could not request results; {e}")

        # Start listening in the background
        global actively_listening 
        actively_listening = True

        Log.log("SYSTEM", "Listening in the background for the wake word...")
        stop_listening = self.recognizer.listen_in_background(self.MIC, callback)

        try:
            # Keep the main thread alive while background listening is active
            while actively_listening:
                time.sleep(0.1)

        except KeyboardInterrupt:
            Log.log("SYSTEM", "KeyboardInterrupt detected. Stopping background listening...")
            stop_listening(wait_for_stop=False)



#== Methods ==#

def find_microphone(device_name: str) -> int:
    # Find Device Name using this
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if device_name in name:
                 return index
            

#=== Legacy Code ===#
    #== Old Wake Word Engine ==#
    # def listen_for_action_word(self):
    #     """
    #     Continuously listen for the wake word.
    #     """
    #     self.recognizer.pause_threshold = 0.8 # default

    #     text = ""
    #     with self.MIC as source:
    #         self.recognizer.adjust_for_ambient_noise(source)
    #         Log.log("SYSTEM", "Listening for action...")

    #         while True:
    #             try:
    #                 audio = self.recognizer.listen(source, steam=True, phrase_time_limit=1)

    #                 if self.recognizer_name == Recognizer.GOOGLE:
    #                     text = self.recognizer.recognize_google(audio).lower()
                        

    #                 elif self.recognizer_name == Recognizer.SPHINX:
    #                     text = self.recognizer.recognize_sphinx(audio, keyword_entries=[("lane", 1.0)] ).lower()
                        
                        
    #                 elif self.recognizer_name == Recognizer.HOUNDIFY:
    #                     text = self.recognizer.recognize_houndify(audio).lower()
                        
    #                 print(f"Heard: {text}")
                    
    #                 if self.wake_word in text:
    #                     print(f"Wake word '{self.wake_word}' detected!")
    #                     return True
    #                 elif self.sleep_word in text:
    #                     print(f"Sleep word '{self.sleep_word}' detected, stopping...")
    #                     return False
    #             except sr.UnknownValueError:
    #                 continue
    #             except sr.RequestError as e:
    #                 Log.log("ERROR", f"Could not request results from {self.recognizer_name} Recognition service; {e}")