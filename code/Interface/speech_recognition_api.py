import sys
import os
# Ignore useless ALSA warnings
os.environ["ALSA_NO_WARN"] = "1"

import speech_recognition as sr
import time


# Find Device Name using this
#for index, name in enumerate(sr.Microphone.list_microphone_names()):
#    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))


MIC_INDEX = 3 # default

def speech_recognition():
    """
    Function to recognize speech from audio input.
    """
    try:
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Use the microphone as the audio source
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print("Please say something...")
            # Adjust for ambient noise and record audio
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        #TODO: Get a API key as the default is only good for personal/testing purposes 

        # Test multiple recognizers
        # print("Testing Google Web Speech API...")
        # Measure time for Google Web Speech API
        start_time = time.time()
        text = recognizer.recognize_google(audio)
        end_time = time.time()
        print(f"Google Web Speech API: {text} (Time taken: {(end_time - start_time):.2f} s)")

        # Measure time for Sphinx
        #print("Testing Sphinx...")Testing Google Web Speech API...
        #try:
        #    start_time = time.time()
        #    sphinx_text = recognizer.recognize_sphinx(audio)
        #    end_time = time.time()
        #    print(f"Sphinx: {sphinx_text} (Time taken: {(end_time - start_time):.2f} s)")
        #except sr.UnknownValueError:
        #    sphinx_text = "Sphinx could not understand audio"
        #    print(sphinx_text)

        # Measure time for Houndify
        #print("Testing Houndify...")
        #try:
        #    start_time = time.time()
        #    houndify_text = recognizer.recognize_houndify(audio, client_id="QGbfTnsp6zpB8m6yFC4Cfg==", client_key="xsISiTIHIsCKckTShaOay6sBX8zduFibr3v3DhKYDN3UvOMpZTCa66NDw6tFMLRlDW9KGkjtVCWC0l-uX5h_eg==")
        #    end_time = time.time()
        #    print(f"Houndify: {houndify_text} (Time taken: {(end_time - start_time):.2f} s)")
        #except sr.UnknownValueError:
        #    houndify_text = "Houndify could not understand audio"
        #    print(houndify_text)
        #except sr.RequestError as e:
        #    houndify_text = f"Houndify request error: {e}"
        #    print(houndify_text)


    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

    return text



def main():
    speech_recognition()


if __name__ == "__main__":
    main()
