import re
import pyttsx3 as tts


voice = tts.init()
user_input = input("What would you like to say?\n")

def does_text_contain(regex, text, flags=0):
    """
    Checks if a regex pattern is present in a text string.

    Args:
        regex: The regular expression pattern to search for.
        text: The text string to search within.
        flags: Optional flags like re.IGNORECASE for case-insensitive matching.

    Returns:
        True if the regex pattern is found in the text, False otherwise.
    """
    match = re.search(regex, text, flags)
    return bool(match)

if does_text_contain(r"hi|hey|hello", user_input, re.IGNORECASE):
    # L.A.I.N = Lane Artifical Intelligence Navigator:
    voice.say("Hello! My name is LAIN, nice to meet you!")
    voice.runAndWait()
elif does_text_contain(r"name|call you", user_input, re.IGNORECASE):
    voice.say("My name is LAIN.")
    voice.runAndWait()

