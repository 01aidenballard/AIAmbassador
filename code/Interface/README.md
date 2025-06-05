# Interface

**conversate.py**
: API with our models, follow documentation in code/CRG for more information. This includes the list of compatible/best pipelines. Output speech using CMU's Festival Lite CLI.

**speech_recognition_api.py**
: Utilizes Uberi's SpeechRecognition library to interface Google's Web Speech Recognition tool. No parameters, but make sure the program checks for the correct input device. Audio is activated and listened to in this program, then text is output to where the API is called.
