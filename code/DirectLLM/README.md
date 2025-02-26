# DirectLLM
## Setup
Create a python virtual environment named `venv`
``` 
python3.10 -m venv venv
```
Activate by `source venv/bin/activate`

Install the required libraries (NOT FINISHED)
```
pip3 install 'torch<2.0' transformers datasets evaluate sacrebleu
```

## Python Files
`chatbot.py` - [outdated] attempt to make a single python file to run any model (didnt work). Uses Roberta model and has code to run as interactive chatbot.

`BART.py` - finetune (if not already done) and run BART. Outputs answers to test dataset, F1 score, BLEU score, and average response time.

`flan_tf.py` - finetune (if not already done) and run Flan T5. Outputs answers to test dataset, F1 score, BLEU score, and average response time.