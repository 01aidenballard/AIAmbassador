# AI Ambassador
Implementation of an AI-powered robotic tour guide system for the Lane Department
of Computer Science and Electrical Engineering (LCSEE) at West Virginia University. 
Two methodologies were explored: 
1. DirectLLM approach that fine-tunes a large language model on
a custom dataset
2. Classify-Retrieve-Generate (CRG) pipeline that modularizes classification, answer retrieval, and natural response generation. 

A custom SQuAD-style dataset was developed using LCSEE data, supporting both pipelines. The system was deployed on a Raspberry Pi 4 integrated with a Mang-Dang Mini Pupper robot. 

![LCSEE Logo](report/assets/lcsee_logo.png)

## Implementation
The system diagram can be seen below. A user first asks a question to the MangDang robot and an on board microphone will take the speech input and transcribe it using [INSERT]. The question is then passed either the DirectLLM model or the CRG model to generate an answer for the user. The user will then hear the answer played back via an onboard speaker on the MangDang robot.

![System Diagram](report/assets/system_diagram.png)

## Dataset
A custom SQuAD-style dataset was created to support both models, consisting of 747 Q&A pairs across 10 categories:

Degree Programs,
Research Opportunities,
Facilities and Resources,
Clubs and Organizations,
Career Opportunities,
Internships,
Financial Aid and Scholarships,
Faculty Info,
Admissions,
Contact/Location

Data augmentation was performed using ChatGPT to enhance variety and generalization.

## DirectLLM

## Classify-Retrieve-Generate

## Project Directory & Resources 
**Folders**
- `code` - development and production code
  - `DirectLLM` - v1.0 of model. A fine-tuned LLM processes the question directly and generates an answer in one step
  - `CRG` - Classify-Retrieve-Generate, A classifier categorizes the question, retrieves relevant information from a database, and an LLM refines the response for a natural output.
  - `Interface` - User interaction between the Mini Pupper and the LLMs.
- `report` - report tex files and associated assets

**Resources** - [Google Drive](https://drive.google.com/drive/u/0/folders/0ACyJj38rAVkhUk9PVA) includes resources and various documentation
