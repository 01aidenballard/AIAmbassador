# Results Setup
Below is a table showing CRG configurations to use during testing.

| CRG Model | Classification | Extraction | Retrieval |
|-----------|----------------|------------|-----------|
| CRG-1     | LR             | Vector     | CS-VEC    |
| CRG-2     | SVM            | NER        | EKI       |
| CRG-3     | DistilBERT     | Vector     | CS-VEC    |

Train dataset: 750 QA Pairs (04/28/2025)

Test dataset (CRG add 5 INIT at beginning): 
```
test_dataset = {'data': [
    {'question': 'What degree programs does the department offer?', 'label': 'Degree Programs'},
    {'question': 'What dual degrees can I pursue?', 'label': 'Degree Programs'},
    {'question': 'What are the various research areas in the Lane Department?', 'label': 'Research Opportunities'},
    {'question': 'What research is done in the biometrics field?', 'label': 'Research Opportunities'},
    {'question': 'What are the student orgs I can join as a LCSEE student?', 'label': 'Clubs and Organizations'},
    {'question': 'What kind of activities do CyberWVU students do?', 'label': 'Clubs and Organizations'},
    {'question': 'What can I do with a computer engineering degree?', 'label': 'Career Opportunities'},
    {'question': 'What can I do with a computer science degree?', 'label': 'Career Opportunities'},
    {'question': 'What type of internships do students get?', 'label': 'Internships'},
    {'question': 'How can students get internships?', 'label': 'Internships'},
    {'question': 'What type of scholarships are available for incoming students?', 'label': 'Financial Aid and Scholarships'},
    {'question': 'How can freshmen get scholarships?', 'label': 'Financial Aid and Scholarships'},
    {'question': 'How can I get into contact with the Lane Department?', 'label': 'Location and Contact'},
    {'question': 'Where is the Lane Department Located?', 'label': 'Location and Contact'},
]}
```

# Results Table
## MacBook 
| Model | Acc (%) | RS     | Avg. Resp. Time (s) | Avg. CPU (%) | Avg. RAM (MB) |
|-------|---------|--------|---------------------|--------------|---------------|
| CRG-1 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |
| CRG-2 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |
| CRG-3 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |

## Raspberry Pi 4B
| Model | Acc (%) | RS     | Avg. Resp. Time (s) | Avg. CPU (%) | Avg. RAM (MB) |
|-------|---------|--------|---------------------|--------------|---------------|
| CRG-1 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |
| CRG-2 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |
| CRG-3 | 00.000  | 0.0000 | 00.0000             | 00.000       | 000.000       |