# Classify Model Information
- [Classify Model Information](#classify-model-information)
  - [LR + TF-IDF](#lr--tf-idf)
    - [Model Train 03/02/2025](#model-train-03022025)
    - [Model Train 03/10/2025](#model-train-03102025)
  - [SVM + TF-IDF](#svm--tf-idf)
    - [Model Train 03/04/2025](#model-train-03042025)
    - [Model Train 03/10/2025](#model-train-03102025-1)
  - [BERT](#bert)
    - [Model Train 03/05/2025](#model-train-03052025)
    - [Model Train 03/11/2025](#model-train-03112025)
  - [DistilBERT](#distilbert)
    - [Model Train 03/09/2025](#model-train-03092025)
    - [Model Train 03/09/2025](#model-train-03092025-1)
    - [Model Train 03/11/2025](#model-train-03112025-1)

## LR + TF-IDF
### Model Train 03/02/2025
Training:
- Train dataset = 263
- Test dataset = 8
- Classes = 4
- Epoch = 100

| Epoch |  Loss  |
|-------|--------|
| 10    | 1.3070 |
| 20    | 1.2141 |
| 30    | 1.1236 |
| 40    | 1.0436 |
| 50    | 0.9795 |
| 60    | 0.9314 |
| 70    | 0.8959 |
| 80    | 0.8696 |
| 90    | 0.8499 |
| 100   | 0.8346 |

Results:
- Accuracy: 100.00%
- F1 = 1.0000
- Avg Resp Time = 1.44 ms
- Model Size = 8 KB

### Model Train 03/10/2025
Training:
- Train dataset = 421
- Test dataset = 8
- Classes = 8
- Epoch = 100

| Epoch |  Loss  |
|-------|--------|
| 10    | 2.0252 |
| 20    | 1.9481 |
| 30    | 1.8560 |
| 40    | 1.7601 |
| 50    | 1.6707 |
| 60    | 1.5940 |
| 70    | 1.5333 |
| 80    | 1.4866 |
| 90    | 1.4506 |
| 100   | 1.4228 |

Results:
- Accuracy: 87.50%
- F1 = 0.9167
- Avg Resp Time = 2.47 ms
- Model Size = 20 KB

## SVM + TF-IDF
### Model Train 03/04/2025
Training:
- Train dataset = 263
- Test dataset = 8
- Classes = 4
- Epoch = 100
- Batch size = 8

| Epoch |  Loss  |
|-------|--------|
| 10    | 0.7500 |
| 20    | 0.7500 |
| 30    | 0.7500 |
| 40    | 0.7500 |
| 50    | 0.7500 |
| 60    | 0.7500 |
| 70    | 0.7500 |
| 80    | 0.7500 |
| 90    | 0.7500 |
| 100   | 0.7500 |

Results:
- Accuracy: 100.00%
- F1 = 1.0000
- Avg Resp Time = 1.28 ms
- Model Size = 8 KB

### Model Train 03/10/2025
Training:
- Train dataset = 461
- Test dataset = 8
- Classes = 8
- Epoch = 100
- Batch size = 8

| Epoch |  Loss  |
|-------|--------|
| 10    | 0.8750 |
| 20    | 0.8750 |
| 30    | 0.8750 |
| 40    | 0.8750 |
| 50    | 0.8750 |
| 60    | 0.8750 |
| 70    | 0.8750 |
| 80    | 0.8750 |
| 90    | 0.8750 |
| 100   | 0.8750 |

Results:
- Accuracy: 87.50%
- F1 = 0.9167
- Avg Resp Time = 2.19 ms
- Model Size = 20 KB

## BERT
### Model Train 03/05/2025
Training:
- Train dataset = 263
- 80/20 train/validation split
- Test dataset = 8
- Classes = 4
- Epoch = 5
- Batch size = 8

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | No log        | 1.387188        |
| 2     | 1.365000      | 1.206351        |
| 3     | 1.135700      | 1.100538        |
| 4     | 1.135700      | 1.051871        |
| 5     | 0.970900      | 1.018002        |

Results:
- Accuracy = 75.00%
- F1 Score = 0.7500
- Avg Resp Time = 0.03 ms
- Model Size = 438.2 MB

### Model Train 03/11/2025
Training:
- Train dataset = 421
- 80/20 train/validation split
- Test dataset = 8
- Classes = 8
- Epoch = 10
- Batch size = 8

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 2.109100      | 1.970921        |
| 2     | 1.956100      | 1.893361        |
| 3     | 1.840700      | 1.801788        |
| 4     | 1.728300      | 1.751369        |
| 5     | 1.633400      | 1.643540        |
| 6     | 1.504200      | 1.547681        |
| 7     | 1.519800      | 1.492157        |
| 8     | 1.338400      | 1.394458        |
| 9     | 1.333500      | 1.326182        |
| 10    | 1.241100      | 1.309167        |

Results:
- Accuracy = 75.00%
- F1 Score = 0.7500
- Avg Resp Time = 0.15 ms
- Model Size = 438.2 MB

## DistilBERT
### Model Train 03/09/2025
Training:
- Train dataset = 263
- 80/20 train/validation split
- Test dataset = 8
- Classes = 4
- Epoch = 5
- Batch size = 8

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | No log        | 1.372388        |
| 2     | 1.348400      | 1.308466        |
| 3     | 1.212000      | 1.262107        |
| 4     | 1.212000      | 1.209039        |
| 5     | 1.146100      | 1.181725        |

Results:
- Accuracy = 62.75%
- F1 Score = 0.5333
- Avg Resp Time = 0.10 ms
- Model Size = 246 KB

### Model Train 03/09/2025
Training:
- Train dataset = 263
- 80/20 train/validation split
- Test dataset = 8
- Classes = 4
- Epoch = 10
- Batch size = 8

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | No log        | 1.353241        |
| 2     | 1.364000      | 1.263629        |
| 3     | 1.212700      | 1.158168        |
| 4     | 1.212700      | 1.029161        |
| 5     | 1.064800      | 0.904184        |
| 6     | 0.827200      | 0.806753        |
| 7     | 0.827200      | 0.735427        |
| 8     | 0.681500      | 0.675407        |
| 9     | 0.588500      | 0.643484        |
| 10    | 0.528200      | 0.631965        |

Results:
- Accuracy = 75.00%
- F1 Score = 0.6667
- Avg Resp Time = 0.09 ms
- Model Size = 268 KB

### Model Train 03/11/2025
Training:
- Train dataset = 461
- 80/20 train/validation split
- Test dataset = 8
- Classes = 8
- Epoch = 10
- Batch size = 8

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 2.061700      | 1.987463        |
| 2     | 1.911400      | 1.859253        |
| 3     | 1.743100      | 1.724385        |
| 4     | 1.513700      | 1.573582        |
| 5     | 1.369300      | 1.437045        |
| 6     | 1.184400      | 1.322064        |
| 7     | 1.073800      | 1.238781        |
| 8     | 0.959300      | 1.168174        |
| 9     | 0.897400      | 1.129166        |
| 1     | 0.824900      | 1.119619        |

Results:
- Accuracy = 75.00%
- F1 Score = 0.6667
- Avg Resp Time = 0.21 ms
- Model Size = 268 KB
