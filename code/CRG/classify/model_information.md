# Classify Model Information
## LR + TD-IDF
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

## SVM + TD-IDF
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
| 1     | No log     | 1.387188        |
| 2     | 1.365000      | 1.206351        |
| 3     | 1.135700      | 1.100538        |
| 4     | 1.135700      | 1.051871        |
| 5     | 0.970900      | 1.018002        |

Results:
- Accuracy = 75.00%
- F1 Score = 0.7500
- Avg Resp Time = 0.03 ms
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

