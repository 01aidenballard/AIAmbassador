
Train dataset = 747, Test dataset = 22, Classes = 11, 80/20 Train/Val Split

Approach       | Type       | Epoch  | LR   | Loss       | Accuracy | F1     | Avg Resp | Size     |
---------------|------------|--------|------|------------|----------|--------|----------|----------|
Trad. ML       | LR         | 5000   | 0.01 | CE=0.8095  | 86.36%   | 0.8545 | 4.56 ms  | 8 KB
Trad. ML       | SVM        | 5000   | 0.01 | HL=1.5348  | 77.27%   | 0.7364 | 4.33 ms  | 8 KB
Finetine Trans | BERT       | 5      | 2e-5 | CE=0.9709  | 72.73%   | 0.6242 | 0.53 ms  | 438.2 MB
Finetine Trans | DistilBERT | 10     | 2e-5 | CE=0.5282  | 72.73%   | 0.6398 | 0.32 ms  | 268 KB


HL for SVM = 24.5562, batch size 16
