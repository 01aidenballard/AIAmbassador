
Train dataset = 461, Test dataset = 14, Classes = 8, 80/20 Train/Val Split

Approach       | Type       | Epoch  | LR   | Loss       | Accuracy | F1     | Avg Resp | Size     |
---------------|------------|--------|------|------------|----------|--------|----------|----------|
Trad. ML       | LR         | 5000   | 0.01 | CE=0.8095  | 85.71%   | 0.8095 | 4.66 ms  | 8 KB
Trad. ML       | SVM        | 5000   | 0.01 | HL=1.5348  | 71.43%   | 0.6905 | 4.40 ms  | 8 KB
Finetine Trans | BERT       | 5      | 2e-5 | CE=0.9709  | 75.00%   | 0.7500 | X.XX ms  | 438.2 MB
Finetine Trans | DistilBERT | 10     | 2e-5 | CE=0.5282  | 75.00%   | 0.7500 | X.XX ms  | 268 KB


HL for SVM = 24.5562, batch size 16
