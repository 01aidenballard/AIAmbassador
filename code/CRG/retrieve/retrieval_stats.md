# Retrieval Runs
Dataset information:
- v2 (n=461)
- Test dataset = 8

| Classification | Extraction | Retrieval | Avg RS | Avg RT     |
|----------------|------------|-----------|--------|------------|
| LR             | NER        | EKI       | 0.575  | 0.00009 ms |
|                |            | Jaccard   | 0.525  | 0.58132 ms |
|                |            | JEKI      | 0.500  | 0.39105 ms |
|                | TFIDF      | EKI       | 0.725  | 0.00013 ms |
|                |            | Jaccard   | 0.625  | 0.04869 ms |
|                |            | JEKI      | 0.625  | 0.06073 ms |
|                | vec        | CSS-TFIDF | 0.625  | 0.10124 ms |
|                |            | CSS-vec   | 0.875  | 1.07098 ms |
| SVM            | NER        | EKI       | 0.625  | 0.00011 ms |
|                |            | Jaccard   | 0.500  | 0.57809 ms |
|                |            | JEKI      | 0.625  | 0.63554 ms |
|                | TFIDF      | EKI       | 0.650  | 0.00013 ms |
|                |            | Jaccard   | 0.625  | 0.05274 ms |
|                |            | JEKI      | 0.625  | 0.05876 ms |
|                | vec        | CSS-TFIDF | 0.625  | 0.13332 ms |
|                |            | CSS-vec   | 0.875  | 1.11223 ms |
| BERT           | NER        | EKI       | 0.525  | 0.00010 ms |
|                |            | Jaccard   | 0.575  | 0.60552 ms |
|                |            | JEKI      | 0.625  | 0.57710 ms |
|                | TFIDF      | EKI       | 0.600  | 0.00135 ms | 
|                |            | Jaccard   | 0.625  | 0.06224 ms |
|                |            | JEKI      | 0.625  | 0.04638 ms |
|                | vec        | CSS-TFIDF | 0.625  | 0.08620 ms |
|                |            | CSS-vec   | 0.750  | 0.93663 ms |
| DistilBERT     | NER        | EKI       | 0.475  | 0.00014 ms |
|                |            | Jaccard   | 0.375  | 0.72857 ms |
|                |            | JEKI      | 0.375  | 0.73186 ms |
|                | TFIDF      | EKI       | 0.475  | 0.00011 ms |
|                |            | Jaccard   | 0.500  | 0.05573 ms |
|                |            | JEKI      | 0.500  | 0.04743 ms |
|                | vec        | CSS-TFIDF | 0.500  | 0.10132 ms |
|                |            | CSS-vec   | 0.750  | 1.18178 ms | 

