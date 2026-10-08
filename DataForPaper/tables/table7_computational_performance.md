### Table 7: Training Time, Inference Latency, and Model Footprint

| Dataset   | Model                   | Training Time (s)   | Single-Sample Latency (ms)   | Batch (1,000) Latency (ms)   | Model Disk Size (MB)   |
|:----------|:------------------------|:--------------------|:-----------------------------|:-----------------------------|:-----------------------|
| ISOT      | Logistic Regression     | 21.13 s             | 0.24 ms                      | 13.01 ms                     | 1.07 MB                |
| ISOT      | Multinomial Naive Bayes | 20.74 s             | 0.25 ms                      | 12.15 ms                     | 1.60 MB                |
| ISOT      | Linear SVM              | 22.97 s             | 0.98 ms                      | 11.32 ms                     | 1.43 MB                |
| ISOT      | Decision Tree           | 42.76 s             | 0.22 ms                      | 10.69 ms                     | 0.94 MB                |
| ISOT      | Random Forest           | 40.97 s             | 29.65 ms                     | 47.51 ms                     | 95.85 MB               |
| LIAR      | Logistic Regression     | 0.032 s             | 0.24 ms                      | 13.01 ms                     | 1.07 MB                |
| LIAR      | Multinomial Naive Bayes | 0.004 s             | 0.25 ms                      | 12.15 ms                     | 1.60 MB                |
| LIAR      | Linear SVM              | 0.049 s             | 0.98 ms                      | 11.32 ms                     | 1.43 MB                |
| LIAR      | Decision Tree           | 0.903 s             | 0.22 ms                      | 10.69 ms                     | 0.94 MB                |
| LIAR      | Random Forest           | 3.144 s             | 29.65 ms                     | 47.51 ms                     | 95.85 MB               |
