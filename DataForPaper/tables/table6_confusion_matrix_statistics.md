### Table 6: Confusion Matrix Statistics & Error Distribution

| Dataset           | Model                   |   Threshold |   TP (Real as Real) |   TN (Fake as Fake) |   FP (Fake as Real) |   FN (Real as Fake) | Sensitivity (Recall)   | Specificity   |   Disparity |FP - FN| |
|:------------------|:------------------------|------------:|--------------------:|--------------------:|--------------------:|--------------------:|:-----------------------|:--------------|----------------------:|
| ISOT              | Logistic Regression     |        0.5  |                4224 |                3528 |                  53 |                  15 | 99.65%                 | 98.52%        |                    38 |
| ISOT              | Multinomial Naive Bayes |        0.5  |                4079 |                3411 |                 170 |                 160 | 96.23%                 | 95.25%        |                    10 |
| ISOT              | Linear SVM              |        0.5  |                4235 |                3566 |                  15 |                   4 | 99.91%                 | 99.58%        |                    11 |
| ISOT              | Decision Tree           |        0.5  |                4222 |                3564 |                  17 |                  17 | 99.60%                 | 99.53%        |                     0 |
| ISOT              | Random Forest           |        0.5  |                4233 |                3569 |                  12 |                   6 | 99.86%                 | 99.66%        |                     6 |
| LIAR (Calibrated) | Logistic Regression     |        0.55 |                 958 |                 631 |                 501 |                 469 | 67.13%                 | 55.74%        |                    32 |
| LIAR (Default)    | Logistic Regression     |        0.5  |                1115 |                 475 |                 657 |                 312 | 78.14%                 | 41.96%        |                   345 |
| LIAR (Calibrated) | Multinomial Naive Bayes |        0.55 |                1107 |                 478 |                 654 |                 320 | 77.58%                 | 42.23%        |                   334 |
| LIAR (Default)    | Multinomial Naive Bayes |        0.5  |                1220 |                 338 |                 794 |                 207 | 85.49%                 | 29.86%        |                   587 |
| LIAR (Calibrated) | Linear SVM              |        0.55 |                 956 |                 646 |                 486 |                 471 | 66.99%                 | 57.07%        |                    15 |
| LIAR (Default)    | Linear SVM              |        0.5  |                1121 |                 476 |                 656 |                 306 | 78.56%                 | 42.05%        |                   350 |
| LIAR (Calibrated) | Decision Tree           |        0.5  |                 927 |                 533 |                 599 |                 500 | 64.96%                 | 47.08%        |                    99 |
| LIAR (Default)    | Decision Tree           |        0.5  |                 927 |                 533 |                 599 |                 500 | 64.96%                 | 47.08%        |                    99 |
| LIAR (Calibrated) | Random Forest           |        0.55 |                1023 |                 584 |                 548 |                 404 | 71.69%                 | 51.59%        |                   144 |
| LIAR (Default)    | Random Forest           |        0.5  |                1157 |                 443 |                 689 |                 270 | 81.08%                 | 39.13%        |                   419 |
