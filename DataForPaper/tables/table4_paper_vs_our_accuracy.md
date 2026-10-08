### Table 4: Reference Paper Baseline vs Our Empirical Test Accuracy

| Dataset Evaluation         | Model                   | Reference Paper Acc   | Our Test Acc   |   Delta (Abs) | Delta (pp)   | Outcome     |
|:---------------------------|:------------------------|:----------------------|:---------------|--------------:|:-------------|:------------|
| ISOT                       | Logistic Regression     | 99.30%                | 99.13%         |       -0.0017 | -0.17 pp     | COMPARABLE  |
| ISOT                       | Multinomial Naive Bayes | 94.90%                | 95.78%         |        0.0088 | +0.88 pp     | BEATS PAPER |
| ISOT                       | Linear SVM              | 99.60%                | 99.76%         |        0.0016 | +0.16 pp     | BEATS PAPER |
| ISOT                       | Decision Tree           | 99.60%                | 99.57%         |       -0.0003 | -0.03 pp     | COMPARABLE  |
| ISOT                       | Random Forest           | 99.80%                | 99.77%         |       -0.0003 | -0.03 pp     | COMPARABLE  |
| LIAR (Default tau=0.50)    | Logistic Regression     | 62.80%                | 62.13%         |       -0.0067 | -0.67 pp     | COMPARABLE  |
| LIAR (Calibrated tau=0.55) | Logistic Regression     | 62.80%                | 62.09%         |       -0.0071 | -0.71 pp     | COMPARABLE  |
| LIAR (Default tau=0.50)    | Multinomial Naive Bayes | 62.30%                | 60.88%         |       -0.0142 | -1.42 pp     | COMPARABLE  |
| LIAR (Calibrated tau=0.55) | Multinomial Naive Bayes | 62.30%                | 61.94%         |       -0.0036 | -0.36 pp     | COMPARABLE  |
| LIAR (Default tau=0.50)    | Linear SVM              | 62.60%                | 62.41%         |       -0.0019 | -0.19 pp     | COMPARABLE  |
| LIAR (Calibrated tau=0.55) | Linear SVM              | 62.60%                | 62.60%         |        0      | +0.00 pp     | BEATS PAPER |
| LIAR (Default tau=0.50)    | Decision Tree           | 57.10%                | 57.05%         |       -0.0005 | -0.05 pp     | COMPARABLE  |
| LIAR (Calibrated tau=0.55) | Decision Tree           | 57.10%                | 57.05%         |       -0.0005 | -0.05 pp     | COMPARABLE  |
| LIAR (Default tau=0.50)    | Random Forest           | 62.50%                | 62.52%         |        0.0002 | +0.02 pp     | BEATS PAPER |
| LIAR (Calibrated tau=0.55) | Random Forest           | 62.50%                | 62.80%         |        0.003  | +0.30 pp     | BEATS PAPER |
