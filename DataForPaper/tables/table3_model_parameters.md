### Table 3: Model Hyperparameters and Specifications

| Model Family            | Algorithm / Class                       | ISOT Hyperparameters                                           | LIAR Hyperparameters                                                   | Calibration / Threshold               |
|:------------------------|:----------------------------------------|:---------------------------------------------------------------|:-----------------------------------------------------------------------|:--------------------------------------|
| Logistic Regression     | sklearn.linear_model.LogisticRegression | C=1.0, solver='lbfgs', max_iter=1000, random_state=42          | C=1.0, solver='liblinear', max_iter=1000, random_state=42              | Default (tau=0.50) / Tuned (tau=0.55) |
| Multinomial Naive Bayes | sklearn.naive_bayes.MultinomialNB       | alpha=1.0, fit_prior=True                                      | alpha=1.0, fit_prior=True                                              | Default (tau=0.50) / Tuned (tau=0.55) |
| Linear SVM              | sklearn.svm.LinearSVC                   | C=1.0, penalty='l2', loss='squared_hinge', max_iter=2000       | CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000), cv=3)          | Sigmoid / Tuned (tau=0.55)            |
| Decision Tree           | sklearn.tree.DecisionTreeClassifier     | criterion='gini', splitter='best', random_state=42             | max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42 | Default (tau=0.50)                    |
| Random Forest           | sklearn.ensemble.RandomForestClassifier | n_estimators=100, criterion='gini', random_state=42, n_jobs=-1 | n_estimators=200, max_depth=None, min_samples_split=5, random_state=42 | Default (tau=0.50) / Tuned (tau=0.55) |
