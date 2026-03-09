Predictive Modeling of Genetic Disorders & Subclasses

Executive Summary

This project features a robust, end-to-end machine learning pipeline designed to predict the presence of primary genetic disorders (3 classes) and granular subclasses (9 classes).

The Business Value: Designed to assist medical diagnostic processes, this pipeline successfully tackles the reality of messy, real-world clinical data—specifically addressing severe class imbalance, missing values (>50% in some features), and mixed data types to uncover hard-to-detect hereditary patterns.

Technical Highlights & Architecture

Employers, here is a quick look at the problem-solving approaches utilized in this project:

Handling Extreme Imbalance (ADASYN): Real-world medical data is rarely balanced. Standard tree models completely failed (0% recall) on rare diseases like Alzheimer's and Cancer. I implemented ADASYN (Adaptive Synthetic Sampling) to intelligently generate synthetic data along the decision boundaries of these "hard-to-learn" minority classes, successfully reviving minority class detection.

Advanced Data Imputation (MICE): Rather than blindly dropping rows or filling with means, I architected a pipeline using IterativeImputer. This models missing biological features as a function of others, preserving crucial underlying biological correlations.

Domain-Driven Feature Engineering: Engineered bespoke features like Genetic Risk Scores and non-linear age bins to capture complex, multi-variable biological interactions that base algorithms struggle to find independently.

"Super Learner" Stacking Ensemble: Built a meta-learning architecture utilizing Logistic Regression to intelligently combine the distinct predictive strengths of XGBoost, CatBoost, and SVC, resulting in a highly generalized final model.

Production-Ready Safeguards: Engineered defense-in-depth data sanitization pipelines. The prediction script includes runtime safety nets to sanitize incoming test batches, preventing pipeline crashes caused by unexpected NaN values or shape mismatches in production environments.

Tech Stack

Core & Data Manipulation: Python, Pandas, NumPy

Machine Learning & Modeling: Scikit-Learn, XGBoost, CatBoost

Sampling & Imputation: Imbalanced-Learn (imblearn), MICE

Evaluation & Visualization: Matplotlib, Seaborn (Custom Confusion Matrices, SHAP-ready)

📊 Impact & Results

The ensemble approach successfully navigated the critical medical trade-off between global accuracy and minority class detection:

Subclass Prediction: Achieved the highest generalized F1-score (~0.34) on the highly complex, heavily overlapping 9-class Subclass prediction task.

Minority Recall Breakthrough: Improved minority class recall (e.g., Alzheimer's, Cancer) from 0.0% in baseline models to ~4.4% through strategic ADASYN integration and Stacking, proving the model's capability to flag rare anomalies.
