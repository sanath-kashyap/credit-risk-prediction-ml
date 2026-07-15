# Logistic Regression for Credit Risk Assessment

## Project Objective
Predicting credit default is a classic financial problem where identifying high-risk borrowers is essential for capital preservation. This project builds a binary classification pipeline using Python and `scikit-learn` to predict `loan_status` (1 = default, 0 = fully paid) using loan-level data, analyzing features like loan-to-income ratio, interest rates, and historical credit scores to quantify baseline risk boundaries.

## Model Performance & Diagnostics
After implementing feature scaling and training a baseline Logistic Regression model, the evaluation yielded the following quantitative metrics on the test data:

* **Overall Accuracy:** **83.24%** — The model correctly classifies approximately 83% of total borrowers in the test set.
* **ROC-AUC Score:** **0.84** — Demonstrates a strong mathematical ability to distinguish between default risk probabilities and creditworthy applicants.
* **Default Recall (Class 1):** **40.00%** — The model successfully flags only 40% of actual default events, missing 1,109 out of 1,861 real defaults.

## Key Insights & Structural Limitations
1. **The Class Imbalance Trap:** While an overall accuracy of 83.24% appears high on paper, it is heavily inflated by the dataset's underlying class imbalance (a vast majority of corporate observations are non-defaults). The model inherently skews toward predicting "No Default."
2. **Lending Risk Exposure:** In consumer credit, missing an active default (a False Negative) is substantially costlier than turning down a qualified borrower (a False Positive). Because the baseline model's recall sits at 40%, it leaves the institution exposed by letting 60% of defaults slip through undetected.
3. **Strategic Improvement Plan:** To resolve this weakness in production, the next iteration introduces hyperparameter modifications—specifically setting `class_weight='balanced'` inside the `LogisticRegression()` instantiation. This penalizes missed defaults heavily during optimization, lifting default recall to protect lending capital at a minor, acceptable trade-off to overall accuracy.

## Tech Stack
* **Language:** Python
* **Machine Learning:** `scikit-learn` (LogisticRegression, train_test_split, StandardScaler, classification_report)
* **Data Engineering & Analytics:** `pandas`, `numpy`
* **Visualization:** `seaborn`, `matplotlib`
