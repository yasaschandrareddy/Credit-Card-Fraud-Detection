# Credit-Card-Fraud-Detection

## Project Overview

The **Credit-Card-Fraud-Detection** is a machine learning-based solution designed to detect fraudulent transactions in financial data. This system uses various classification models to predict whether a transaction is fraudulent or not, based on historical transaction data. It leverages data preprocessing techniques, handles imbalanced datasets using **SMOTE (Synthetic Minority Over-sampling Technique)**, and evaluates model performance using standard classification metrics.

### Key Features:
- Data preprocessing (handling missing values, feature encoding, and normalization).
- Handling imbalanced datasets using **SMOTE**.
- Multiple machine learning models: Logistic Regression, Decision Tree, Random Forest.
- Evaluation using metrics like accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.
- Visualizations for fraud distribution, model performance, and evaluation metrics.

## Features

### 1. Data Preprocessing
The dataset is cleaned using various preprocessing steps:

- **Missing Data Handling**: Missing values are identified and handled appropriately.
- **Feature Engineering**: Features such as `gender` are transformed, and unneeded features are removed.
- **Normalization**: Numerical features, such as transaction amounts, are normalized for better model performance.
- **Categorical Encoding**: The `gender` feature is encoded with binary values (`1` for female, `0` for male).

### 2. SMOTE for Handling Class Imbalance
Since the dataset is likely imbalanced (more non-fraudulent transactions), **SMOTE** is used to generate synthetic samples for the minority class (fraudulent transactions). This improves the model's ability to detect fraud.

### 3. Machine Learning Models
The following models are trained and evaluated on the dataset:

- **Logistic Regression**: A simple linear model used for binary classification.
- **Decision Tree Classifier**: A non-linear model used for classification tasks.
- **Random Forest Classifier**: An ensemble model that combines multiple decision trees to improve performance.

Each model is evaluated using standard classification metrics and cross-validation.

## Models

### 1. Logistic Regression
A linear model used to predict the probability of the target class. It is simple, fast, and interpretable but might not perform well on complex datasets.

### 2. Decision Tree Classifier
A non-linear classifier that splits the dataset based on feature values. It is easy to interpret but prone to overfitting.

### 3. Random Forest Classifier
An ensemble method that builds multiple decision trees and combines their predictions. It is less prone to overfitting than individual decision trees and usually provides better results.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positives among all positive predictions.
- **Recall**: The proportion of true positives among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, giving a balanced measure of performance.
- **Confusion Matrix**: A matrix showing the true and predicted classifications.

