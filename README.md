Credit Card Fraud Detection – Final Year Project

This project focuses on cleaning, analyzing, and modeling transactional data to detect fraudulent credit card activity. The process includes data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model training, and evaluation to identify anomalous transactions in highly imbalanced data.

Project Overview

The primary goal is to transform raw credit card transaction data into a clean, structured format suitable for machine learning. A supervised learning approach is applied to detect fraudulent transactions while minimizing false positives that affect legitimate customers.

The project demonstrates a complete end-to-end data science workflow, from raw data preparation to a trained and evaluated fraud detection model.

Key Components

Fraud_Detection_Dataset.csv: Raw credit card transaction dataset.

df_clean.csv: Cleaned dataset after handling missing values and inconsistencies.

df_encoded.csv: Dataset with categorical variables encoded into numerical format.

df_advanced.csv: Dataset containing advanced engineered features.

fraud_detection_clean.ipynb: Jupyter Notebook documenting EDA and data cleaning.

trainml.py: Python script for training the machine learning model.

evaluation.py: Script containing model evaluation utilities.

lightgbm_champion.pkl: Final trained fraud detection model.

Data Cleaning and Preparation

The fraud_detection_clean.ipynb notebook documents the full preprocessing workflow applied to the raw dataset:

Handling Missing Data: Missing or inconsistent values were cleaned or imputed where necessary.

Data Type Correction: Ensured all numerical and categorical features were in the correct format for modeling.

Encoding Categorical Variables: Converted categorical transaction features into numerical representations.

Feature Engineering: Created advanced features to better capture transactional behavior and anomalies.

Class Imbalance Handling: Prepared the data to address the extreme imbalance between fraudulent and non-fraudulent transactions.

Machine Learning Model

The trainml.py script trains a machine learning model on the engineered dataset:

Loads the processed feature set

Splits data into training and testing sets

Trains the fraud detection model

Saves the final trained model as lightgbm_champion.pkl

The evaluation.py script evaluates model performance using appropriate classification metrics for imbalanced data, such as precision, recall, F1-score, and ROC-AUC.

How to Run

To run the project locally, follow these steps:

Prerequisites

Python 3.x

pip

Setup
git clone https://github.com/bontlemphahlele/credit-card-fraud-final-year-project-75188cd.git
cd credit-card-fraud-final-year-project-75188cd
pip install pandas numpy scikit-learn lightgbm jupyter

Train the Model
python trainml.py

Explore the Data
jupyter notebook


Open fraud_detection_clean.ipynb to review EDA and preprocessing steps.

Outcome

The final model demonstrates strong performance in identifying fraudulent transactions from imbalanced data, making it suitable for further experimentation or deployment in fraud detection systems.
