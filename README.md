🏡 House Price Prediction Using Machine Learning

A complete end–to–end data science project involving data cleaning, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, model evaluation, and predictions using a Random Forest Regressor.

📌 Project Overview

This project aims to build a machine learning model that can accurately predict housing prices based on important features such as square footage, number of bedrooms, bathrooms, location coordinates, and property characteristics.

The workflow includes:

Data cleaning & preprocessing

Exploratory data analysis (EDA)

Feature scaling

Model training using Random Forest

Hyperparameter tuning

Feature importance analysis

Model evaluation

Saving the model, scaler, and column structure

Predicting new unseen data

This repository contains all the required code, model files, and results.

📂 Project Structure
House-Price-Prediction-Project/
│
├── data/
│   ├── cleaned_dataset.csv
│   ├── (optional) original_dataset.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│
├── notebooks/
│   ├── house_hold.ipynb
│
├── results/
│   ├── actual_vs_predicted.csv
│
├── scripts/
│   ├── predict_new.py   (optional)
│
├── README.md
├── requirements.txt
└── .gitignore

🧹 Data Preparation

The dataset undergoes the following cleaning steps:

Handling missing values

Converting data types

Outlier removal

Scaling numerical features using StandardScaler

Renaming and organizing columns

Preparing train–test splits

The cleaned dataset is stored as:

➡️ cleaned_dataset.csv

📊 Exploratory Data Analysis

The notebook includes visualizations such as:

Distribution plots

Histograms

Boxplots

Correlation heatmap

Scatterplots

Price trends

Location-based patterns (lat/long)

EDA helps identify feature relationships, patterns, and outliers.

🤖 Model Development

A Random Forest Regressor was chosen based on performance after testing multiple algorithms.

✔ Training steps include:

Splitting into train & test

Scaling selected numerical features

Hyperparameter tuning

Cross-validation

Model evaluation metrics

The following files are saved:

best_model.pkl → trained Random Forest model

scaler.pkl → StandardScaler fitted on training data

columns.pkl → ensures correct feature order during prediction

📈 Model Evaluation

The evaluation was performed on test data and results stored in:

➡️ actual_vs_predicted.csv

This file contains:

Actual prices

Predicted prices

Error difference

All original feature values (after inverse scaling)

Metrics used:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

🧠 Feature Importance

Feature importance was extracted from the Random Forest model to identify which factors influence house pricing the most.

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

Random Forest Regression

🏁 Final Output

The project successfully builds a robust model capable of predicting house prices with high accuracy and exports:

A trained Random Forest model

Scaler and column pipeline

Evaluation results

Prediction script

Visual insights through EDA
