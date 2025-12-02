# 🏡 House Price Prediction – Machine Learning Project
### 📘 Overview

This project is a complete end-to-end House Price Prediction System built using machine learning.
It demonstrates the full workflow followed in real-world data science projects — from data cleaning to model training, evaluation, and prediction.

The goal of this project is to predict house prices based on features such as number of bedrooms, bathrooms, square footage, location, and more.

### 📁 Project Structure
Section	Description
1️⃣ Data Cleaning & Preprocessing	Cleaned the raw dataset by fixing missing values, converting data types, handling duplicates, and preparing features.
2️⃣ Exploratory Data Analysis (EDA)	Performed statistical summaries and visual exploration to understand patterns, trends, and correlations.
3️⃣ Feature Engineering	Extracted useful features, encoded categorical values, scaled numeric fields, and prepared data for modeling.
4️⃣ Model Training	Trained multiple ML models and evaluated them using metrics like RMSE, MAE, and R² score.
5️⃣ Model Selection	Selected the Random Forest Regressor as the best model and saved it (best_rf_model.pkl).
6️⃣ Model Evaluation	Compared Actual vs Predicted values and computed error metrics. Saved results in actual_vs_predicted.csv.
7️⃣ Deployment-Ready Artifacts	Exported scaler.pkl, columns.pkl, and the trained model for real-world use.
### 📦 Dataset Files
##### File	Description
original_dataset.csv	Raw dataset before cleaning
cleaned_data.csv	Dataset after preprocessing
actual_vs_predicted.csv	Model predictions vs actual values with errors
best_rf_model.pkl	Final trained Random Forest model
scaler.pkl	Scaler used for numeric feature normalization
columns.pkl	Stores the order of feature columns
### 📊 Model Performance
Metric	Value
RMSE	Computed using test data
MAE	Computed using test data
R² Score	Shows how well the model fits the data

### ✅ Best Model: Random Forest Regressor
It delivered the best accuracy and lowest error among the tested models.

### 📉 Actual vs Predicted Analysis

A detailed CSV (actual_vs_predicted.csv) includes:

Actual house prices

Predicted house prices

Absolute error

Percentage error

This helps visualize model performance and identify under- or over-estimations.

### 🧮 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook

Pickle (for model export)

### 🎯 Key Learnings

How to clean and preprocess real-world datasets

How to build and evaluate ML regression models

How to save models and related artifacts for deployment

How to compare actual vs predicted values

How to document a full ML project professionally

### 🚀 How to Use the Project

Clone this repository

Open the Jupyter Notebook

Run all cells to see the complete workflow

Use the saved model files for prediction in external scripts

### 📎 Appendix

This repository contains:

Notebook source code

Cleaned dataset

Trained model files

Actual vs predicted results

Visual charts generated during EDA
