# Job Acceptance Prediction System

## 📌 Project Overview

The Job Acceptance Prediction System is a Machine Learning project that analyzes candidate placement data and predicts job placement status.

The system uses candidate academic performance, technical skills, communication skills, interview score, skills match, company tier, competition level and other placement-related factors.

## 🎯 Objectives

- Analyze candidate placement data
- Clean and preprocess real-world data
- Perform Exploratory Data Analysis (EDA)
- Identify important placement-related factors
- Build a Machine Learning classification model
- Predict candidate placement status
- Provide an interactive Streamlit dashboard

## 📊 Dataset

The cleaned dataset contains **40,646 candidate records**.

The dataset includes information such as:

- Age
- Gender
- Academic Performance
- Technical Score
- Communication Score
- Aptitude Score
- Coding Score
- Interview Score
- Skills Match Score
- Internship
- Certification
- Projects Count
- Company Tier
- Competition Level
- Offered Salary
- Expected Salary
- Placement Status

## 🧹 Data Preprocessing

The project includes:

- Missing value handling
- Duplicate removal
- Categorical data standardization
- Numerical data preprocessing
- Feature engineering
- One-hot encoding

## 🤖 Machine Learning

Two classification models were evaluated:

- Logistic Regression
- Random Forest Classifier

The Random Forest model was selected for the Streamlit prediction system.

### Random Forest Performance

- Accuracy: 88.39%
- Precision: 85.86%
- Recall: 73.59%
- F1 Score: 79.25%

## 📈 Exploratory Data Analysis

The project analyzes:

- Interview Score vs Placement
- Skills Match vs Placement
- Company Tier vs Placement
- Competition Level vs Placement
- Academic Performance vs Placement
- Numerical feature relationships

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides:

- Candidate dataset preview
- Placement statistics
- Placement charts
- Company tier analysis
- Skills match analysis
- Competition analysis
- Candidate job placement prediction

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Random Forest
- Streamlit

## ▶️ How to Run

Install the required packages:

```bash
pip install -r requirements.txt
