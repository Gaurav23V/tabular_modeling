# Tabular Modeling - Loan Default Prediction

## Purpose/Description

This project is a machine learning solution for predicting loan default status using tabular data. Built as part of a Kaggle competition (Playground Series S4E10), it implements multiple classification models to predict whether a loan applicant will default based on various financial and personal features.

## Tech Stack

- **Language**: Python
- **ML Frameworks**: TensorFlow, TensorFlow Decision Forests, XGBoost, LightGBM, CatBoost
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (LogisticRegression, KNeighborsClassifier, RandomForestClassifier, VotingClassifier)
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Yellowbrick (ConfusionMatrix, ROCAUC)
- **Imbalanced Data Handling**: imbalanced-learn (SMOTE)

## Implementation Details

### Data Pipeline

The project processes loan application data with the following features:
- **Person Information**: Age, income, home ownership status, employment length
- **Loan Details**: Intent, grade, amount, interest rate, percent of income
- **Credit History**: Default on file status, credit history length
- **Target**: Loan status (0 = no default, 1 = default)

### Models Implemented

1. **Random Forest Classifier** - Ensemble tree-based model
2. **XGBoost** - Gradient boosting framework
3. **LightGBM** - Light gradient boosting machine
4. **CatBoost** - CatBoost classifier for categorical features
5. **TensorFlow Decision Forests** - TensorFlow's random forest implementation
6. **Logistic Regression** - Linear classification baseline
7. **K-Nearest Neighbors** - Instance-based learning classifier
8. **Voting Classifier** - Ensemble of multiple models

### Key Techniques

- **Data Preprocessing**: MinMaxScaler, StandardScaler for feature normalization
- **Label Encoding**: Converting categorical variables to numerical
- **SMOTE**: Handling class imbalance through oversampling
- **Model Evaluation**: ROC-AUC curves, confusion matrices

## Project Structure

```
tabular_modeling/
├── notebook.ipynb          # Main Jupyter notebook with all analysis
├── data/
│   ├── train.csv           # Training dataset (58,645 samples)
│   ├── test.csv            # Test dataset for predictions
│   ├── sample_submission.csv
│   └── playground-series-s4e10.zip
└── catboost_info/          # CatBoost model cache
```

## Installation/Setup

```bash
pip install tensorflow tensorflow-decision-forests
pip install xgboost lightgbm catboost
pip install pandas numpy scikit-learn
pip install imbalanced-learn
pip install yellowbrick matplotlib seaborn
```

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

The notebook contains:
1. Library imports and data loading
2. Exploratory data analysis
3. Data preprocessing and feature engineering
4. Model training and comparison
5. Prediction and submission generation

## Dataset Features

| Feature | Description |
|---------|-------------|
| person_age | Age of the applicant |
| person_income | Annual income |
| person_home_ownership | RENT, OWN, MORTGAGE |
| person_emp_length | Employment length in years |
| loan_intent | Purpose: EDUCATION, MEDICAL, PERSONAL, VENTURE, etc. |
| loan_grade | Loan grade (A-G) |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate |
| loan_percent_income | Loan as percentage of income |
| cb_person_default_on_file | Previous default history (Y/N) |
| cb_person_cred_hist_length | Credit history length |
| loan_status | Target variable (0/1) |