# Boston Housing Price Prediction

This project is an open-ended regression challenge based on the Boston Housing dataset.
The goal is to predict the median house price (`medv`) using various socio-economic and
environmental features, and submit the results to a class Kaggle competition.

## Dataset
- `train.csv`: Training data with features and target value (`medv`)
- `test.csv`: Testing data without target value
- Evaluation metric: **RMSE (Root Mean Squared Error)**

## Approach
This project adopts an ensemble (voting) regression strategy combining multiple models:

- **Ridge Regression**
  - RobustScaler for handling outliers
  - Polynomial feature interaction
- **Support Vector Regression (SVR)**
  - RBF kernel
- **XGBoost Regressor**

Different weight combinations are evaluated on a validation set to select the best
ensemble configuration based on RMSE.

## Workflow
1. Load and preprocess training/testing data
2. Split training data into training and validation sets
3. Train individual models (Ridge, SVR, XGBoost)
4. Perform weighted ensemble (voting)
5. Select best weights based on validation RMSE
6. Generate prediction file for Kaggle submission

## Output
- The final prediction file is exported as a CSV file in the format required by Kaggle:
