# House Prices Prediction

A machine learning project that predicts house prices using demographic and geographic features from the California Housing dataset.

## Dataset
- Source: `sklearn.datasets.fetch_california_housing`
- Target: `median_house_value`
- Features include income, population, housing age, and location data.

## Approach
- Exploratory data analysis
- Stratified train-test split based on income categories
- Data preprocessing (missing values, scaling, one-hot encoding)
- Feature engineering
- Model training and evaluation

## Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation Metric
- RMSE (Root Mean Squared Error)

## Result
Random Forest achieved the best generalization performance compared to baseline models.

## Tech Stack
Python, NumPy, Pandas, Matplotlib, Scikit-learn
