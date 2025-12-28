from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

# Loading the dataset
housing = pd.read_csv("housing.csv")

# Create stratified test set
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

housing = strat_train_set.copy()

# Separate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# Separate numerical and categorical columns
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

# Making a pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Executing the pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

# Train the model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
print(f'The root mean squared error for Linear Regression is {lin_rmse}')

dec_tree = DecisionTreeRegressor()
dec_tree.fit(housing_prepared, housing_labels)
dec_preds = dec_tree.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_tree, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(dec_rmses).describe())

random_forest = RandomForestRegressor()
random_forest.fit(housing_prepared, housing_labels)
random_preds = random_forest.predict(housing_prepared)
# random_rmse = root_mean_squared_error(housing_labels, random_preds)
random_rmses = -cross_val_score(random_forest, housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(random_rmses).describe())

