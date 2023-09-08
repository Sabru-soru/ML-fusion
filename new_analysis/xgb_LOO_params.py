import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

# Check the number of unique combinations of angle, heat, field, and emission
unique_combinations = df[['angle', 'heat', 'field', 'emission']].drop_duplicates()
num_unique_combinations = unique_combinations.shape[0]

# Check the number of unique x_m values
num_unique_x_m = df['x_m'].nunique()

num_unique_combinations, num_unique_x_m

# Separate features and target variable
X = df[['angle', 'heat', 'field', 'emission', 'x_m']]
y = df['Pot']

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Initialize lists to store results
rf_mape_scores = []
xgb_mape_scores = []

# Iterate through each unique combination for "leave-one-curve-out" validation
for index, row in unique_combinations.iterrows():
    # Filter data for training and testing based on the unique combination
    test_condition = (df['angle'] == row['angle']) & (df['heat'] == row['heat']) & \
                     (df['field'] == row['field']) & (df['emission'] == row['emission'])
    
    X_train = X.loc[~test_condition]
    y_train = y.loc[~test_condition]
    X_test = X.loc[test_condition]
    y_test = y.loc[test_condition]
    
    # Grid search for Random Forest
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, 
                                  cv=5, n_jobs=-1, verbose=0)
    rf_grid_search.fit(X_train, y_train)
    rf_best_params = rf_grid_search.best_params_
    rf_pred = rf_grid_search.predict(X_test)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
    rf_mape_scores.append({'MAPE': rf_mape, 'Params': rf_best_params})
    
    # Grid search for XGBoost
    xgb_grid_search = GridSearchCV(XGBRegressor(), xgb_param_grid, 
                                   cv=5, n_jobs=-1, verbose=0)
    xgb_grid_search.fit(X_train, y_train)
    xgb_best_params = xgb_grid_search.best_params_
    xgb_pred = xgb_grid_search.predict(X_test)
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)
    xgb_mape_scores.append({'MAPE': xgb_mape, 'Params': xgb_best_params})

# Show some results
rf_mape_scores[:3], xgb_mape_scores[:3]