import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import itertools

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


# Reduced hyperparameter grid for XGBoost for faster computation
xgb_param_grid_reduced = {
    'n_estimators': [100, 600, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(*xgb_param_grid_reduced.values()))
hyperparameter_keys = xgb_param_grid_reduced.keys()

print(f'Number of hyperparameter combinations: {len(hyperparameter_combinations)}')

# Initialize list to store overall results
overall_results = []

# Iterate through each hyperparameter combination
for hyperparameters in hyperparameter_combinations:
    i=0
    params = dict(zip(hyperparameter_keys, hyperparameters))
    
    # Initialize list to store MAPE scores for each "leave-one-curve-out"
    mape_scores = []
    
    # Iterate through each unique combination for "leave-one-curve-out" validation
    for index, row in unique_combinations.iterrows():
        # Filter data for training and testing based on the unique combination
        test_condition = (df['angle'] == row['angle']) & (df['heat'] == row['heat']) & \
                         (df['field'] == row['field']) & (df['emission'] == row['emission'])
        
        X_train = X.loc[~test_condition]
        y_train = y.loc[~test_condition]
        X_test = X.loc[test_condition]
        y_test = y.loc[test_condition]
        
        # Train XGBoost model with the current hyperparameter combination
        xgb_model = XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Calculate MAPE for the test set
        mape = mean_absolute_percentage_error(y_test, xgb_pred)
        mape_scores.append(mape)
    
    # Calculate average MAPE across all test sets
    avg_mape = np.mean(mape_scores)
    
    # Store the results
    overall_results.append({'Params': params, 'Avg_MAPE': avg_mape})

    print(f'Finished hyperparameter combination: {params}')
    print(f'Avg_MAPE: {avg_mape}')
    print(f'Iteration: {i}')
    i+=1

# Sort the results by Avg_MAPE to find the best hyperparameters
overall_results_sorted = sorted(overall_results, key=lambda x: x['Avg_MAPE'])

# Show the best hyperparameters based on lowest Avg_MAPE
overall_results_sorted[0]