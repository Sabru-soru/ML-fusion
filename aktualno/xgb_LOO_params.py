"""
Script to find the best hyperparameters for XGBoost using "leave-one-curve-out" validation.
The best parameters are saved and used in xgb_model.py.
"""

#%%
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import numpy as np
import itertools
import matplotlib.pyplot as plt

file_path = 'data/data_cleaned_sparse_all.xlsx'
df = pd.read_excel(file_path)

# Check the number of unique combinations of angle, heat, field, and emission
unique_combinations = df[['angle', 'heat', 'field', 'emission']].drop_duplicates()
# Check the number of unique x_m values
num_unique_x_m = df['x_m'].nunique()

# Separate features and target variable
X = df[['angle', 'heat', 'field', 'emission', 'x_m']]
# y = df['Pot']
y = df['Tn']


#%%
# Reduced hyperparameter grid for XGBoost for faster computation
#could increase this search space
xgb_param_grid_reduced = {
    'n_estimators': [300, 500, 600, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_param_grid_reduced = {
    'n_estimators': [300],
    'max_depth': [7],
    'learning_rate': [0.1],
    'subsample': [1],
    'colsample_bytree': [1]
}

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(*xgb_param_grid_reduced.values()))
hyperparameter_keys = xgb_param_grid_reduced.keys()

print(f'Number of hyperparameter combinations: {len(hyperparameter_combinations)}')

#%%
# Initialize list to store overall results
overall_results = []
i=0
# Iterate through each hyperparameter combination
for hyperparameters in hyperparameter_combinations:
    print(f'Iteration: {i}')
    i+=1
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
        
        #if y_test is 0 set it to 0.0001
        y_test[y_test==0]=0.0001

        # Train XGBoost model with the current hyperparameter combination
        xgb_model = XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Calculate MAPE for the test set
        mape = mean_absolute_percentage_error(y_test.iloc[6:-6].reset_index(drop=True), xgb_pred[6:-6])*100
        mape_scores.append(mape)

        # plot the results
        plt.figure()
        plt.plot(y_test.iloc[6:-6].reset_index(drop=True), label='actual')
        plt.plot(xgb_pred[6:-6], label='predicted')
        plt.legend()
        plt.show()
    
    # Calculate average MAPE across all test sets
    avg_mape = np.mean(mape_scores)
    
    # Store the results
    overall_results.append({'Params': params, 'Avg_MAPE': avg_mape})

#%%
# Sort the results by Avg_MAPE to find the best hyperparameters
overall_results_sorted = sorted(overall_results, key=lambda x: x['Avg_MAPE'])

# Show the best hyperparameters based on lowest Avg_MAPE
overall_results_sorted[0]
# Save the results to a CSV file
pd.DataFrame(overall_results_sorted).to_csv('data/xgb_LOO_results_Tn.csv', index=False)
# %%
