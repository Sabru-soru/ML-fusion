"""
This script implements the separate models approach for the x_m values.
Used as a test script.
"""

#%%
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

# Given hyperparameters
#found in seperate_models_xm_xgb_finding.py
hyper_params = {
    'learning_rate': 0.1,
    'max_depth': 2,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'n_estimators': 600
}

# Unique x_m values
unique_x_m = df['x_m'].unique()

# Initialize dictionary to store trained models for each x_m
trained_models = {}

# New feature set for prediction
new_feature_set = {
    'angle': 3,
    'heat': 0.15,
    'field': 3,
    'emission': 0.9
}

# Initialize dictionary to store predictions for each x_m
predictions = {}
#%%
# Train a model for each unique x_m value
for x_m_value in unique_x_m:
    # Filter the data for the current x_m value
    x_m_data = df[df['x_m'] == x_m_value]
    
    # Separate features and target
    X_train_xm = x_m_data[['angle', 'heat', 'field', 'emission']]
    y_train_xm = x_m_data['Pot']
    
    # Train the XGBoost model
    xgb_model_xm = XGBRegressor(**hyper_params)
    xgb_model_xm.fit(X_train_xm, y_train_xm)
    
    # Store the trained model
    trained_models[x_m_value] = xgb_model_xm
    
    # Make a prediction for the new feature set
    prediction = xgb_model_xm.predict(pd.DataFrame([new_feature_set]))
    
    # Store the prediction
    predictions[x_m_value] = prediction[0]

# %%
# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(list(predictions.keys()), list(predictions.values()), 'o-')
plt.xlabel('x_m')
plt.ylabel('Pot')
plt.title('Predictions for new feature set')   
plt.show()