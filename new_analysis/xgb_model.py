#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.signal import savgol_filter

# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

#%%
# Split the data into features (X) and target variable (y)
X = df.drop('Pot', axis=1)
y = df['Pot']

# Initialize XGBoost model with specified hyperparameters
xgb_model = xgb.XGBRegressor(
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=1,
    subsample=0.5,
    colsample_bytree=0.8,
    n_estimators=600,
    objective='reg:squarederror'  # Use squared error as the objective function
)

# Train the model
xgb_model.fit(X, y)

#%%
# Get feature importances from the trained model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame for feature importances
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by the importances
features_df = features_df.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
#train on all data
# Initialize Random Forest Regressor with specified hyperparameters
rf_model = RandomForestRegressor()

# Train the model
rf_model.fit(X, y)

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_
# Create a DataFrame for feature importances
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by the importances
features_df = features_df.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#%%
#simple xgb model without hyperparameters
# Initialize XGBoost model
xgb_model_simple = xgb.XGBRegressor()
xgb_model_simple.fit(X, y)
# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_
# Create a DataFrame for feature importances
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by the importances
features_df = features_df.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# %% Make new graph
# Extract unique x_m values from the original dataset
unique_x_m = df['x_m'].unique()

# Create a new DataFrame with the given feature values and unique x_m values
new_data = pd.DataFrame({
    'angle': 3,
    'heat': 0.15,
    'field': 3,
    'emission': 0.9,
    'x_m': unique_x_m
})

# Show the first few rows of the new DataFrame
new_data.head()

# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = xgb_model.predict(new_data)

predicted_Pot_rf = rf_model.predict(new_data)

predicted_Pot_xgb_simple = xgb_model_simple.predict(new_data)

#smooth the predicted Pot values

predicted_Pot = savgol_filter(predicted_Pot, 5, 3)

# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

new_graph_data_rf = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot_rf
})

new_graph_data_xgb_simple = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot_xgb_simple
})

# Plot the new graph using the predicted Pot values
plt.figure(figsize=(12, 6))
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.plot(new_graph_data_rf['x_m'], new_graph_data_rf['Predicted_Pot'], label='Predicted Pot RF')
plt.plot(new_graph_data_xgb_simple['x_m'], new_graph_data_xgb_simple['Predicted_Pot'], label='Predicted Pot XGB Simple')
plt.xlabel('x_m [m]')
plt.ylabel('Predicted Pot')
plt.title('New Graph for Specified Feature Values')
plt.legend()
plt.grid()
plt.show()

# %%
#load new_Ivona_data excel file
file_path = 'new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='all_together')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.plot(new_graph_data_rf['x_m'], new_graph_data_rf['Predicted_Pot'], label='Predicted Pot RF')
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()
#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)




#%%
new_data = pd.DataFrame({
    'angle': 3,
    'heat': 0,
    'field': 2.2,
    'emission': 0.8,
    'x_m': unique_x_m
})
# Show the first few rows of the new DataFrame
new_data.head()
# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = xgb_model.predict(new_data)

predicted_Pot_rf = rf_model.predict(new_data)
#if predicted_Pot is negative, make it 0
predicted_Pot[predicted_Pot < 0] = 0
#smooth the predicted Pot values
predicted_Pot = savgol_filter(predicted_Pot, 5, 3)
# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

new_graph_data_rf = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot_rf
})

file_path = 'new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='angle_new 3')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.plot(new_graph_data_rf['x_m'], new_graph_data_rf['Predicted_Pot'], label='Predicted Pot RF')
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)

#%%
new_data = pd.DataFrame({
    'angle': 6,
    'heat': 0.15,
    'field': 2.2,
    'emission': 0.8,
    'x_m': unique_x_m
})
# Show the first few rows of the new DataFrame
new_data.head()
# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = xgb_model.predict(new_data)
#if predicted_Pot is negative, make it 0
predicted_Pot[predicted_Pot < 0] = 0
#smooth the predicted Pot values
predicted_Pot = savgol_filter(predicted_Pot, 5, 3)
# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

file_path = 'new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='heat_new_0.15')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)

#%%
new_data = pd.DataFrame({
    'angle': 6,
    'heat': 0,
    'field': 3,
    'emission': 0.8,
    'x_m': unique_x_m
})
# Show the first few rows of the new DataFrame
new_data.head()
# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = xgb_model.predict(new_data)
#if predicted_Pot is negative, make it 0
predicted_Pot[predicted_Pot < 0] = 0
#smooth the predicted Pot values
predicted_Pot = savgol_filter(predicted_Pot, 5, 3)
# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

file_path = 'new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='field_new_3')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)

#%%
new_data = pd.DataFrame({
    'angle': 6,
    'heat': 0,
    'field': 2.2,
    'emission': 0.9,
    'x_m': unique_x_m
})
# Show the first few rows of the new DataFrame
new_data.head()
# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = xgb_model.predict(new_data)
#if predicted_Pot is negative, make it 0
predicted_Pot[predicted_Pot < 0] = 0
#smooth the predicted Pot values
predicted_Pot = savgol_filter(predicted_Pot, 5, 3)
# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

file_path = 'new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='emission_new_0.9')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', mae)
print('Mean Absolute Percentage Error:', mape)







#%%
#try gridsearch
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200,500, 600],
    'max_depth': [1, 2, 3, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)


# Initialize Grid Search for Random Forest without feature scaling
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                              cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Get the best parameters from the Grid Search
best_params_rf = grid_search_rf.best_params_

# Train a Random Forest model with the best parameters
best_rf_model = RandomForestRegressor(**best_params_rf, random_state=42)
best_rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_best_rf = best_rf_model.predict(X_test)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

# Train and evaluate an XGBoost model for comparison
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Get feature importances for the best Random Forest model and the XGBoost model
feature_importances_best_rf = best_rf_model.feature_importances_
feature_importances_xgb = xgb_model.feature_importances_

# Create DataFrames for feature importances
features_best_rf_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_RF': feature_importances_best_rf
})

features_xgb_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_XGB': feature_importances_xgb
})

# Combine the feature importance DataFrames
features_combined_df = pd.merge(features_best_rf_df, features_xgb_df, on='Feature')

# Create a DataFrame to store the evaluation metrics for the best Random Forest model and XGBoost
metrics_best_models_df = pd.DataFrame({
    'Model': ['Best Random Forest', 'XGBoost'],
    'MAE': [mae_best_rf, mae_xgb],
    'MSE': [mse_best_rf, mse_xgb],
    'R2 Score': [r2_best_rf, r2_xgb]
})

best_params_rf, metrics_best_models_df, features_combined_df



























# %%






from sklearn.preprocessing import StandardScaler

# Identify unique combinations of 'angle', 'heat', 'field', and 'emission' to represent different curves
unique_curves = df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])[['angle', 'heat', 'field', 'emission']]

# Show the number of unique curves and the first few rows
num_unique_curves = unique_curves.shape[0]
unique_curves.head(), num_unique_curves

#%%
# Initialize lists to store performance metrics for each iteration
mae_list = []
mse_list = []
r2_list = []

# Loop through each unique curve to perform LOOCV
for index, row in unique_curves.iterrows():
    angle, heat, field, emission = row['angle'], row['heat'], row['field'], row['emission']
    
    # Create train and test sets based on the current curve
    test_set = df[(df['angle'] == angle) & (df['heat'] == heat) & (df['field'] == field) & (df['emission'] == emission)]
    train_set = df.drop(test_set.index)
    
    # Split the data into features (X) and target variable (y)
    X_train = train_set.drop('Pot', axis=1)
    y_train = train_set['Pot']
    X_test = test_set.drop('Pot', axis=1)
    y_test = test_set['Pot']
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the Random Forest Regressor
    rf_model_loocv = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_loocv.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model_loocv.predict(X_test_scaled)
    
    # Evaluate the model and store the metrics
    mae_list.append(mean_absolute_error(y_test, y_pred))
    mse_list.append(mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

# Calculate average performance metrics
avg_mae = sum(mae_list) / len(mae_list)
avg_mse = sum(mse_list) / len(mse_list)
avg_r2 = sum(r2_list) / len(r2_list)

avg_mae, avg_mse, avg_r2

# %%
