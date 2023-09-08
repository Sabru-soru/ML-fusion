#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

#%% Distributions of the features and target variable
# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# List of features and target variable
columns = ['angle', 'heat', 'field', 'emission', 'x_m', 'Pot']

# Loop through each column to create a subplot for each one
for i, col in enumerate(columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# %% Target variable vs features
# Set up the matplotlib figure
plt.figure(figsize=(20, 12))

# Loop through each feature to create a subplot for each one
for i, col in enumerate(columns[:-1], 1):  # Exclude 'Pot' from features
    plt.subplot(2, 3, i)
    sns.scatterplot(x=col, y='Pot', data=df, alpha=0.6)
    plt.title(f'Pot vs {col}')
    plt.xlabel(col)
    plt.grid()
    plt.ylabel('Pot')

plt.tight_layout()
plt.show()
# %% ML model

from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df.drop('Pot', axis=1)
y = df['Pot']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show the shape of the training and testing data
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#%% random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae, mse, r2
# %%

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

# Show some results of the predictions and the test set
results_df = pd.DataFrame({
    'True_Pot': y_test,
    'Predicted_Pot': y_pred
})
results_df = results_df.reset_index(drop=True)
results_df.head(10)
# %%
# Plotting the true and predicted Pot values for the test set
plt.figure(figsize=(15, 8))
plt.scatter(range(len(y_test)), y_test, label='True Pot', alpha=0.6, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted Pot', alpha=0.6, color='red')
plt.title('True vs Predicted Pot Values')
plt.xlabel('Index')
plt.ylabel('Pot Value')
plt.legend()
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
predicted_Pot = rf_model.predict(new_data)

# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

# Plot the new graph using the predicted Pot values
plt.figure(figsize=(12, 6))
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
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
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#%%
import numpy as np
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
