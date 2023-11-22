"""
In this script we compare the models
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.signal import savgol_filter

# Load the data from the Excel file
file_path = 'data/data_cleaned_sparse_all.xlsx'
df = pd.read_excel(file_path)
X = df[['angle', 'heat', 'field', 'emission', 'x_m']]

prediction_parameter = 'Pot'
y = df[prediction_parameter]

#%% 
"Train the model from LOO method"
# Initialize XGBoost model with specified hyperparameters
if prediction_parameter=='Pot':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=3,
        subsample=1,
        colsample_bytree=0.8,
        n_estimators=300,
        objective='reg:squarederror'  #Pot
    )
elif prediction_parameter=='Tn':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=7,
        subsample=1,
        colsample_bytree=1,
        n_estimators=300,
        objective='reg:squarederror'  #Tn
    )
elif prediction_parameter=='Te':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.3,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        objective='reg:squarederror'  #Te
    )
elif prediction_parameter=='Ti':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.3,
        max_depth=7,
        subsample=1,
        colsample_bytree=1,
        n_estimators=1000,
        objective='reg:squarederror'  #Ti
    )
elif prediction_parameter=='Vi':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.3,
        max_depth=3,
        subsample=1,
        colsample_bytree=0.8,
        n_estimators=1000,
        objective='reg:squarederror'  #Vi
    )
elif prediction_parameter=='Vn':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.01,
        max_depth=5,
        subsample=1,
        colsample_bytree=1,
        n_estimators=600,
        objective='reg:squarederror'  #Vn
    )
elif prediction_parameter=='nn':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=1,
        n_estimators=1000,
        objective='reg:squarederror'  #nn
    )
elif prediction_parameter=='E':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.01,
        max_depth=3,
        subsample=1,
        colsample_bytree=1,
        n_estimators=300,
        objective='reg:squarederror'  #E
    )
elif prediction_parameter=='Ve':
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        objective='reg:squarederror'  #Ve
    )

# Train the model
xgb_model.fit(X, y)

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

#%%
"simple xgb model without hyperparameters for comparison"
# Initialize XGBoost model
xgb_model_simple = xgb.XGBRegressor()
xgb_model_simple.fit(X, y)

# %% 
"Make new graph"
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

# Use the trained Random Forest model to predict the Pot values for the new data
predicted_parameter = xgb_model.predict(new_data)
#smooth the predicted Pot values
predicted_parameter = savgol_filter(predicted_parameter, 5, 3)

predicted_parameter_xgb_simple = xgb_model_simple.predict(new_data)

# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    f'Predicted_{prediction_parameter}': predicted_parameter
})

new_graph_data_xgb_simple = pd.DataFrame({
    'x_m': unique_x_m,
    f'Predicted_{prediction_parameter}': predicted_parameter_xgb_simple
})

#%%
"predict using new_model for each x_m"
# Given hyperparameters from the separate_models_xm_xgb_finding.py script
hyper_params = {
    'learning_rate': 0.1,
    'max_depth': 2,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'n_estimators': 600
}
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
predictions5000 = {}

# Train a model for each unique x_m value
for x_m_value in unique_x_m:
    # Filter the data for the current x_m value
    x_m_data = df[df['x_m'] == x_m_value]
    
    # Separate features and target
    X_train_xm = x_m_data[['angle', 'heat', 'field', 'emission']]
    y_train_xm = x_m_data[prediction_parameter]
    
    # Train the XGBoost model. If the prediction parameter is Pot, use the given hyperparameters
    if prediction_parameter=='Pot':
        xgb_model_xm = xgb.XGBRegressor(**hyper_params)
    else:
        xgb_model_xm = xgb.XGBRegressor()

    xgb_model_xm.fit(X_train_xm, y_train_xm)
    
    # Store the trained model
    trained_models[x_m_value] = xgb_model_xm
    
    # Make a prediction for the new feature set
    prediction = xgb_model_xm.predict(pd.DataFrame([new_feature_set]))
    
    # Store the prediction
    predictions5000[x_m_value] = prediction[0]


# %%
#load new_Ivona_data excel file
# file_path = 'data/new_output_fusion.xlsx'
# new_Ivona = pd.read_excel(file_path,sheet_name='all_together')
# #multiply x_m values by 10
# new_Ivona['x_m'] = new_Ivona['x_m']*10

# #take only unique_x_m values from new_Ivona data
# new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]
# #save new_Ivona data to excel file
# new_Ivona.to_excel('data/new_output_fusion_sparse.xlsx', index=False)

#load new_Ivona_data excel file
file_path = 'data/new_output_fusion_sparse.xlsx'
new_Ivona = pd.read_excel(file_path)

#%%
#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona[prediction_parameter], label=f'Actual {prediction_parameter}')
plt.plot(new_graph_data['x_m'], new_graph_data[f'Predicted_{prediction_parameter}'], '.-',label=f'Predicted {prediction_parameter}')
plt.plot(new_graph_data_xgb_simple['x_m'], new_graph_data_xgb_simple[f'Predicted_{prediction_parameter}'], label=f'Predicted {prediction_parameter} XGB Simple', linestyle='--')
plt.plot(list(predictions5000.keys()), list(predictions5000.values()), label=f'Predicted {prediction_parameter} separate x_m')
plt.xlabel('x_m [m]')
plt.ylabel(prediction_parameter)
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values to get error
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')
#calculate mean absolute error, mape
mae = mean_absolute_error(merged[prediction_parameter], merged[f'Predicted_{prediction_parameter}'])
mape = np.mean(np.abs((merged[prediction_parameter] - merged[f'Predicted_{prediction_parameter}']) / merged[prediction_parameter])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2),'%')


#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona[prediction_parameter], label=f'Actual {prediction_parameter}')
plt.plot(new_graph_data['x_m'], new_graph_data[f'Predicted_{prediction_parameter}'], label=f'Predicted {prediction_parameter}')
# plt.fill_between(new_graph_data['x_m'], new_graph_data[f'Predicted_{prediction_parameter}']-mae, new_graph_data[f'Predicted_{prediction_parameter}']+mae, alpha=0.3)
plt.xlabel('x_m [m]')
plt.ylabel(prediction_parameter)
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()


#%%
"""
Below are tests for changing one parameter for fusion parameter Pot
"""


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

#if predicted_Pot is negative, make it 0
predicted_Pot[predicted_Pot < 0] = 0
#smooth the predicted Pot values
predicted_Pot = savgol_filter(predicted_Pot, 5, 3)
# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot
})

file_path = 'data/new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='angle_new 3')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]


unique_combinations = new_data[['angle', 'heat', 'field', 'emission']].drop_duplicates()

#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Actual Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
#plot the text for unique_combinations
for i in range(len(unique_combinations)):
    plt.text(0.05, 0.95-i*0.05, f"angle:{unique_combinations.iloc[0]['angle']},\
                                heat:{unique_combinations.iloc[0]['heat']},\
                                field:{unique_combinations.iloc[0]['field']},\
                                emission:{unique_combinations.iloc[0]['emission']}",
                                transform=plt.gca().transAxes)
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')

#dont consider rows where Pot is 0
merged = merged[merged['Pot']!=0]

#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2))

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

file_path = 'data/new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='heat_new_0.15')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]
unique_combinations = new_data[['angle', 'heat', 'field', 'emission']].drop_duplicates()
#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
#plot the text for unique_combinations
for i in range(len(unique_combinations)):
    plt.text(0.05, 0.95-i*0.05, f"angle:{unique_combinations.iloc[0]['angle']},\
                                heat:{unique_combinations.iloc[0]['heat']},\
                                field:{unique_combinations.iloc[0]['field']},\
                                emission:{unique_combinations.iloc[0]['emission']}",
                                transform=plt.gca().transAxes)
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')
#dont consider rows where Pot is 0
merged = merged[merged['Pot']!=0]
#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2))

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

file_path = 'data/new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='field_new_3')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]
unique_combinations = new_data[['angle', 'heat', 'field', 'emission']].drop_duplicates()
#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
#plot the text for unique_combinations
for i in range(len(unique_combinations)):
    plt.text(0.05, 0.95-i*0.05, f"angle:{unique_combinations.iloc[0]['angle']},\
                                heat:{unique_combinations.iloc[0]['heat']},\
                                field:{unique_combinations.iloc[0]['field']},\
                                emission:{unique_combinations.iloc[0]['emission']}",
                                transform=plt.gca().transAxes)
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')
#dont consider rows where Pot is 0
merged = merged[merged['Pot']!=0]
#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2))

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

file_path = 'data/new_output_fusion.xlsx'
new_Ivona = pd.read_excel(file_path,sheet_name='emission_new_0.9')
#multiply x_m values by 10
new_Ivona['x_m'] = new_Ivona['x_m']*10

#take only unique_x_m values from new_Ivona data
new_Ivona = new_Ivona[new_Ivona['x_m'].isin(unique_x_m)]
unique_combinations = new_data[['angle', 'heat', 'field', 'emission']].drop_duplicates()
#plot new_Ivona data and new_graph_data
plt.figure(figsize=(12, 6))
plt.plot(new_Ivona['x_m'], new_Ivona['Pot'], label='Ivona Pot')
plt.plot(new_graph_data['x_m'], new_graph_data['Predicted_Pot'], label='Predicted Pot')
#plot the text for unique_combinations
for i in range(len(unique_combinations)):
    plt.text(0.05, 0.95-i*0.05, f"angle:{unique_combinations.iloc[0]['angle']},\
                                heat:{unique_combinations.iloc[0]['heat']},\
                                field:{unique_combinations.iloc[0]['field']},\
                                emission:{unique_combinations.iloc[0]['emission']}",
                                transform=plt.gca().transAxes)
plt.xlabel('x_m [m]')
plt.ylabel('Pot')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_Ivona, new_graph_data, on='x_m', how='left')
#dont consider rows where Pot is 0
merged = merged[merged['Pot']!=0]
#calculate mean absolute error, mape
mae = mean_absolute_error(merged['Pot'], merged['Predicted_Pot'])
mape = np.mean(np.abs((merged['Pot'] - merged['Predicted_Pot']) / merged['Pot'])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2))
# %%
