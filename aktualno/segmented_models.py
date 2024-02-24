#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# Load the data from the Excel file
file_path = 'data/data_cleaned_sparse_all.xlsx'
df = pd.read_excel(file_path)

#load actual values
file_path = 'data/new_output_fusion_sparse.xlsx'
new_Ivona = pd.read_excel(file_path)
#%%
parameters = ['Pot','Tn','Te','Ti','Vi','Vn','nn','E','Ve']
prediction_parameter = 'Pot'
df = df[['angle', 'heat', 'field', 'emission', 'x_m', prediction_parameter]]

#%%
# Sort the DataFrame by x_m values to find cutoff points
df_sorted = df.sort_values(by='x_m')
x_m_values = df_sorted['x_m'].values

# Calculate indices for the 10% and 90% cutoffs
first_5_percent_index = int(len(x_m_values) * 0.1)
last_5_percent_index = int(len(x_m_values) * 0.9)

# Get cutoff values for x_m
first_5_percent_cutoff = x_m_values[first_5_percent_index]
last_5_percent_cutoff = x_m_values[last_5_percent_index]

# Split the DataFrame into three segments
df_first_5 = df_sorted[df_sorted['x_m'] <= first_5_percent_cutoff]
df_last_5 = df_sorted[df_sorted['x_m'] >= last_5_percent_cutoff]
df_middle = df_sorted[(df_sorted['x_m'] > first_5_percent_cutoff) & (df_sorted['x_m'] < last_5_percent_cutoff)]

#%%
# Define a function to train an XGBoost model
def train_xgb_model(X, y):
    if prediction_parameter=='Pot':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=3,
            subsample=1,
            colsample_bytree=0.8,
            n_estimators=300,
            objective='reg:squarederror'  #Pot
        )
    elif prediction_parameter=='Tn':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=7,
            subsample=1,
            colsample_bytree=1,
            n_estimators=300,
            objective='reg:squarederror'  #Tn
        )
    elif prediction_parameter=='Te':
        model = xgb.XGBRegressor(
            learning_rate=0.3,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            objective='reg:squarederror'  #Te
        )
    elif prediction_parameter=='Ti':
        model = xgb.XGBRegressor(
            learning_rate=0.3,
            max_depth=7,
            subsample=1,
            colsample_bytree=1,
            n_estimators=1000,
            objective='reg:squarederror'  #Ti
        )
    elif prediction_parameter=='Vi':
        model = xgb.XGBRegressor(
            learning_rate=0.3,
            max_depth=3,
            subsample=1,
            colsample_bytree=0.8,
            n_estimators=1000,
            objective='reg:squarederror'  #Vi
        )
    elif prediction_parameter=='Vn':
        model = xgb.XGBRegressor(
            learning_rate=0.01,
            max_depth=5,
            subsample=1,
            colsample_bytree=1,
            n_estimators=600,
            objective='reg:squarederror'  #Vn
        )
    elif prediction_parameter=='nn':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=1,
            n_estimators=1000,
            objective='reg:squarederror'  #nn
        )
    elif prediction_parameter=='E':
        model = xgb.XGBRegressor(
            learning_rate=0.01,
            max_depth=3,
            subsample=1,
            colsample_bytree=1,
            n_estimators=300,
            objective='reg:squarederror'  #E
        )
    elif prediction_parameter=='Ve':
        model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            objective='reg:squarederror'  #Ve
        )
    model.fit(X, y)
    return model

# Train a model for each segment
model_first_5 = train_xgb_model(df_first_5[['angle', 'heat', 'field', 'emission', 'x_m']], df_first_5[prediction_parameter])
model_last_5 = train_xgb_model(df_last_5[['angle', 'heat', 'field', 'emission', 'x_m']], df_last_5[prediction_parameter])
model_middle = train_xgb_model(df_middle[['angle', 'heat', 'field', 'emission', 'x_m']], df_middle[prediction_parameter])

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

def predict_with_segmented_models(new_data):
    predictions = []  # to store predictions for each row in new_data
    for _, row in new_data.iterrows():
        x_m = row['x_m']
        # Select features for prediction
        features = row[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, -1)
        # Determine which model to use based on x_m value
        if x_m <= first_5_percent_cutoff:
            prediction = model_first_5.predict(features)
        elif x_m >= last_5_percent_cutoff:
            prediction = model_last_5.predict(features)
        else:
            prediction = model_middle.predict(features)
        predictions.append(prediction[0])  # Assuming prediction returns a list, take the first element
    return predictions

# Use the function to make predictions for the new_data
predicted_values = predict_with_segmented_models(new_data)

# Add the predicted values to the new_data DataFrame
new_data['predicted'] = predicted_values


# %%
# smooth the predicted values
new_data['predicted_smooth'] = savgol_filter(new_data['predicted'], 30, 3)

#plot x_m and predicted values
fig = go.Figure()
fig.add_trace(go.Scatter(x=new_data['x_m'], y=new_data['predicted'], mode='lines', name='Predicted'))
fig.add_trace(go.Scatter(x=new_data['x_m'], y=new_data['predicted_smooth'], mode='lines', name='Predicted Smooth'))
fig.add_trace(go.Scatter(x=new_Ivona['x_m'], y=new_Ivona[prediction_parameter], mode='lines', name='Actual'))


# Iterate through each unique combination of parameters
unique_combinations = df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])

for index, row in unique_combinations.iterrows():
    # Filter the dataset for the current combination
    subset = df[(df['angle'] == row['angle']) & 
                (df['heat'] == row['heat']) & 
                (df['field'] == row['field']) & 
                (df['emission'] == row['emission'])]
    
    # Create a trace for the current combination
    fig.add_trace(go.Scatter(x=subset['x_m'], y=subset[prediction_parameter],
                             mode='lines', name=f'Params: {row["angle"]}, {row["heat"]}, {row["field"]}, {row["emission"]}',
                             visible='legendonly',  # Make the trace not visible initially
                             opacity=0.35,
                             line=dict(dash='dash', width=2)))  # Semi-transparent and dashed lines

# Adjust layout if necessary
fig.update_layout(title='Predicted vs. Actual Data with Training Data',
                  xaxis_title='x_m',
                  yaxis_title='Value',
                  legend_title='Traces')

fig.show()


# %%
