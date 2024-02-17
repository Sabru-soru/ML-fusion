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
prediction_parameter = 'Pot'
df = df[['angle', 'heat', 'field', 'emission', 'x_m', prediction_parameter]]

#%%
n_segments = 20

# Determine the range of x_m values
x_m_min = df['x_m'].min()
x_m_max = df['x_m'].max()

# Calculate the segment size
segment_size = (x_m_max - x_m_min) / n_segments

# Dictionary to store models for each segment
models = {}

# Loop through each segment and filter data
for segment in range(n_segments):
    lower_bound = x_m_min + segment * segment_size
    upper_bound = lower_bound + segment_size
    
    # Filter the dataset for the current segment
    df_segment = df[(df['x_m'] >= lower_bound) & (df['x_m'] < upper_bound)]
    
    # Prepare the features and target for the segment
    X_segment = df_segment[['angle', 'heat', 'field', 'emission', 'x_m']]
    y_segment = df_segment[prediction_parameter]  # Replace 'target_column' with the name of your target column
    
    # Train the XGBoost model for the segment
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=3,
        subsample=1,
        colsample_bytree=0.8,
        n_estimators=300,
        objective='reg:squarederror'
    )
    model.fit(X_segment, y_segment)
    
    # Store the model
    models[segment] = model

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

# Function to determine which segment an x_m value belongs to
def get_segment_for_x_m(x_m, x_m_min, segment_size):
    if x_m == x_m_max:  # Edge case where x_m is the maximum value
        return (n_segments-1)  # The last segment
    return int((x_m - x_m_min) / segment_size)

# Predicting with the segmented models
predicted_values = []

for _, row in new_data.iterrows():
    x_m = row['x_m']
    segment = get_segment_for_x_m(x_m, x_m_min, segment_size)
    model = models[segment]
    # Make sure to reshape the row to the correct shape for prediction
    features = row[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, -1)
    predicted_value = model.predict(features)
    predicted_values.append(predicted_value[0])

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

# %%
