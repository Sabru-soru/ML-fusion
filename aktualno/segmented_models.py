#%%
import pandas as pd
import xgboost as xgb
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
import numpy as np

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
first_segment_percent_index = int(len(x_m_values) * 0.1)
last_segment_percent_index = int(len(x_m_values) * 0.9)

# Get cutoff values for x_m
first_segment_percent_cutoff = x_m_values[first_segment_percent_index]
last_segment_percent_cutoff = x_m_values[last_segment_percent_index]

# Split the DataFrame into three segments
df_first_segment = df_sorted[df_sorted['x_m'] <= first_segment_percent_cutoff]
df_last_segment = df_sorted[df_sorted['x_m'] >= last_segment_percent_cutoff]
df_middle = df_sorted[(df_sorted['x_m'] > first_segment_percent_cutoff) & (df_sorted['x_m'] < last_segment_percent_cutoff)]

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

#%%
# Train a model for each segment
model_first_segment = train_xgb_model(df_first_segment[['angle', 'heat', 'field', 'emission', 'x_m']], df_first_segment[prediction_parameter])
model_last_segment = train_xgb_model(df_last_segment[['angle', 'heat', 'field', 'emission', 'x_m']], df_last_segment[prediction_parameter])

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

model_first_segment_max = df_first_segment['x_m'].max()
#predict on model_first_segment_max with model_first_segment
predict_first = model_first_segment.predict(new_data[new_data['x_m']==model_first_segment_max][['angle', 'heat', 'field', 'emission', 'x_m']])

model_last_segment_min = df_last_segment['x_m'].min()
#predict on model_last_segment_min with model_last_segment
predict_last = model_last_segment.predict(new_data[new_data['x_m']==model_last_segment_min][['angle', 'heat', 'field', 'emission', 'x_m']])

#append to df_middle
df_middle = df_middle.append({'angle': 3, 'heat': 0.15, 'field': 3, 'emission': 0.9, 'x_m': model_first_segment_max, prediction_parameter: predict_first[0]}, ignore_index=True)
# df_middle = df_middle.append({'angle': 3, 'heat': 0.15, 'field': 3, 'emission': 0.9, 'x_m': model_last_segment_min, prediction_parameter: predict_last[0]}, ignore_index=True)

# Train a model for the middle segment
model_middle = train_xgb_model(df_middle[['angle', 'heat', 'field', 'emission', 'x_m']], df_middle[prediction_parameter])


def predict_with_segmented_models(new_data):
    predictions = []  # to store predictions for each row in new_data
    for _, row in new_data.iterrows():
        x_m = row['x_m']
        # Select features for prediction
        features = row[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, -1)
        # Determine which model to use based on x_m value
        if x_m <= first_segment_percent_cutoff:
            prediction = model_first_segment.predict(features)
        elif x_m >= last_segment_percent_cutoff:
            prediction = model_last_segment.predict(features)
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
new_data['predicted_smooth'] = savgol_filter(new_data['predicted'], 20, 3)

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
fig.update_layout(title=f'Predicted vs. Actual Data with Training Data. Parameter: {prediction_parameter}',
                  xaxis_title='x_m',
                  yaxis_title='Value',
                  legend_title='Traces')

fig.show()


# %%
#merge dataframes to get new_Ivona data with predicted Pot values
merged = pd.merge(new_data, new_Ivona[['x_m', prediction_parameter]], on='x_m', how='left')

merged = merged[merged[prediction_parameter]>10]

mae = mean_absolute_error(merged[prediction_parameter], merged['predicted'])
mape = np.mean(np.abs((merged[prediction_parameter] - merged['predicted']) / merged[prediction_parameter])) * 100
print('Mean Absolute Error:', round(mae,2))
print('Mean Absolute Percentage Error:', round(mape,2))
# %%
