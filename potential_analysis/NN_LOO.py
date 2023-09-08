#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)
#%%
# Separate features and target variable from DataFrame
X = df[['angle', 'heat', 'field', 'emission', 'x_m']]
y = df['Pot']

#get all the unique combinations of angle, heat, field and emission
unique_combinations = df[['angle', 'heat', 'field', 'emission']].drop_duplicates()

# Initialize an empty list to store the mean squared error for each test set
mse_errors = []
mae_errors = []
mape_errors = []

# Loop through each unique combination of 'angle', 'heat', 'field', 'emission'
for index, row in unique_combinations.iterrows():
    # Create the test set based on the unique combination
    condition = (
        (df['angle'] == row['angle']) & 
        (df['heat'] == row['heat']) & 
        (df['field'] == row['field']) & 
        (df['emission'] == row['emission'])
    )
    X_test = df[condition][['angle', 'heat', 'field', 'emission', 'x_m']]
    y_test = df[condition]['Pot']
    
    # Create the training set by excluding the current unique combination
    X_train = df[~condition][['angle', 'heat', 'field', 'emission', 'x_m']]
    y_train = df[~condition]['Pot']
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, verbose=2)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    mse_errors.append(mse)
    mae_errors.append(mae)
    mape_errors.append(mape)
    
    print(f"Errors for test set with angle={row['angle']}, heat={row['heat']}, field={row['field']}, emission={row['emission']}:")
    print(f"MSE: {mse}, MAE: {mae}, MAPE: {mape}%")

# Calculate the average errors across all test sets
average_mse = np.mean(mse_errors)
average_mae = np.mean(mae_errors)
average_mape = np.mean(mape_errors)

print(f"Average MSE: {average_mse}")
print(f"Average MAE: {average_mae}")
print(f"Average MAPE: {average_mape}%")





















#%%


# Fit the model
history = model.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2)

# %%
# Evaluate the model
model.evaluate(X_test, y_test)

# %%
import matplotlib.pyplot as plt
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

new_data_scaled = scaler.transform(new_data)

# Use the trained Random Forest model to predict the Pot values for the new data
predicted_Pot = model.predict(new_data_scaled)

# Create a DataFrame for the new graph
new_graph_data = pd.DataFrame({
    'x_m': unique_x_m,
    'Predicted_Pot': predicted_Pot.flatten()
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
"""RNNS"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df_sorted = df.sort_values(by=['angle', 'heat', 'field', 'emission', 'x_m'])
grouped = df_sorted.groupby(['angle', 'heat', 'field', 'emission'])

X, y = [], []

for name, group in grouped:
    X.append(group[['angle', 'heat', 'field', 'emission', 'x_m']].values)
    y.append(group['Pot'].values)

X = np.array(X)
y = np.array(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()

# LSTM layers
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(N, 5)))
model.add(LSTM(50, activation='relu'))

# Dense layer to output N points
model.add(Dense(N))

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=1)
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


# Suppose df_new is your new DataFrame
df_new_sorted = new_data.sort_values(by=['angle', 'heat', 'field', 'emission', 'x_m'])
X_new = df_new_sorted[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, N, 5)

# Making predictions
new_predictions = model.predict(X_new)
#%%
#plot the new_predictions
plt.figure(figsize=(12, 6))
plt.plot(new_data['x_m'], new_predictions.flatten(), label='Predicted Pot')
plt.xlabel('x_m [m]')
plt.ylabel('Predicted Pot')
plt.title('New Graph for Specified Feature Values')
plt.legend()
plt.grid()
plt.show()

# %%
