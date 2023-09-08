#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)
#%%
# Separate features and target variable from DataFrame
X = df[['angle', 'heat', 'field', 'emission', 'x_m']]
y = df['Pot']

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# %%
# Initialize the constructor
model = Sequential()

# Add layers
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# %%
# Fit the model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2)

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
