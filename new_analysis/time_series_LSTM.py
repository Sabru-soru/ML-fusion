#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error




# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

# Drop the first column if it's an unnamed index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Normalize the features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Separate features and target variable
X = df_scaled.drop('Pot', axis=1)
y = df_scaled['Pot']

# Reshape the data into 3D array as needed for LSTM
X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MSE: {loss}')

# Make predictions on test data
y_pred = model.predict(X_test)

# Transform the normalized prediction back to original scale
y_pred_original = scaler.inverse_transform(np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]), y_pred], axis=1))[:,-1]
y_test_original = scaler.inverse_transform(np.concatenate([X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.values.reshape(-1, 1)], axis=1))[:,-1]

# Plot actual vs predicted
plt.figure(figsize=(15, 6))
plt.plot(y_test_original, label='True')
plt.plot(y_pred_original, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Pot Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
# %%







# Identify unique combinations of 'angle', 'heat', 'field', and 'emission' to represent different curves
unique_curves = df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])

# Initialize lists to store performance metrics for each unique curve
mse_list = []
mae_list = []
mape_list = []

for _, row in unique_curves.iterrows():
    angle, heat, field, emission = row['angle'], row['heat'], row['field'], row['emission']

    # Create training and test sets based on the current curve
    test_set = df[(df['angle'] == angle) & (df['heat'] == heat) & (df['field'] == field) & (df['emission'] == emission)]
    train_set = df.drop(test_set.index)

    # Normalize the features
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_set), columns=train_set.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns)

    # Separate features and target variable
    X_train = train_scaled.drop('Pot', axis=1)
    y_train = train_scaled['Pot']
    X_test = test_scaled.drop('Pot', axis=1)
    y_test = test_scaled['Pot']

    # Reshape into 3D array for LSTM
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

    # Evaluate the model using custom metrics
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Multiply by 100 to get percentage
    
    mse_list.append(mse)
    mae_list.append(mae)
    mape_list.append(mape)

# Calculate average MSE, MAE, and MAPE
average_mse = np.mean(mse_list)
average_mae = np.mean(mae_list)
average_mape = np.mean(mape_list)

print(f'Average Test MSE: {average_mse}')
print(f'Average Test MAE: {average_mae}')
print(f'Average Test MAPE: {average_mape}%')


# %%
