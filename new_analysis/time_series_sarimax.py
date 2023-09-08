import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the data from the Excel file
file_path = 'data_cleaned_interpreter.xlsx'
df = pd.read_excel(file_path)

# Drop the first column if it's an unnamed index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Identify unique combinations of 'angle', 'heat', 'field', and 'emission' to represent different curves
unique_curves = df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])[['angle', 'heat', 'field', 'emission']]

# Initialize lists to store performance metrics for each unique curve
mse_list = []
mae_list = []
r2_list = []

# Function to train and evaluate a SARIMA model with exogenous variables
def train_evaluate_sarima(train, test, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), exog_train=None, exog_test=None):
    try:
        # Train the model
        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fitted = model.fit(disp=False)
        
        # Make predictions
        start = len(train)
        end = start + len(test) - 1
        predictions = model_fitted.get_prediction(start=start, end=end, exog=exog_test).predicted_mean

        # Evaluate the model
        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        r2 = r2_score(test, predictions)
        
        return mse, mae, r2
    
    except Exception as e:
        return str(e), None, None

# Loop through each unique curve to perform Leave-One-Curve-Out validation
for index, row in unique_curves.iterrows():
    angle, heat, field, emission = row['angle'], row['heat'], row['field'], row['emission']
    
    # Create train and test sets based on the current curve
    test_set = df[(df['angle'] == angle) & (df['heat'] == heat) & (df['field'] == field) & (df['emission'] == emission)]
    train_set = df.drop(test_set.index)
    
    # Sort by 'x_m' within each curve
    test_set = test_set.sort_values('x_m')
    train_set = train_set.sort_values('x_m')
    
    # Separate the time series and exogenous variables
    y_train = train_set['Pot']
    y_test = test_set['Pot']
    exog_train = train_set.drop('Pot', axis=1)
    exog_test = test_set.drop('Pot', axis=1)
    
    # Train and evaluate the SARIMA model
    mse, mae, r2 = train_evaluate_sarima(y_train, y_test, exog_train=exog_train, exog_test=exog_test)
    
    # Store the performance metrics
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)

# Calculate average performance metrics
avg_mse = sum(mse_list) / len(mse_list)
avg_mae = sum(mae_list) / len(mae_list)
avg_r2 = sum(r2_list) / len(r2_list)

avg_mse, avg_mae, avg_r2