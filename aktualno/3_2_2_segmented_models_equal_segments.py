import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go

class DataProcessor:
    def __init__(self, data_path, fusion_data_path, prediction_parameter):
        self.data = pd.read_pickle(data_path)
        self.fusion_data = pd.read_pickle(fusion_data_path)
        self.prediction_parameter = prediction_parameter
        self.parameters = ['angle', 'heat', 'field', 'emission', 'x_m', prediction_parameter]
        self.data = self.data[self.parameters].copy()
        self.unique_x_m = self.data['x_m'].unique()
        self.x_m_min = self.data['x_m'].min()
        self.x_m_max = self.data['x_m'].max()
        self.n_segments = None
        self.segment_size = None
        self.segments = []

    def create_segments(self, n_segments):
        self.n_segments = n_segments
        self.segment_size = (self.x_m_max - self.x_m_min) / n_segments
        self.segments = []
        for segment in range(n_segments):
            lower_bound = self.x_m_min + segment * self.segment_size
            upper_bound = lower_bound + self.segment_size
            df_segment = self.data[(self.data['x_m'] >= lower_bound) & (self.data['x_m'] < upper_bound)]
            self.segments.append(df_segment)

class ModelTrainer:
    def __init__(self, prediction_parameter):
        self.prediction_parameter = prediction_parameter
        self.models = {}

    def get_model_params(self):
        param_dict = {
            'Pot': {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1,
                    'colsample_bytree': 0.8, 'n_estimators': 300},
            'Tn': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 1,
                   'colsample_bytree': 1, 'n_estimators': 300},
            'Te': {'learning_rate': 0.3, 'max_depth': 7, 'subsample': 0.8,
                   'colsample_bytree': 0.8, 'n_estimators': 1000},
            'Ti': {'learning_rate': 0.3, 'max_depth': 7, 'subsample': 1,
                   'colsample_bytree': 1, 'n_estimators': 1000},
            'Vi': {'learning_rate': 0.3, 'max_depth': 3, 'subsample': 1,
                   'colsample_bytree': 0.8, 'n_estimators': 1000},
            'Vn': {'learning_rate': 0.01, 'max_depth': 5, 'subsample': 1,
                   'colsample_bytree': 1, 'n_estimators': 600},
            'nn': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8,
                   'colsample_bytree': 1, 'n_estimators': 1000},
            'E': {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 1,
                  'colsample_bytree': 1, 'n_estimators': 300},
            'Ve': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8,
                   'colsample_bytree': 0.8, 'n_estimators': 1000}
        }
        return param_dict.get(self.prediction_parameter, {})

    def train_segment_models(self, segments):
        params = self.get_model_params()
        for idx, segment_data in enumerate(segments):
            X_segment = segment_data[['angle', 'heat', 'field', 'emission', 'x_m']]
            y_segment = segment_data[self.prediction_parameter]
            model = xgb.XGBRegressor(objective='reg:squarederror', **params)
            model.fit(X_segment, y_segment)
            self.models[idx] = model

    def get_segment_for_x_m(self, x_m, x_m_min, segment_size, n_segments, x_m_max):
        if x_m == x_m_max:
            return n_segments - 1
        return int((x_m - x_m_min) / segment_size)

    def predict(self, new_data, data_processor):
        predicted_values = []
        for _, row in new_data.iterrows():
            x_m = row['x_m']
            segment = self.get_segment_for_x_m(
                x_m, data_processor.x_m_min, data_processor.segment_size,
                data_processor.n_segments, data_processor.x_m_max
            )
            model = self.models[segment]
            features = row[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, -1)
            predicted_value = model.predict(features)
            predicted_values.append(predicted_value[0])
        return predicted_values

class Evaluator:
    @staticmethod
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Plotter:
    @staticmethod
    def plot_results(new_data, fusion_data, prediction_parameter, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=new_data['x_m'], y=new_data['predicted'], mode='lines', name='Predicted'))
        fig.add_trace(go.Scatter(x=new_data['x_m'], y=new_data['predicted_smooth'], mode='lines', name='Predicted Smooth'))
        fig.add_trace(go.Scatter(x=fusion_data['x_m'], y=fusion_data[prediction_parameter], mode='lines', name='Actual'))
        
        # Iterate through each unique combination of parameters
        unique_combinations = df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])
        for index, row in unique_combinations.iterrows():
            subset = df[(df['angle'] == row['angle']) & 
                        (df['heat'] == row['heat']) & 
                        (df['field'] == row['field']) & 
                        (df['emission'] == row['emission'])]

            # Create a trace for the current combination
            fig.add_trace(go.Scatter(x=subset['x_m'], y=subset[prediction_parameter],
                                     mode='lines', 
                                     name=f"Params: {row['angle']}, {row['heat']}, {row['field']}, {row['emission']}",
                                     visible='legendonly',
                                     opacity=0.35,
                                     line=dict(dash='dash', width=2)))

        fig.update_layout(title='Predicted vs. Actual Data with Training Data',
                          xaxis_title='x_m',
                          yaxis_title=prediction_parameter,
                          legend_title='Traces')

        fig.show()

def main():
    data_processor = DataProcessor(
        data_path='data/df_data.pkl',
        fusion_data_path='data/new_output_fusion_sparse.pkl',
        prediction_parameter='Pot'
    )
    data_processor.create_segments(n_segments=20)

    # Initialize Model Trainer and train segment models
    model_trainer = ModelTrainer(prediction_parameter='Pot')
    model_trainer.train_segment_models(data_processor.segments)

    # Create new data for prediction
    new_data = pd.DataFrame({
        'angle': 3,
        'heat': 0.15,
        'field': 3,
        'emission': 0.9,
        'x_m': data_processor.unique_x_m
    })

    # Make predictions using segmented models
    predicted_values = model_trainer.predict(new_data, data_processor)
    new_data['predicted'] = predicted_values

    # Smooth the predicted values
    new_data['predicted_smooth'] = savgol_filter(new_data['predicted'], 30, 3)

    # Evaluate the results
    merged = pd.merge(new_data, data_processor.fusion_data[['x_m', data_processor.prediction_parameter]], on='x_m', how='left')
    merged = merged.dropna(subset=[data_processor.prediction_parameter])
    mae = Evaluator.calculate_mae(merged[data_processor.prediction_parameter], merged['predicted'])
    mape = Evaluator.calculate_mape(merged[data_processor.prediction_parameter], merged['predicted'])
    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Absolute Percentage Error:', round(mape, 2), '%')

    Plotter.plot_results(new_data, data_processor.fusion_data, data_processor.prediction_parameter, data_processor.data)

if __name__ == "__main__":
    main()
