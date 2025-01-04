import pandas as pd
import xgboost as xgb
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
import numpy as np

class DataProcessor:
    def __init__(self, data_path, fusion_data_path, prediction_parameter):
        self.data = pd.read_pickle(data_path)
        self.fusion_data = pd.read_pickle(fusion_data_path)
        self.prediction_parameter = prediction_parameter
        self.parameters = ['angle', 'heat', 'field', 'emission', 'x_m', prediction_parameter]
        self.data = self.data[self.parameters]
        self.unique_x_m = self.data['x_m'].unique()
        self.first_segment = None
        self.middle_segment = None
        self.last_segment = None
        self.first_cutoff = None
        self.last_cutoff = None

    def split_data_segments(self):
        df_sorted = self.data.sort_values(by='x_m')
        x_m_values = df_sorted['x_m'].values

        # Calculate indices for the 5% and 95% cutoffs
        first_index = int(len(x_m_values) * 0.05)
        last_index = int(len(x_m_values) * 0.95)

        # Get cutoff values for x_m
        self.first_cutoff = x_m_values[first_index]
        self.last_cutoff = x_m_values[last_index]

        # Split the DataFrame into three segments
        self.first_segment = df_sorted[df_sorted['x_m'] <= self.first_cutoff]
        self.last_segment = df_sorted[df_sorted['x_m'] >= self.last_cutoff]
        self.middle_segment = df_sorted[
            (df_sorted['x_m'] > self.first_cutoff) &
            (df_sorted['x_m'] < self.last_cutoff)
        ]

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

    def train_model(self, X, y):
        params = self.get_model_params()
        model = xgb.XGBRegressor(objective='reg:squarederror', **params)
        model.fit(X, y)
        return model

    def train_segment_models(self, data_processor):
        # Train models for the first and last segments
        self.models['first'] = self.train_model(
            data_processor.first_segment[['angle', 'heat', 'field', 'emission', 'x_m']],
            data_processor.first_segment[data_processor.prediction_parameter]
        )
        self.models['last'] = self.train_model(
            data_processor.last_segment[['angle', 'heat', 'field', 'emission', 'x_m']],
            data_processor.last_segment[data_processor.prediction_parameter]
        )

    def train_middle_model(self, data_processor, predict_first, predict_last,
                           model_first_segment_max, model_last_segment_min):
        # Append predictions to the middle segment
        data_processor.middle_segment = data_processor.middle_segment.append(
            {
                'angle': 3,
                'heat': 0.15,
                'field': 3,
                'emission': 0.9,
                'x_m': model_first_segment_max,
                data_processor.prediction_parameter: predict_first
            }, ignore_index=True
        )
        data_processor.middle_segment = data_processor.middle_segment.append(
            {
                'angle': 3,
                'heat': 0.15,
                'field': 3,
                'emission': 0.9,
                'x_m': model_last_segment_min,
                data_processor.prediction_parameter: predict_last
            }, ignore_index=True
        )
        # Train model for the middle segment
        self.models['middle'] = self.train_model(
            data_processor.middle_segment[['angle', 'heat', 'field', 'emission', 'x_m']],
            data_processor.middle_segment[data_processor.prediction_parameter]
        )

    def predict(self, row, data_processor):
        x_m = row['x_m']
        features = row[['angle', 'heat', 'field', 'emission', 'x_m']].values.reshape(1, -1)
        if x_m <= data_processor.first_cutoff:
            prediction = self.models['first'].predict(features)
        elif x_m >= data_processor.last_cutoff:
            prediction = self.models['last'].predict(features)
        else:
            prediction = self.models['middle'].predict(features)
        return prediction[0]

class Evaluator:
    @staticmethod
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Plotter:
    @staticmethod
    def plot_results(new_data, fusion_data, prediction_parameter,
                     first_cutoff, last_cutoff):
        fig = go.Figure()

        # Dashed black lines for predicted
        # fig.add_trace(go.Scatter(
        #     x=new_data['x_m'],
        #     y=new_data['predicted'],
        #     mode='lines',
        #     name='Predicted',
        #     line=dict(dash='dash', width=2)
        # ))
        fig.add_trace(go.Scatter(
            x=new_data['x_m'],
            y=new_data['predicted_smooth'],
            mode='lines',
            name='Segmented prediction',
            line=dict(dash='dash', width=2)
        ))

        # actual data
        fig.add_trace(go.Scatter(
            x=fusion_data['x_m'],
            y=fusion_data[prediction_parameter],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))

        fig.update_layout(
            template='plotly_white',
            title=f'Predicted vs. Actual Data. Parameter: {prediction_parameter}',
            xaxis_title='x_m [m]',
            yaxis_title='Parameter value',
            legend_title='Traces',
            font=dict(color='black')
        )

        fig.add_shape(
            type='line',
            xref='x', yref='paper',
            x0=first_cutoff, x1=first_cutoff,
            y0=0, y1=1,
            line=dict(color='black', dash='dash', width=2),
            name='Segment line'
        )
        fig.add_shape(
            type='line',
            xref='x', yref='paper',
            x0=last_cutoff, x1=last_cutoff,
            y0=0, y1=1,
            line=dict(color='black', dash='dash', width=2),
            name='Segment line'
        )

        fig.write_html("fig/segmented_approach.html", auto_open=True)

def main():
    data_processor = DataProcessor(
        data_path='data/df_data.pkl',
        fusion_data_path='data/new_output_fusion_sparse.pkl',
        prediction_parameter='Pot'
    )
    data_processor.split_data_segments()

    # Initialize Model Trainer and train segment models
    model_trainer = ModelTrainer(prediction_parameter='Pot')
    model_trainer.train_segment_models(data_processor)

    # Create new data for prediction
    new_data = pd.DataFrame({
        'angle': 3,
        'heat': 0.15,
        'field': 3,
        'emission': 0.9,
        'x_m': data_processor.unique_x_m
    })

    # Predict on segment boundaries
    model_first_segment_max = data_processor.first_segment['x_m'].max()
    predict_first = model_trainer.models['first'].predict(
        new_data[new_data['x_m'] == model_first_segment_max][['angle', 'heat', 'field', 'emission', 'x_m']]
    )[0]
    model_last_segment_min = data_processor.last_segment['x_m'].min()
    predict_last = model_trainer.models['last'].predict(
        new_data[new_data['x_m'] == model_last_segment_min][['angle', 'heat', 'field', 'emission', 'x_m']]
    )[0]

    # Train middle segment model
    model_trainer.train_middle_model(
        data_processor, predict_first, predict_last, model_first_segment_max, model_last_segment_min
    )

    # Make predictions using segmented models
    new_data['predicted'] = new_data.apply(
        lambda row: model_trainer.predict(row, data_processor), axis=1
    )

    # Smooth the predicted values
    new_data['predicted_smooth'] = savgol_filter(new_data['predicted'], 20, 3)

    # Pass the cutoffs to plot_results to draw vertical lines
    Plotter.plot_results(
        new_data,
        data_processor.fusion_data,
        data_processor.prediction_parameter,
        data_processor.first_cutoff,
        data_processor.last_cutoff
    )

    # evaluation
    merged = pd.merge(
        new_data,
        data_processor.fusion_data[['x_m', data_processor.prediction_parameter]],
        on='x_m', how='left'
    )
    merged = merged[merged[data_processor.prediction_parameter] > 10]
    mae = Evaluator.calculate_mae(
        merged[data_processor.prediction_parameter],
        merged['predicted']
    )
    mape = Evaluator.calculate_mape(
        merged[data_processor.prediction_parameter],
        merged['predicted']
    )
    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Absolute Percentage Error:', round(mape, 2), '%')

if __name__ == "__main__":
    main()
