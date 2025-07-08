import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.signal import savgol_filter
from sklearn.inspection import permutation_importance

class DataProcessor:
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)
    
    def get_features_and_target(self, features, target):
        X = self.data[features]
        y = self.data[target]
        return X, y

    def get_unique_values(self, column_name):
        return self.data[column_name].unique()

class ModelTrainer:
    def __init__(self, params, objective='reg:squarederror'):
        self.params = params
        self.model = xgb.XGBRegressor(objective=objective, **params)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X_new):
        return self.model.predict(X_new)
    
    def get_model(self):
        return self.model

class PermutationImportanceAnalyzer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.result = None
        self.perm_df = None
    
    def compute_importance(self, n_repeats=20, random_state=42, scoring='neg_mean_absolute_error'):
        self.result = permutation_importance(
            self.model, self.X, self.y, n_repeats=n_repeats,
            random_state=random_state, scoring=scoring
        )
        self.perm_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance_Mean': self.result.importances_mean,
            'Importance_Std': self.result.importances_std
        }).sort_values('Importance_Mean', ascending=False)
    
    def plot_importance(self):
        fig = go.Figure(go.Bar(
            x=self.perm_df['Importance_Mean'],
            y=self.perm_df['Feature'],
            error_x=dict(type='data', array=self.perm_df['Importance_Std']),
            orientation='h',
            marker=dict(color='grey')
        ))
        
        fig.update_layout(
            title='Permutation Feature Importances',
            xaxis_title='Mean Decrease in MAE',
            yaxis_title='Feature',
            template='plotly_white',
            font=dict(family='Times New Roman', color='black'),
            xaxis=dict(range=[min(self.perm_df['Importance_Mean']) * 1.1, 
                             max(self.perm_df['Importance_Mean']) * 1.1]),
            yaxis=dict(automargin=True)
        )
        
        fig.write_html("fig/feature_importance.html", auto_open=True)

class Evaluator:
    @staticmethod
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Plotter:
    def __init__(self):
        pass
    
    def plot_comparison(self, x_m, actual, predicted_dict, prediction_parameter):
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=x_m,
            y=actual,
            mode='lines',
            name=f'{prediction_parameter} obtained by physical simulation',
            line=dict(color='black', width=3)
        ))
        
        # Add predicted data
        colors = ['red', 'blue', 'green']
        for i, (label, predicted) in enumerate(predicted_dict.items()):
            fig.add_trace(go.Scatter(
                x=x_m,
                y=predicted,
                mode='lines',
                name=label,
                line=dict(dash='dash', width=3, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=f'{prediction_parameter} obtained by physical simulation vs ML models',
            xaxis_title='x [m]',
            yaxis_title=prediction_parameter + ' [V]',
            template='plotly_white',
            font=dict(family='Times New Roman', color='black'),
            legend=dict(
            orientation='v',
            yanchor='bottom',
            y=0.02,
            xanchor='right',
            x=0.98
            ),
        )
        
        fig.write_html("fig/comparison_approach.html", auto_open=True)

class XMModelTrainer:
    def __init__(self, data, prediction_parameter, hyper_params):
        self.data = data
        self.prediction_parameter = prediction_parameter
        self.hyper_params = hyper_params
        self.models = {}
        self.predictions = {}
    
    def train_models(self, new_feature_set):
        unique_x_m = self.data['x_m'].unique()
        for x_m_value in unique_x_m:
            x_m_data = self.data[self.data['x_m'] == x_m_value]
            X_train = x_m_data[['angle', 'heat', 'field', 'emission']]
            y_train = x_m_data[self.prediction_parameter]
            model = xgb.XGBRegressor(**self.hyper_params)
            model.fit(X_train, y_train)
            self.models[x_m_value] = model
            prediction = model.predict(pd.DataFrame([new_feature_set]))
            self.predictions[x_m_value] = prediction[0]
    
    def get_predictions(self):
        return self.predictions

def main():

    data_path = 'data/df_data.pkl'
    data_processor = DataProcessor(data_path)
    features = ['angle', 'heat', 'field', 'emission', 'x_m']
    prediction_parameter = 'Pot'
    X, y = data_processor.get_features_and_target(features, prediction_parameter)
    
    param_dict = {
        'Pot': {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1, 'colsample_bytree': 0.8, 'n_estimators': 300},
        'Tn': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 1, 'colsample_bytree': 1, 'n_estimators': 300},
        'Te': {'learning_rate': 0.3, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 1000},
        'Ti': {'learning_rate': 0.3, 'max_depth': 7, 'subsample': 1, 'colsample_bytree': 1, 'n_estimators': 1000},
        'Vi': {'learning_rate': 0.3, 'max_depth': 3, 'subsample': 1, 'colsample_bytree': 0.8, 'n_estimators': 1000},
        'Vn': {'learning_rate': 0.01, 'max_depth': 5, 'subsample': 1, 'colsample_bytree': 1, 'n_estimators': 600},
        'nn': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 1, 'n_estimators': 1000},
        'E': {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 1, 'colsample_bytree': 1, 'n_estimators': 300},
        'Ve': {'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 1000}
    }
    params = param_dict.get(prediction_parameter, {})
    
    # Model training
    model_trainer = ModelTrainer(params)
    model_trainer.train(X, y)
    
    # Permutation importance
    importance_analyzer = PermutationImportanceAnalyzer(
        model_trainer.get_model(), X, y
    )
    importance_analyzer.compute_importance()
    importance_analyzer.plot_importance()
    
    # Simple XGBoost model
    simple_model_trainer = ModelTrainer({})
    simple_model_trainer.train(X, y)
    
    # Create new data for prediction
    unique_x_m = data_processor.get_unique_values('x_m')
    new_data = pd.DataFrame({
        'angle': 3,
        'heat': 0.15,
        'field': 3,
        'emission': 0.9,
        'x_m': unique_x_m
    })
    
    # Predictions
    single_model_results = model_trainer.predict(new_data)
    single_model_results_smoothed = savgol_filter(single_model_results, 5, 3)
    predicted_simple = simple_model_trainer.predict(new_data)
    
    # Predictions new model per x_m
    hyper_params = {
        'learning_rate': 0.1,
        'max_depth': 2,
        'min_child_weight': 1,
        'subsample': 0.5,
        'colsample_bytree': 0.8,
        'n_estimators': 600
    }
    new_feature_set = {
        'angle': 3,
        'heat': 0.15,
        'field': 3,
        'emission': 0.9
    }
    xm_model_trainer = XMModelTrainer(
        data_processor.data, prediction_parameter, hyper_params
    )
    xm_model_trainer.train_models(new_feature_set)
    predictions_separate_xm = xm_model_trainer.get_predictions()
    
    # Load new fusion_data (validation data)
    new_fusion_data = pd.read_pickle('data/new_output_fusion_sparse.pkl')
    
    # Evaluation for main model
    new_graph_data = pd.DataFrame({
        'x_m': unique_x_m,
        f'Predicted_{prediction_parameter}': single_model_results_smoothed
    })
    merged = pd.merge(new_fusion_data, new_graph_data, on='x_m', how='left')
    mape = Evaluator.calculate_mape(
        merged[prediction_parameter], merged[f'Predicted_{prediction_parameter}']
    )
    print(f'Main Model Mean Absolute Percentage Error: {mape:.2f}%')
    
    # Evaluation for simple model
    new_graph_data_simple = pd.DataFrame({
        'x_m': unique_x_m,
        f'Predicted_{prediction_parameter}_simple': predicted_simple
    })
    merged_simple = pd.merge(new_fusion_data, new_graph_data_simple, on='x_m', how='left')
    mape_simple = Evaluator.calculate_mape(
        merged_simple[prediction_parameter], merged_simple[f'Predicted_{prediction_parameter}_simple']
    )
    print(f'Simple Model Mean Absolute Percentage Error: {mape_simple:.2f}%')
    
    # Evaluation for separate x_m model
    new_graph_data_separate_xm = pd.DataFrame({
        'x_m': list(predictions_separate_xm.keys()),
        f'Predicted_{prediction_parameter}_separate_xm': list(predictions_separate_xm.values())
    })
    merged_separate_xm = pd.merge(new_fusion_data, new_graph_data_separate_xm, on='x_m', how='left')
    mape_separate_xm = Evaluator.calculate_mape(
        merged_separate_xm[prediction_parameter], merged_separate_xm[f'Predicted_{prediction_parameter}_separate_xm']
    )
    print(f'Separate x_m Model Mean Absolute Percentage Error: {mape_separate_xm:.2f}%')
    
    # Plotting
    plotter = Plotter()
    predicted_dict = {
        f'Main Model - {prediction_parameter}': single_model_results_smoothed,
        f'Simple Model - {prediction_parameter} ': predicted_simple,
        f'Separate x_m - {prediction_parameter} ': list(predictions_separate_xm.values())
    }
    plotter.plot_comparison(
        new_fusion_data['x_m'],
        new_fusion_data[prediction_parameter],
        predicted_dict,
        prediction_parameter
    )

    print("Done!")

if __name__ == "__main__":
    main()
