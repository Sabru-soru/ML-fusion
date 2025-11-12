# xgb_hyperparameter_tuner.py

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import numpy as np
import itertools

class XGBLOCOHyperparameterTuner:
    def __init__(self, data_path, param_grid, prediction_parameters):
        self.df = pd.read_pickle(data_path)
        self.param_grid = param_grid
        self.prediction_parameters = prediction_parameters
        self.unique_combinations = self.df[['angle', 'heat', 'field', 'emission']].drop_duplicates()
        self.X = self.df[['angle', 'heat', 'field', 'emission', 'x_m']]
        self.hyperparameter_combinations = list(itertools.product(*self.param_grid.values()))
        self.hyperparameter_keys = list(self.param_grid.keys())

    def tune(self):
        for prediction_parameter in self.prediction_parameters:
            print(f"Tuning for parameter: {prediction_parameter}")
            y = self.df[prediction_parameter]
            overall_results = []
            for i, hyperparameters in enumerate(self.hyperparameter_combinations):
                print(f'Hyperparameter combination {i+1}/{len(self.hyperparameter_combinations)}')
                params = dict(zip(self.hyperparameter_keys, hyperparameters))
                mape_scores = self._leave_one_curve_out_validation(params, y)
                avg_mape = np.mean(mape_scores)
                overall_results.append({'Params': params, 'Avg_MAPE': avg_mape})
            self._save_results(overall_results, prediction_parameter)

    def _leave_one_curve_out_validation(self, params, y):
        mape_scores = []
        for _, row in self.unique_combinations.iterrows():
            test_condition = (
                (self.df['angle'] == row['angle']) &
                (self.df['heat'] == row['heat']) &
                (self.df['field'] == row['field']) &
                (self.df['emission'] == row['emission'])
            )
            X_train = self.X.loc[~test_condition]
            y_train = y.loc[~test_condition]
            X_test = self.X.loc[test_condition]
            y_test = y.loc[test_condition].copy()
            y_test[y_test == 0] = 0.0001

            xgb_model = XGBRegressor(**params)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)

            mape = mean_absolute_percentage_error(
                y_test.iloc[6:-6].reset_index(drop=True),
                xgb_pred[6:-6]
            ) * 100
            mape_scores.append(mape)
        return mape_scores

    def _save_results(self, results, prediction_parameter):
        results_sorted = sorted(results, key=lambda x: x['Avg_MAPE'])
        best_params = results_sorted[0]
        print(f"Best hyperparameters for {prediction_parameter}: {best_params}")
        pd.DataFrame(results_sorted).to_csv(
            f'data/xgb_LOO_results_{prediction_parameter}.csv',
            index=False
        )

if __name__ == "__main__":
    param_grid = {
        'n_estimators': [300, 500, 600, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    prediction_parameters = ['Pot', 'Ti', 'Vi', 'Vn', 'nn', 'E', 'Ve']
    tuner = XGBLOCOHyperparameterTuner(
        data_path='data/df_data.pkl',
        param_grid=param_grid,
        prediction_parameters=prediction_parameters
    )
    tuner.tune()
