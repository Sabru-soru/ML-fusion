"""
Script used for finding best hyperparameters if we use separate models for each x_m.
We use XGBoost regressor.
"""

import pandas as pd
import xgboost as xgb
import itertools
from statistics import mean

class XGBoostHyperparameterTuner:
    def __init__(self, data_path, target_variable):
        """
        Initializes the tuner by loading the data and setting the target variable.

        Parameters:
        data_path (str): Path to the data file.
        target_variable (str): The name of the target variable in the dataset.
        """
        # Load data
        self.data = pd.read_pickle(data_path)
        self.target = target_variable

        # Calculate the number of unique combinations of ["angle", "heat", "field", "emission"]
        self.num_unique_combinations = self.calculate_unique_combinations()

        # Define hyperparameters and generate combinations
        self.hyper_params = self.define_hyperparameters()
        self.combinations = self.generate_combinations()
        print(f"Testing {len(self.combinations)} hyperparameter combinations.")

        # Initialize a dictionary to store results
        self.results = {'learning_rate': [], 'max_depth': [], 'min_child_weight': [],
                        'subsample': [], 'colsample_bytree': [], 'n_estimators': [], 'error': []}

    def calculate_unique_combinations(self):
        """
        Calculates the number of unique combinations of ["angle", "heat", "field", "emission"] in the data.

        Returns:
        int: Number of unique combinations.
        """
        unique_combinations = self.data[['angle', 'heat', 'field', 'emission']].drop_duplicates()
        num_combinations = len(unique_combinations)
        print(f"Number of unique combinations: {num_combinations}")
        return num_combinations

    def define_hyperparameters(self):
        """
        Defines the hyperparameters to test.

        Returns:
        dict: A dictionary containing lists of hyperparameter values.
        """
        hyper_params = {
            'learning_rate': [0.1, 0.2, 0.3],
            'max_depth': [1, 2, 4],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.1, 0.5, 0.6],
            'colsample_bytree': [0.4, 0.6, 0.8],
            'n_estimators': [500, 600, 700, 900, 1000]
        }
        return hyper_params

    def generate_combinations(self):
        """
        Generates all possible combinations of hyperparameters.

        Returns:
        list: A list of tuples, each containing a combination of hyperparameter values.
        """
        return list(itertools.product(*self.hyper_params.values()))

    def model_function(self, xg_reg, x_m, X_train, y_train, angle, heat, field, emission):
        """
        Trains the model on data where x_m matches and predicts the target variable.

        Parameters:
        xg_reg (XGBRegressor): The XGBoost regressor instance.
        x_m (value): The value of x_m to filter the data.
        X_train (DataFrame): The training features.
        y_train (DataFrame): The training target.
        angle, heat, field, emission: Feature values for prediction.

        Returns:
        float: The predicted value.
        """

        x_temp = X_train[X_train["x_m"] == x_m].drop(["x_m"], axis=1)
        y_temp = y_train[y_train["x_m"] == x_m].drop(["x_m"], axis=1)

        model = xg_reg.fit(x_temp, y_temp)

        temp_target_df = pd.DataFrame({'angle': [angle], 'heat': [heat],
                                       'field': [field], 'emission': [emission]})

        value = model.predict(temp_target_df)[0]
        return value

    def evaluate_hyperparameters(self):
        """
        Evaluates all combinations of hyperparameters and records the results.
        """
        for idx, params in enumerate(self.combinations, start=1):
            # Create an XGBoost regressor with current hyperparameters
            xg_reg = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_jobs=-1,
                learning_rate=params[0],
                max_depth=params[1],
                min_child_weight=params[2],
                subsample=params[3],
                colsample_bytree=params[4],
                n_estimators=params[5]
            )

            error_avg = []

            for iteration in range(self.num_unique_combinations):

                test = self.data.iloc[iteration::self.num_unique_combinations, :].copy()
                train = pd.merge(self.data, test, indicator=True, how='outer') \
                          .query('_merge=="left_only"').drop('_merge', axis=1)

                X_train = train[["angle", "heat", "field", "emission", "x_m"]]
                y_train = train[[self.target, "x_m"]]
                X_test = test[["angle", "heat", "field", "emission", "x_m"]].copy()
                y_test = test[[self.target]].copy()

                X_test['predictions'] = X_test.apply(
                    lambda row: self.model_function(
                        xg_reg, row['x_m'], X_train, y_train,
                        row['angle'], row['heat'], row['field'], row['emission']
                    ), axis=1
                )

                y_test.loc[y_test[self.target] == 0, self.target] = 0.001

                # Calculate relative errors in percent
                rel_errors = (abs(X_test['predictions'] - y_test[self.target]) / y_test[self.target]) * 100
                # Exclude the first and last entries if necessary
                error = rel_errors.iloc[1:-1].mean()
                error_avg.append(error)

            # Compute the average error over all iterations
            avg_error = mean(error_avg)

            self.results['learning_rate'].append(params[0])
            self.results['max_depth'].append(params[1])
            self.results['min_child_weight'].append(params[2])
            self.results['subsample'].append(params[3])
            self.results['colsample_bytree'].append(params[4])
            self.results['n_estimators'].append(params[5])
            self.results['error'].append(avg_error)

            print(f"Completed {idx}/{len(self.combinations)} combinations.")

    def save_results(self, output_file):
        """
        Saves the results to a CSV file and prints the best hyperparameters.

        Parameters:
        output_file (str): The file path to save the results.
        """

        df_results = pd.DataFrame(self.results)

        sorted_results = df_results.sort_values(by="error")

        sorted_results.to_csv(output_file, index=False)

        best = sorted_results.iloc[0]
        print("Best hyperparameters found:")
        print(best)

    def run(self, output_file):
        """
        Runs the hyperparameter tuning process and saves the results.

        Parameters:
        output_file (str): The file path to save the results.
        """
        self.evaluate_hyperparameters()
        self.save_results(output_file)


if __name__ == "__main__":
    
    target_variable='Pot'

    tuner = XGBoostHyperparameterTuner(data_path='data/df_data.pkl', target_variable=target_variable)
    
    tuner.run(output_file='testing_parameter_{target_variable}.csv')
