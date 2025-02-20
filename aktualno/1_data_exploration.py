import pandas as pd
import plotly.graph_objects as go

class DataExplorer:
    """
    A class used to process and visualize data.

    ...

    Attributes
    ----------
    data_path : str
        a formatted string to define the path of the data file
    prediction_parameter : str
        a string to define the parameter to predict
    df : DataFrame
        a pandas DataFrame to store the data
    fig : Figure
        a plotly Figure to store the plot

    Methods
    -------
    load_data():
        Loads the data from the file specified in data_path.
    process_new_simulation(file_path):
        Processes a new simulation and saves the results.
    generate_plot():
        Generates a plot of the data.
    show_plot():
        Displays the plot.
    save_plot(file_path):
        Saves the plot to the specified file.
    describe_data():
        Returns a statistical description of the data.
    """

    def __init__(self, data_path, prediction_parameter):
        """Initializes DataExplorer with the data path and prediction parameter."""
        self.data_path = data_path
        self.prediction_parameter = prediction_parameter
        self.df = None
        self.fig = None

    def load_data(self):
        """Loads the data from the file specified in data_path."""
        self.df = pd.read_pickle(self.data_path)
        self.df = self.df[['angle', 'heat', 'field', 'emission', 'x_m', self.prediction_parameter]]

    def process_new_simulation(self, file_path):
        """
        Processes a new simulation and saves the results.

        Parameters:
            file_path (str): The path of the file containing the simulation results.
        """
        unique_x_m = self.df['x_m'].unique()
        new_fusion_results = pd.read_excel(file_path, sheet_name='all_together')
        new_fusion_results['x_m'] = new_fusion_results['x_m']*10
        new_fusion_results = new_fusion_results[new_fusion_results['x_m'].isin(unique_x_m)]
        new_fusion_results.drop_duplicates(subset='x_m', inplace=True)
        new_fusion_results.to_pickle('data/new_output_fusion_sparse.pkl')

    def generate_plot(self):
        """Generates a plot of the data."""
        unique_combinations = self.df.drop_duplicates(subset=['angle', 'heat', 'field', 'emission'])
        self.fig = go.Figure()

        for index, row in unique_combinations.iterrows():
            subset = self.df[
                (self.df['angle'] == row['angle']) &
                (self.df['heat'] == row['heat']) &
                (self.df['field'] == row['field']) &
                (self.df['emission'] == row['emission'])
            ]


            if (row['angle'] == 6 and row['field'] == 2.2 and 
                row['heat'] == 0 and row['emission'] == 0.8):
                line_style = dict(color='black', width=4)
            else:
                line_style = dict(dash='dash', width=2)

            self.fig.add_trace(go.Scatter(
                x=subset['x_m'], 
                y=subset[self.prediction_parameter],
                mode='lines', 
                name=f'Params: {row["angle"]}, {row["heat"]}, {row["field"]}, {row["emission"]}',
                opacity=0.7,
                line=line_style
            ))

        self.fig.update_layout(
            template='plotly_white',
            title=f'Actual Data. Parameter: {self.prediction_parameter}',
            xaxis_title='x [m]',
            yaxis_title='Value',
            legend_title='angle, heat, field, emission',
            font=dict(color='black'),
        )

    def describe_data(self):
        """Returns a statistical description of the data."""
        return self.df.describe().round(2)


if __name__ == "__main__":
    explorer = DataExplorer('data/df_data.pkl', prediction_parameter='Pot')
    explorer.load_data()
    explorer.process_new_simulation('data/new_output_fusion.xlsx')
    explorer.generate_plot()
    # explorer.fig.show()
    explorer.fig.write_html("fig/actual_data.html", auto_open=True)
    print(explorer.describe_data())