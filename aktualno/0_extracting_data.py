"""
@author: urosu
"""
import pandas as pd
import numpy as np
import time

class DataExtractor:
    """
    A class used to extract data from an Excel file and transform it into a DataFrame.

    Attributes
    ----------
    file_path : str
        a formatted string to print out the file_path
    df1 : DataFrame
        a pandas DataFrame containing the data from the '1.5 degree' sheet
    df2 : DataFrame
        a pandas DataFrame containing the data from the '4 degree' sheet
    df3 : DataFrame
        a pandas DataFrame containing the data from the '6 degree' sheet

    Methods
    -------
    create_initial_dataframe()
        Creates an initial DataFrame with predefined values for 'angle', 'heat', 'field', and 'emission'.
    add_length_tokamak(df)
        Adds the 'x_[m]' column to the DataFrame based on the 'x  (m)' column in df1.
    add_target(target, df)
        Adds a target column to the DataFrame based on the target parameter.
    add_all_targets(df)
        Adds all target columns to the DataFrame.
    save_data(df, output_file)
        Saves the DataFrame to a pickle file.
    """
    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the DataExtractor object.

        Parameters
        ----------
            file_path : str
                file path of the Excel file
        """
        self.file_path = file_path
        self.df1 = pd.read_excel(self.file_path, sheet_name='1.5 degree')
        self.df2 = pd.read_excel(self.file_path, sheet_name='4 degree')
        self.df3 = pd.read_excel(self.file_path, sheet_name='6 degree')

    def create_initial_dataframe(self):
        """
        Creates an initial DataFrame with predefined values for 'angle', 'heat', 'field', and 'emission'.
        
        Returns
        -------
            df : DataFrame
                A DataFrame with columns 'angle', 'heat', 'field', 'emission' and predefined values.
        """
        angle = [6]*7 + [4]*7 + [1.5]*7
        heat = [0, 0.1, 0.2] + [0]*4 + [0, 0.1, 0.2] + [0]*4 + [0, 0.1, 0.2] + [0]*4
        field = [2.2]*3 + [1, 4] + [2.2]*5 + [1, 4] + [2.2]*5 + [1, 4] + [2.2]*2
        emission = [0.8]*5 + [0, 1] + [0.8]*5 + [0, 1] + [0.8]*5 + [0, 1]

        df = pd.DataFrame(list(zip(angle, heat, field, emission)), columns=['angle', 'heat', 'field', 'emission'])
        return df

    def add_length_tokamak(self, df):
        """
        Adds the 'x_m' column to the DataFrame based on the 'x  (m)' column in df1.

        Parameters
        ----------
            df : DataFrame
                The DataFrame to which the 'x_m' column will be added.

        Returns
        -------
            df : DataFrame
                The DataFrame with the added 'x_m' column.
        """
        length_tokamak = pd.DataFrame(self.df1["x  (m)"]*10) #Takes the length of the wall. Error in input data - need to multiply by 10
        df = pd.concat([df]*len(length_tokamak), ignore_index=True)
        length_tokamak = pd.DataFrame(np.repeat(length_tokamak.values, 21, axis=0))
        df.loc[:,"x_m"] = length_tokamak
        return df

    def add_target(self, target, df):
        """
        Adds a target column to the DataFrame based on the target parameter.

        Parameters
        ----------
            target : str
                The name of the target column to add.
            df : DataFrame
                The DataFrame to which the target column will be added.

        Returns
        -------
            df : DataFrame
                The DataFrame with the added target column.
        """
        target_list = [self.df3[f'{target}'], self.df3[f'{target} (heat 0.1)'], self.df3[f'{target} (heat 0.2)'], self.df3[f'{target} (1 T)'], self.df3[f'{target} (4 T)'], self.df3[f'{target} (recycling 0)'], self.df3[f'{target} (recycling 1)'],
                       self.df2[f'{target}'], self.df2[f'{target} (heat 0.1)'], self.df2[f'{target} (heat 0.2)'], self.df2[f'{target} (1 T)'], self.df2[f'{target} (4 T)'], self.df2[f'{target} (recycling 0)'], self.df2[f'{target} (recycling 1)'],
                       self.df1[f'{target}'], self.df1[f'{target} (heat 0.1)'], self.df1[f'{target} (heat 0.2)'], self.df1[f'{target} (1 T)'], self.df1[f'{target} (4 T)'], self.df1[f'{target} (recycling 0)'], self.df1[f'{target} (recycling 1)']]

        df_target = pd.DataFrame(target_list)
        df_target = df_target.reset_index(level=0)
        df_target = pd.melt(df_target, id_vars='index')
        df.loc[:,f'{target}'] = df_target["value"]
        return df

    def add_all_targets(self, df):
        """
        Adds all target columns to the DataFrame.

        Parameters
        ----------
            df : DataFrame
                The DataFrame to which the target columns will be added.

        Returns
        -------
            df : DataFrame
                The DataFrame with the added target columns.
        """
        targets = ['ne','ni','nn','Te','Ti','Tn','Ve','Vi','Vn','E','Pot']
        for target in targets:
            df = self.add_target(target, df)
        return df

    def save_sparse_data(self, df, output_file, sparsity=200):
        """
        Saves a sparsified version of the DataFrame to a pickle file.

        The DataFrame is sparsified by selecting every nth unique value of the 'x_m' column, where n is the sparsity.

        Parameters
        ----------
            df : DataFrame
                The DataFrame to save.
            output_file : str
                The file path of the output pickle file.
            sparsity : int, optional
                The sparsity of the data. Every nth unique value of the 'x_m' column is selected, where n is the sparsity. By default, 200.
        """
        df_sparse = df.loc[df['x_m'].isin(df['x_m'].unique()[::sparsity]), :]
        df_sparse.to_pickle(output_file)

if __name__ == "__main__":
    start_time = time.time()
    extractor = DataExtractor('data/output_fusion.xlsx')
    df = extractor.create_initial_dataframe()
    df = extractor.add_length_tokamak(df)
    df = extractor.add_all_targets(df)
    extractor.save_sparse_data(df, 'data/df_data.pkl', sparsity=200)
    print('Execution time: ', time.time() - start_time)