# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:13:06 2023

@author: urosu
"""

import h2o
from h2o.automl import H2OAutoML
# import pandas as pd
# from sklearn.model_selection import train_test_split

# Start the H2O cluster (locally)
h2o.init()


# data=pd.read_csv('df_data.csv')
# data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

df = h2o.import_file('df_h20.csv')

y="Pot"
splits = df.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]


#%%
# Run AutoML for 20 base models
aml = H2OAutoML(max_models=200, seed=1)
aml.train(y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# The leader model is stored here
aml.leader

#%%
# To generate predictions on a test set, you can make predictions
# directly on the `H2OAutoML` object or on the leader model
# object directly
preds = aml.predict(test)

#%%
# Explain leader model & compare with all AutoML models
exa = aml.explain(test)

#%%
# Explain a single H2O model (e.g. leader model from AutoML)
exm = aml.leader.explain(test)

#%%
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
lb