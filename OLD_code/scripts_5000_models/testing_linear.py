# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:19:02 2023

@author: urosu
"""

from sklearn import linear_model


import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import itertools
from statistics import mean 

data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

data=data.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)


#For faster tuning of hyperparameters
data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::50]), :]

target="Pot"

#%%HYPERPARAMETERS

temp_target_df= pd.DataFrame(columns=["angle","heat","field","emission"], dtype='int8')

reg = linear_model.LinearRegression()
#%%FUNCTION

def model_function(angle, heat, field, emission, x_m):
    x_temp=X_train[X_train["x_m"]==x_m]
    x_temp=x_temp.drop(["x_m"],axis=1)
    y_temp=y_train[y_train["x_m"]==x_m]
    y_temp=y_temp.drop(["x_m"],axis=1)
    
    model=reg.fit(x_temp, y_temp)
    
    temp_target_df.loc[0] = [angle,heat,field,emission]
    value=model.predict(temp_target_df)[0]
    
    return value

#%%

i=0
values = {'learning_rate': [], 'max_depth': [], 'min_child_weight': [], 'subsample': [], 'colsample_bytree': [], 'n_estimators': [], 'error': []}

    
error_avg=[]

for iteration in range(21):
    test=data.iloc[iteration::21, :]

    train=pd.merge(data,test, indicator=True, how='outer')\
             .query('_merge=="left_only"')\
             .drop('_merge', axis=1)

    X_train=train[["angle","heat","field","emission","x_m"]]
    y_train=train[[target,"x_m"]]
    X_test=test[["angle","heat","field","emission","x_m"]]
    y_test=test[[target]]
    
    
    
    X_test['predictions'] = X_test.apply(lambda row: model_function(row['angle'], row['heat'], row['field'], row['emission'], row['x_m']), axis=1)

    
    #Because we divide and so we dont get infinity
    y_test.loc[y_test[target]==0,target]=0.0001
    
    #Get in percent
    rel_errors=(abs(X_test['predictions']-y_test[target])/y_test[target])*100
    error=rel_errors.iloc[10:-10].mean()
    
    plt.plot(X_test['x_m'], X_test['predictions'],label="Prediction")
    plt.plot(X_test['x_m'], y_test[target],label="True")
    plt.legend()
    plt.xlabel('Length [m]')
    plt.ylabel('Pot [...]')
    plt.show()
    
    error_avg.append(error)
    



#%% EXPORT
# df=pd.DataFrame(values)
# # sort_df=df.sort_values(["error"])
# # df.to_csv('Tested_further_Pot.csv') 

# best=df[df["error"]==df["error"].min()]
# print(best)
# #%%
# fig = plt.figure()
# plt.plot(X_test['x_m'], X_test['predictions'],label="Prediction")
# plt.plot(X_test['x_m'], y_test[target],label="True")

# plt.xlabel('Length [m]')
# plt.ylabel('Pot [...]')
# #plt.gcf().autofmt_xdate()
# plt.grid()
# plt.tight_layout()
# #fig.savefig("3_1.png", dpi = 200)
# plt.legend()
# #plt.savefig(f"figs/{picture}",dpi=100)
# plt.show()

# error=abs(X_test['predictions']-y_test[target]).sum()
# print(error)





