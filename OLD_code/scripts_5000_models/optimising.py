# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:56:16 2023

@author: urosu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:27:32 2022

@author: urosu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

data=data.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)


#For faster tuning of hyperparameters
data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::200]), :]

target="Pot"

#%%
test1=data.iloc[3::21, :]

test=test1

train=pd.merge(data,test, indicator=True, how='outer')\
         .query('_merge=="left_only"')\
         .drop('_merge', axis=1)


X_train=train[["angle","heat","field","emission","x_m"]]
y_train=train[[target,"x_m"]]

X_test=test[["angle","heat","field","emission","x_m"]]
y_test=test[[target]]



# xg_reg = xgb.XGBRegressor(
#         objective = 'reg:squarederror',
#         learning_rate=0.5,#0.25
#         max_depth = 2, #5
#         min_child_weight = 1,#3
#         subsample = 0.5,#0.5
#         colsample_bytree = 0.5,#0.2
#         n_estimators = 200)

#%%
import itertools
temp_target_df= pd.DataFrame(columns=["angle","heat","field","emission"], dtype='int8')
hyper_params = {
    'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'max_depth': [1,2,3,4,5],
    'min_child_weight': [1, 2, 3,],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'n_estimators' : [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

a = hyper_params.values()
combinations = list(itertools.product(*a))
print("Testing",len(combinations),"values.")

def model_function(angle, heat, field, emission, x_m):
    x_temp=X_train[X_train["x_m"]==x_m]
    x_temp=x_temp.drop(["x_m"],axis=1)
    y_temp=y_train[y_train["x_m"]==x_m]
    y_temp=y_temp.drop(["x_m"],axis=1)
    model=xg_reg.fit(x_temp, y_temp)
    
    temp_target_df.loc[0] = [angle,heat,field,emission]
    value=model.predict(temp_target_df)[0]
    
    return value
i=0
values = {'learning_rate': [], 'max_depth': [], 'min_child_weight': [], 'subsample': [], 'colsample_bytree': [], 'n_estimators': [], 'error': []}
for c in combinations:
    xg_reg = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            learning_rate=c[0],
            max_depth = c[1],
            min_child_weight = c[2],
            subsample = c[3],
            colsample_bytree = c[4],
            n_estimators = c[5])
    
    X_test['predictions'] = X_test.apply(lambda row: model_function(row['angle'], row['heat'], row['field'], row['emission'], row['x_m']), axis=1)
    
    # error=abs(X_test['predictions']-y_test[target]).sum()
    
    #Because we divide
    y_test.loc[y_test[target]==0,target]=0.001
    
    rel_errors=(abs(X_test['predictions']-y_test[target])/y_test[target])*100
    error=rel_errors.iloc[100:-100].mean()
    
    
    values['learning_rate'].append(c[0])
    values['max_depth'].append(c[1])
    values['min_child_weight'].append(c[2])
    values['subsample'].append(c[3])
    values['colsample_bytree'].append(c[4])
    values['n_estimators'].append(c[5])
    values['error'].append(error)
    
    i+=1
    print(i)
    # X_test['predictions'].plot()
    # y_test[target].plot()
    # plt.show()
    # print(c)
#%%
df=pd.DataFrame(values)
sort_df=df.sort_values(["error"])
df.to_csv('Tested_Pot.csv') 

best=df[df["error"]==df["error"].min()]
print(best)
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





