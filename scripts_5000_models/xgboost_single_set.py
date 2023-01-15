# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:27:32 2022

@author: urosu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

data=data.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)


#For faster tuning of hyperparameters
data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::50]), :]

target="Pot"

#%%
test1=data.iloc[3::21, :]
# test2=data.iloc[8::21, :]
# test3=data.iloc[15::21, :]
# test=pd.concat([test1,test2,test3],ignore_index=True)
test=test1.copy()

# test=test.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)

train=pd.merge(data,test, indicator=True, how='outer')\
         .query('_merge=="left_only"')\
         .drop('_merge', axis=1)
         
train=train.copy()
# train=train.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)

X_train=train[["angle","heat","field","emission","x_m"]]
y_train=train[[target,"x_m"]]

X_test=test[["angle","heat","field","emission","x_m"]]
y_test=test[[target]]

# xg_reg = xgb.XGBRegressor(
#         objective = 'reg:squarederror',
#         learning_rate=0.1,#0.25
#         max_depth = 3, #5
#         min_child_weight = 3,#3
#         subsample = 0.7,#0.5
#         colsample_bytree = 0.6,#0.2
#         n_estimators = 900)

xg_reg = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        learning_rate=0.1,#0.25
        max_depth = 1, #5
        min_child_weight = 1,#3
        subsample = 0.1,#0.5
        colsample_bytree = 0.1,#0.2
        n_estimators = 500)

# tok_length=data["x_m"].unique()

#%%
temp_target_df= pd.DataFrame(columns=["angle","heat","field","emission"], dtype='int8')
def model_function(angle, heat, field, emission, x_m):
    x_temp=X_train[X_train["x_m"]==x_m]
    x_temp=x_temp.drop(["x_m"],axis=1)
    y_temp=y_train[y_train["x_m"]==x_m]
    y_temp=y_temp.drop(["x_m"],axis=1)
    model=xg_reg.fit(x_temp, y_temp)
    
    
    temp_target_df.loc[0] = [angle,heat,field,emission]
    value=model.predict(temp_target_df)[0]
    
    return value

X_test['predictions'] = X_test.apply(lambda row: model_function(row['angle'], row['heat'], row['field'], row['emission'], row['x_m']), axis=1)

#%%
#Even Faster?
#Not working
# X_test['predictions2'] = model_function(X_test['angle'], X_test['heat'], X_test['field'], X_test['emission'], X_test['x_m'])


#%%
fig = plt.figure()
plt.plot(X_test['x_m'], X_test['predictions'],label="Prediction")
plt.plot(X_test['x_m'], y_test[target],label="True")

plt.xlabel('Length [m]')
plt.ylabel(f'{target} [...]')
#plt.gcf().autofmt_xdate()
plt.grid()
plt.tight_layout()
#fig.savefig("3_1.png", dpi = 200)
plt.legend()
#plt.savefig(f"figs/{picture}",dpi=100)
plt.show()

#Because we divide
y_test.loc[y_test[target]==0,target]=0.001

error=abs(X_test['predictions']-y_test[target]).sum()


#Need to change this 100 if we consider less data
rel_errors=(abs(X_test['predictions']-y_test[target])/y_test[target])*100
rel_errors.iloc[10:-10].plot()
plt.show()
print(rel_errors.iloc[10:-10].mean())

rel_error=(abs(X_test['predictions']-y_test[target])/y_test[target]).mean()*100



print(error)
print(rel_error)

# #SOMETHING IS NOT GOOD
# x=data[(data["angle"]==4) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["x_m"]
# y_plot=data[(data["angle"]==4) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["Pot"]
# plt.plot(x, y_plot, label="Reference, 4,0.2,2.2,0.8",alpha=0.4)

# x=data[(data["angle"]==6) & (data["heat"]==0) & (data["field"]==2.2) & (data["emission"]==0.8)]["x_m"]
# y_plot=data[(data["angle"]==6) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["Pot"]
# plt.plot(x, y_plot, label="Reference, 6,0.2,2.2,0.8",alpha=0.4)
# #plt.title(f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[5]}")
# plt.legend()
# #plt.savefig(f"figs/{picture}",dpi=100)
# plt.show()



#testing(p1, p2, p3, p4, p5, p6)
#TODO
#Dolžina x_m je nekej 

#%%
# xnew = np.linspace(X_test['x_m'].min(), X_test['x_m'].max(), num=150)
# from scipy.interpolate import UnivariateSpline
# spl = UnivariateSpline(X_test['x_m'], X_test['predictions'])
# spl.set_smoothing_factor(30)


# plt.plot(xnew, spl(xnew), 'g')

# plt.show()

# #%%
# from scipy.interpolate import interp1d
# f2 = interp1d(X_test['x_m'], X_test['predictions'], kind='cubic')

# xnew = np.linspace(X_test['x_m'].min(), X_test['x_m'].max(), num=400)
# plt.plot(xnew,f2(xnew))