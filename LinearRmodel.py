# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:12:05 2022

@author: urosu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.linear_model import LinearRegression

data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

X=data[["angle","heat","field","emission","x_m"]]
# y=data.drop(["angle","heat","field","emission","x_m"], axis=1)
y=data["Pot"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



# def hyperParameterTuning(X_train, y_train):
#     param_tuning = {
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5, 7, 10],
#         'min_child_weight': [1, 3, 5],
#         'subsample': [0.5, 0.7],
#         'colsample_bytree': [0.5, 0.7],
#         'n_estimators' : [100, 200, 500],
#         'objective': ['reg:squarederror']
#     }

#     xgb_model = xgb.XGBRegressor()

#     gsearch = GridSearchCV(estimator = xgb_model,
#                            param_grid = param_tuning,                        
#                            #scoring = 'neg_mean_absolute_error', #MAE
#                            #scoring = 'neg_mean_squared_error',  #MSE
#                            cv = 3,
#                            n_jobs = -1,
#                            verbose = 1)

#     gsearch.fit(X_train,y_train)

#     return gsearch.best_params_

# best_hyperparams=hyperParameterTuning(X_train, y_train)

# #{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 500, 'objective': 'reg:squarederror', 'subsample': 0.7}
# print("The best hyperparameters are : ","\n")
# print(best_hyperparams)

#%%
# xg_reg = xgb.XGBRegressor(
#         objective = 'reg:squarederror',
#         colsample_bytree = 0.7,
#         learning_rate = 0.1,
#         max_depth = 10,
#         min_child_weight = 3,
#         n_estimators = 500,
#         subsample = 0.7)


reg = linear_model.LinearRegression().fit(X_train, y_train)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
# xg_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)


# xg_reg = xgb.XGBRegressor()#objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(X_train,y_train)
# #xg_reg.fit(X,y)

preds = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# score = xg_reg.score(X_train,y_train)  
# print("Training score: ", score)

# scores = cross_val_score(xg_reg, X_train, y_train,cv=10)
# print("Mean cross-validation score: %.2f" % scores.mean())

reg = linear_model.LinearRegression().fit(X, y)

#%%
#Try to get new values from the model
new_values=pd.DataFrame()
angle_new=[5]
heat_new=[0.2]
field_new=[2.2]
emission_new=[0.8]

new_values = pd.DataFrame(angle_new, columns=['angle'])
new_values['heat'] = heat_new
new_values['field'] = field_new
new_values['emission'] = emission_new
new_values=pd.concat([new_values]*len(data["x_m"].unique()), ignore_index=True)
new_values['x_m'] = (data["x_m"].unique())


prediction = reg.predict(new_values)
# print("Prediction for new values: %f" % (prediction))
# print(format(prediction[0],'.1E'))

#%%
fig = plt.figure()
plt.plot(data["x_m"].unique(), prediction,label="Prediction, 5,0.2,2.2,0.8")
plt.xlabel('Length [m]')
plt.ylabel('Pot [...]')
#plt.gcf().autofmt_xdate()
plt.grid()
plt.tight_layout()
#fig.savefig("3_1.png", dpi = 200)

#SOMETHING IS NOT GOOD
x=data[(data["angle"]==4) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["x_m"]
y=data[(data["angle"]==4) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["Pot"]
plt.plot(x, y, label="Reference, 4,0.2,2.2,0.8",alpha=0.4)

x=data[(data["angle"]==6) & (data["heat"]==0) & (data["field"]==2.2) & (data["emission"]==0.8)]["x_m"]
y=data[(data["angle"]==6) & (data["heat"]==0.2) & (data["field"]==2.2) & (data["emission"]==0.8)]["Pot"]
plt.plot(x, y, label="Reference, 6,0.2,2.2,0.8",alpha=0.4)

plt.legend()
plt.show()


#TODO
#Dolžina x_m je nekej 
