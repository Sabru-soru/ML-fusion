# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:43:36 2023

@author: urosu
"""

"angle","heat","field","emission","x_m"
# 3, 0.15, 3, 0.9
#%%
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import itertools
from statistics import mean 
#%%
# Read the data from the Excel file
file_path = "../new_output_fusion.xlsx"
df = pd.read_excel(file_path)
df = df.rename(columns={'x  (m)': 'x_m'})
df=df.loc[df['x_m'].isin(df["x_m"].unique()[0::20]), :]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['x_m'], df['Pot'], '-o', label='Pot')
plt.xlabel('x (m)')
plt.ylabel('Pot')
plt.title('Pot vs x (m)')
plt.legend()
plt.grid(True)
plt.show()


#%%
data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py

data=data.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)

#For faster tuning of hyperparameters
# data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::20]), :]

target="Pot"
#%%
#keep only columns angle, heat, field, emission, x_m and target
data=data[["angle","heat","field","emission","x_m",target]]

#%%
data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::20]), :]
#%%
#save data to excel
data.to_excel("data_cleaned_interpreter.xlsx", index=False)


#%%

#%%HYPERPARAMETERS
xg_reg = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth = 2,
        min_child_weight = 1,
        subsample = 0.5,
        colsample_bytree = 0.8,
        n_estimators = 600)


temp_target_df= pd.DataFrame(columns=["angle","heat","field","emission"], dtype='int8')
# hyper_params = {
#     'learning_rate': [0.1],
#     'max_depth': [2],
#     'min_child_weight': [1],
#     'subsample': [0.5],
#     'colsample_bytree': [0.8],
#     'n_estimators' : [600]
# }

# a = hyper_params.values()
# combinations = list(itertools.product(*a))
# print("Testing",len(combinations),"values.")

#%%FUNCTION

def model_function(angle, heat, field, emission, x_m, xg_reg):
    x_temp=X_train[X_train["x_m"]==x_m]
    x_temp=x_temp.drop(["x_m"],axis=1)
    y_temp=y_train[y_train["x_m"]==x_m]
    y_temp=y_temp.drop(["x_m"],axis=1)
    model=xg_reg.fit(x_temp, y_temp)
    
    temp_target_df.loc[0] = [angle,heat,field,emission]
    value=model.predict(temp_target_df)[0]
    
    return value

#%%

test=df[['x_m','Pot']]
test["angle"]=3
test["heat"]=0.15
test["field"]=3
test["emission"]=0.9

X_test=test[["angle","heat","field","emission","x_m"]]
y_test=test[[target]]

X_train=data[["angle","heat","field","emission","x_m"]]
y_train=data[[target,"x_m"]]

#%% train and predict
X_test=X_test.head().copy()

#tukaj učim še na testu in moram preuredit
X_test['predictions'] = X_test.apply(lambda row: model_function(row['angle'], row['heat'], row['field'], row['emission'], row['x_m'], xg_reg), axis=1)

#%%
#Because we divide and so we dont get infinity
y_test.loc[y_test[target]==0,target]=0.00001

#Get in percent
rel_errors=(abs(X_test['predictions']-y_test[target])/y_test[target])*100
error=rel_errors.iloc[10:-10].mean()

plt.plot(X_test['x_m'], X_test['predictions'],label="Prediction")
plt.plot(X_test['x_m'], y_test[target],label="True")
plt.legend()
plt.xlabel('Length [m]')
plt.ylabel('Pot [...]')
plt.show()

#%% EXPORT
