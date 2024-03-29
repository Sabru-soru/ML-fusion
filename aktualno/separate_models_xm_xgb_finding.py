"""
Script used for finding best hyperprameters if we use separate models for each x_m.
We use XGBoost regressor.
"""
#%%
import pandas as pd
import xgboost as xgb
import itertools
from statistics import mean 
data=pd.read_csv('data/df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #could change this in extracting_data.py

data=data.drop_duplicates(subset=["angle","heat","field","emission","x_m"], keep=False)

#For faster tuning of hyperparameters
data=data.loc[data['x_m'].isin(data["x_m"].unique()[0::500]), :]

target="Te"

#%%HYPERPARAMETERS
temp_target_df= pd.DataFrame(columns=["angle","heat","field","emission"], dtype='int8')
hyper_params = {
    'learning_rate': [0.1,0.2,0.3],
    'max_depth': [1,2,4],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.1, 0.5, 0.6],
    'colsample_bytree': [0.4, 0.6, 0.8],
    'n_estimators' : [500, 600, 700, 900, 1000]
}

a = hyper_params.values()
combinations = list(itertools.product(*a))
print("Testing",len(combinations),"values.")

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
i=0
values = {'learning_rate': [], 'max_depth': [], 'min_child_weight': [], 'subsample': [], 'colsample_bytree': [], 'n_estimators': [], 'error': []}
for c in combinations:
    xg_reg = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            n_jobs=-1,
            learning_rate=c[0],
            max_depth = c[1],
            min_child_weight = c[2],
            subsample = c[3],
            colsample_bytree = c[4],
            n_estimators = c[5])
    
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
        
        X_test['predictions'] = X_test.apply(lambda row: model_function(row['angle'], row['heat'], row['field'], row['emission'], row['x_m'], xg_reg), axis=1)

        #Because we divide and so we dont get infinity
        y_test.loc[y_test[target]==0,target]=0.001
        
        #Get in percent
        rel_errors=(abs(X_test['predictions']-y_test[target])/y_test[target])*100
        error=rel_errors.iloc[1:-1].mean()
        
        error_avg.append(error)
    
    error=mean(error_avg)
        
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


#%% EXPORT
df=pd.DataFrame(values)
sort_df=df.sort_values(["error"])
sort_df.to_csv('Tested_further_Te.csv') 

best=df[df["error"]==df["error"].min()]
print(best)