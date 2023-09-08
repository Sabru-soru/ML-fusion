# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:07:04 2022

@author: urosu
"""

import pandas as pd
import time
import numpy as np

start_time = time.time()
"load file"
df1 = pd.read_excel('data/output_fusion.xlsx', sheet_name='1.5 degree')
df2 = pd.read_excel('data/output_fusion.xlsx', sheet_name='4 degree')
df3 = pd.read_excel('data/output_fusion.xlsx', sheet_name='6 degree')

#%%
df = pd.DataFrame()
angle=[6,6,6,6,6,6,6,4,4,4,4,4,4,4,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
heat=[0,0.1,0.2,0,0,0,0,0,0.1,0.2,0,0,0,0,0,0.1,0.2,0,0,0,0]
field=[2.2,2.2,2.2,1,4,2.2,2.2,2.2,2.2,2.2,1,4,2.2,2.2,2.2,2.2,2.2,1,4,2.2,2.2]
emission=[0.8,0.8,0.8,0.8,0.8,0,1,0.8,0.8,0.8,0.8,0.8,0,1,0.8,0.8,0.8,0.8,0.8,0,1]

df = pd.DataFrame(angle, columns=['angle'])
df['heat'] = heat
df['field'] = field
df['emission'] = emission
#%%
#Takes the length of the wall. Error in input data - need to mutiply by 10
length_tokamak=pd.DataFrame(df1["x  (m)"]*10)

df=pd.concat([df]*len(length_tokamak), ignore_index=True) # Ignores the index

#Add the column 
length_tokamak=pd.DataFrame(np.repeat(length_tokamak.values, 21, axis=0))
df.loc[:,"x_[m]"]=length_tokamak

#%%
#Add the target values
target=['ne','ni','nn','Te','Ti','Tn','Ve','Vi','Vn','E','Pot']

def add_target(target,df):
    target_list=[df3[f'{target}'],df3[f'{target} (heat 0.1)'],df3[f'{target} (heat 0.2)'],df3[f'{target} (1 T)'],df3[f'{target} (4 T)'],df3[f'{target} (recycling 0)'],df3[f'{target} (recycling 1)'],
        df2[f'{target}'],df2[f'{target} (heat 0.1)'],df2[f'{target} (heat 0.2)'],df2[f'{target} (1 T)'],df2[f'{target} (4 T)'],df2[f'{target} (recycling 0)'],df2[f'{target} (recycling 1)'],
        df1[f'{target}'],df1[f'{target} (heat 0.1)'],df1[f'{target} (heat 0.2)'],df1[f'{target} (1 T)'],df1[f'{target} (4 T)'],df1[f'{target} (recycling 0)'],df1[f'{target} (recycling 1)']]
    
    
    df_target=pd.DataFrame(target_list)
    df_target=df_target.reset_index(level=0)
    df_target=pd.melt(df_target, id_vars='index')
    #Add
    df.loc[:,f'{target}']=df_target["value"]
    return df

for target in target:
    df=add_target(target,df)

#%%
intermediate_time = time.time()
print('Čas za branje je: ',intermediate_time - start_time)

#%%
#Save the data
df.to_csv('df_data.csv',index=False)