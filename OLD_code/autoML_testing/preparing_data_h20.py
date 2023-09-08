# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:37:40 2023

@author: urosu
"""
import pandas as pd
data=pd.read_csv('df_data.csv')
data=data.rename(columns={"x_[m]": "x_m"})  #Change this in extracting_data.py
data=data[["angle","heat","field","emission","x_m","Pot"]]

data.to_csv('df_h20.csv',index=False)