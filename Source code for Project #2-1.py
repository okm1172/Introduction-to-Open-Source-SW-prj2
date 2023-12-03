# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:34:52 2023

@author: User
"""
import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

#1
value=['H','avg','HR','OBP']
aa=(data_df['year'] >= 2015) & (data_df['year'] <= 2018)
real_data=data_df[aa]
for value_name in value:
    print(value_name + " : ")
    print(real_data.sort_values(by=value_name).iloc[:11]['batter_name'])

#2
abc=data_df[data_df['year']==2018]
print(abc)
data_=abc.sort_values(by='war',ascending=False)
print(data_)
position=['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
for position_name in position:
    bb=(data_['cp']==position_name)
    print(position_name + " : ")
    print(data_[bb].sort_values(by='war',ascending=False)['batter_name'].head(5))
    
#3
drop_value=['P','H','HR','RBI','SB','war','avg','OBP','SLG']
data_salary = data_df['salary']
data_not_salary = data_df.drop('salary',axis=1)
for value in data_not_salary.columns:
    if value not in drop_value:
        data_not_salary = data_not_salary.drop(value,axis=1)
print("The highest correlation with salary : ")
print(data_not_salary.corrwith(data_salary).sort_values(ascending=False).index[0])