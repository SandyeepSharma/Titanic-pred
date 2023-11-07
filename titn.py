# -*- coding: utf-8 -*-
"""
Created on Thu May 28 05:06:33 2020

@author: sandy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv(r"C:/Users/sandy/Dropbox/data/titanic/train.csv")
test = pd.read_csv(r"C:/Users/sandy/Dropbox/data/titanic/test.csv")

#Preprocessing
train.describe()

train.isnull().sum()
name = train.iloc[:, 3].str.split(",", n = 1, expand = True)
first_name= name[1].str.split(".", n = 1, expand = True)
train["salutation"] = first_name[0]

train['salutation'].value_counts().plot('bar')


b = train['salutation'].value_counts()
b = b[b < 8].index
print (b)

train['salutation'] = train['salutation'].replace({x:"others" for x in b})

train["Age"].plot("hist", bins = 20)
train['family'] = train['SibSp'] +train['Parch']

df_train = train.iloc[: , [1,2,4,5,9,11,12,13]]
df_train.isnull().sum()


from fancyimpute import MICE as MICE
df_complete=MICE().complete(df_train)