# Building Height Estimation
# Accumulate input data from zonal statistics
# Thepchai Srinoi
# Department of Survey Engineering

# Divided and Conquer .... Too long if you do median stat with all data

import pandas as pd

df1 = pd.read_csv('traindata_0305.csv')
df1=df1.dropna()
df1.to_csv('input1.csv')

print(df1)

df2 = pd.read_csv('traindata_0510.csv')
df2=df2.dropna()
df2.to_csv('input2.csv')
print(df2)

df3 = pd.read_csv('traindata_1015.csv')
df3=df3.dropna()
df3.to_csv('input3.csv')
print(df3)

df4 = pd.read_csv('traindata_1520.csv')
df4=df4.dropna()
df4.to_csv('input4.csv')
print(df4)

df5 = pd.read_csv('traindata_20toinf.csv')
df5 =df5.dropna()
df5.to_csv('input5.csv')
print(df5)

print('-------------------------------')
df = pd.concat([df1, df2, df3, df4, df5] )
df = df.drop(columns='Unnamed: 0')
df = df.dropna()
df = df.reset_index(drop=True)
print(df)

df.to_csv('myinput.csv')

import pdb; pdb.set_trace()
