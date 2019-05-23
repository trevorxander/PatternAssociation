#%%
import pandas as pd
import numpy as np
import math
#%%
raw_file_loc = './datasets/self-efficacy.xlsx'

#%%
init_db = pd.read_excel(raw_file_loc, index_col=None, header=None)
header = ['id','time']
for i in range(len(init_db.columns)-2):
    header.append(i)
#%%
init_db.columns=header
init_db
#%%
''' Extract [id],[time],[data] from init_db '''

id_table = init_db.loc[:,'id']
id_table
#%%
time_table = init_db.loc[:, 'time']
time_table
#%%
data = init_db.drop(columns=['id','time'])
data
#%%
numRow = len(data.index)
numCol = len(data.columns)
''' Calculate Mean '''
mean = []
for j in range(numCol):
    sum = 0
    for i in range(numRow):
        sum += data.iloc[i,j]
    mean.append(sum/numRow)
mean
#%%
''' Caclulate Sample SD '''
standDev = []
for j in range(numCol):
    varSum = 0
    for i in range(numRow):
        varSum += (data.iloc[i,j]-mean[j])**2
    standDev.append(math.sqrt(varSum/(numRow-1)))
standDev
#%%
# data.loc[len(data)] = mean
data.rename(index={numRow:"avg"},inplace=True)
data
#%%
# data.loc[len(data)] = standDev
data.rename(index={numRow+1:"sd"},inplace=True)
data

#%%


