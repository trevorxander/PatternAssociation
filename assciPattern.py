#%%
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis as PCA
#%%
raw_file_loc = './datasets/self-efficacy.xlsx'

#%%
init_db = pd.read_excel(raw_file_loc, index_col=None, header=None)
header = ['id','time']
for i in range(len(init_db.columns)-2):
    header.append(i)
#%%
init_db.columns=header
#%%
''' Extract [id],[time],[data] from init_db '''

id_table = init_db.loc[:,'id']
#%%
time_table = init_db.loc[:, 'time']
#%%
data = init_db.drop(columns=['id','time'])
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
#%%
''' Caclulate Sample SD '''
standDev = []
for j in range(numCol):
    varSum = 0
    for i in range(numRow):
        varSum += (data.iloc[i,j]-mean[j])**2
    standDev.append(math.sqrt(varSum/(numRow-1)))
#%%
# data.loc[len(data)] = mean
data.rename(index={numRow:"avg"},inplace=True)
#%%
# data.loc[len(data)] = standDev
data.rename(index={numRow+1:"sd"},inplace=True)

#%%
cols = data.columns
belief_df = pd.DataFrame()
action_df = pd.DataFrame()
#%%
for col_index in range(len(cols)):
    if col_index%2 == 0:
        belief_df[col_index] = data[cols[col_index]]
    else:
        action_df[col_index] = data[cols[col_index]]


#%%

x = []
y = []
print('%20s|%15s'%("number of components", 'total variance'))
for n in reversed(range(1, 22)):
    scaled_action_df = StandardScaler().fit_transform(action_df)
    # print(scaled_action_df)
    numcol = len(belief_df.columns)

    pca = PCA(n_components=n)
    pc = pca.fit_transform(scaled_action_df)
    pcDF = pd.DataFrame(pc)
    pcDF
    
    x.append(n)
    # ratios = pca.explained_variance_ratio_
    sum = 0
    for ratio in ratios:
        sum += ratio
    print('%20d|%15f'%(n, sum*100))
    y.append((sum)*100)
    # print("=================")
#%%
plt.xlabel("number of components")
plt.ylabel("total Variance")

# plt.plot(x,y)
# pca.explained_variance_ratio_
#%%