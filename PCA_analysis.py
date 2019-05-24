#%%
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%
raw_file_loc = './datasets/self-efficacy.xlsx'

init_db = pd.read_excel(raw_file_loc, index_col=None, header=None)
init_db
#%%
''' extract column header '''
header = ['id','time']
for i in range(len(init_db.columns)-2):
    header.append(i)
init_db.columns=header
init_db.columns
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

''' Caclulate Sample SD '''
standDev = []
for j in range(numCol):
    varSum = 0
    for i in range(numRow):
        varSum += (data.iloc[i,j]-mean[j])**2
    standDev.append(math.sqrt(varSum/(numRow-1)))

''' Append [mean] and [sd] to [data] table '''
data.loc['avg'] = mean
data.loc['sd'] = standDev
data
#%%
''' Seperate [belief], [action] table '''
cols = data.columns
belief_df = pd.DataFrame()
action_df = pd.DataFrame()

for col_index in range(len(cols)):
    if col_index%2 == 0:
        belief_df[col_index] = data[cols[col_index]]
    else:
        action_df[col_index] = data[cols[col_index]]
belief_df
#%%
action_df
#%%
''' PCA Analysis '''
x = []
y = []
print('%20s|%15s'%("# of componets", 'total variance'))
for n in reversed(range(1, 22)):
    scaled_action_df = StandardScaler().fit_transform(action_df)
    # print(scaled_action_df)
    numcol = len(belief_df.columns)

    pca = PCA(n_components=n)
    pc = pca.fit_transform(scaled_action_df)
    pcDF = pd.DataFrame(pc)
    pcDF
    
    x.append(n)
    ratios = pca.explained_variance_ratio_
    sum = 0
    for ratio in ratios:
        sum += ratio
    print('%20d|%15f'%(n, sum*100))
    y.append((sum)*100)
#%%
plt.xlabel("number of components")
plt.ylabel("total Variance")

plt.plot(x,y)
#%%