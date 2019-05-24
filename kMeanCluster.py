#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%%
K_clusters = 3      # number of clusters
raw_file_loc = './datasets/self-efficacy.xlsx'

init_db = pd.read_excel(raw_file_loc, index_col=None, header=None)
header = ['id','time']
for i in range(len(init_db.columns)-2):
    header.append(i)
init_db.columns=header

''' Extract [id],[time],[data] from init_db '''
id_table = init_db.loc[:,'id']
time_tabletime_tabl  = init_db.loc[:, 'time']
data = init_db.drop(columns=['id','time'])
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
''' cluster action_table '''
# create kmeans object
kmeans = KMeans(n_clusters=K_clusters)
# fit kmeans object to data
x = kmeans.fit(action_df)
# print location of clusters learned by kmeans object
centroids = kmeans.cluster_centers_
#%%
# save new clusters for chart
y_km = kmeans.fit_predict(action_df)
print(y_km)
action_df['Kmeans Cluster'] = y_km
action_df
#%%
''' clauster belief_table '''
# create kmeans object
kmeans = KMeans(n_clusters=K_clusters)
# fit kmeans object to data
x = kmeans.fit(belief_df)
# print location of clusters learned by kmeans object
centroids = kmeans.cluster_centers_
#%%
# save new clusters for chart
y_km = kmeans.fit_predict(belief_df)
print(y_km)
belief_df['Kmeans Cluster'] = y_km
belief_df
#%%