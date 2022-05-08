#%%
import os
os.chdir('C:/Users/matkinson/Documents/Math-533')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random as random
#%%
#Read in data and look at first few rows
fire = pd.read_csv('Alberta_Wildfire_Final-1996-2002.csv')
fire.head()

#%%
fire['Growth'].value_counts()
#get summary statistics
fire.select_dtypes(include='float64').describe()
#%%
#calculate log of EX_Size and BH
fire['log_Ex_Size'] = np.log(fire['Ex_Size'])
fire['log_BH_Size'] = np.log(fire['BH_Size'])
#create values for plotting
fire['Growth_Plot'] = fire['Growth'].apply(lambda x: 'BH Size = IA Size' if x == 1 else 'BH Size > IA Size')
fig , ax = plt.subplots(nrows= 2, ncols = 2 ,figsize = (15,10))
ax[0,0].set_title('Violin of Log Extinguish Size over Growth')
v_plot = sns.violinplot(x='Growth_Plot',y='log_Ex_Size' ,data =fire, palette ="viridis" ,ax =ax[0,0])
ax[0,0].set(xlabel='Growth')
ax[0,1].set_title('Violin of Log Initial Attack Size over Growth')
v_plot = sns.violinplot(x='Growth_Plot',y='logIA_Size' ,data =fire, palette ="viridis" ,ax =ax[0,1])
ax[0,1].set(xlabel='Growth')
ax[1,0].set_title('Violin of Log Being Held Size Over Growh')
v_plot = sns.violinplot(x='Growth_Plot',y='log_BH_Size',data =fire, palette ="viridis" ,ax =ax[1,0])
ax[1,0].set(xlabel='Growth')
ax[1,1].set_title('Violin of Log Response Time over Growth')
v_plot = sns.violinplot(x='Growth_Plot',y='logResp_time' ,data =fire, palette ="viridis" ,ax =ax[1,1])
ax[1,1].set(xlabel='Growth')
plt.show()
plt.tight_layout()

#%%
fig , ax = plt.subplots(nrows= 1, ncols = 2 ,figsize = (15,10))
ax[0].set_title('Countplot By Period By Growth')
v_plot = sns.countplot(x = 'Period', hue='Growth_Plot',data =fire, palette ="viridis" ,ax =ax[0])
ax[0].set(xlabel='Period')
ax[1].set_title('Coount by Fuel Type By Growth')
v_plot = sns.countplot(x = 'Fuel_type', hue='Growth_Plot',data =fire, palette ="viridis" ,ax =ax[1])
ax[1].set(xlabel='Fuel Type')
plt.show()
plt.tight_layout()
#%%
fig , ax = plt.subplots(nrows= 1, ncols = 2 ,figsize = (15,10))
ax[0].set_title('Countplot By Detection By growth')
v_plot = sns.countplot(x = 'Detection', hue='Growth_Plot',data =fire, palette ="viridis" ,ax =ax[0])
ax[0].set(xlabel='Detection')
ax[1].set_title('Count by Method By Growth')
v_plot = sns.countplot(x = 'Method', hue='Growth_Plot',data =fire, palette ="viridis" ,ax =ax[1])
ax[1].set(xlabel='Method')
plt.show()
plt.tight_layout()

#%%
fig , ax = plt.subplots(nrows= 1, ncols = 2 ,figsize = (15,10))
sns.scatterplot(x='logIA_Size', y='ISI',hue ='Growth_Plot',data=fire,ax=ax[0], palette ="viridis")
ax[0].set_title('ISI Vs LogIA_Size by Growth')
sns.scatterplot(x='logIA_Size', y='FWI',hue ='Growth_Plot',data=fire,ax=ax[1], palette ="viridis")
ax[1].set_title('ISI Vs FWI by Growth')
plt.show()
plt.tight_layout()

#%%
fig , ax = plt.subplots(nrows= 1, ncols = 2 ,figsize = (15,10))
sns.scatterplot(x='log_BH_Size', y='ISI',hue ='Growth_Plot',data=fire,ax=ax[0], palette ="viridis")
ax[0].set_title('ISI Vs LogBH_Size by Growth')
sns.scatterplot(x='log_BH_Size', y='FWI',hue ='Growth_Plot',data=fire,ax=ax[1], palette ="viridis")
ax[1].set_title('FWI VS LogBH_Size by Growth')
plt.show()
plt.tight_layout()


#%%
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
def r2(x, y):
return stats.pearsonr(x, y)[0] ** 2
growth = fire[fire['Growth']==0].reset_index(drop=True)
growth['Log_BHIA_Gap'] =np.log(growth['BH_Size'] - growth['IA_Size'])
from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0]**2
x,y = growth['Log_BHIA_Gap'] ,growth['FWI']
sns.jointplot(x, y, kind="reg", stat_func=r2 ,height= 8 )
plt.tight_layout()
plt.show()

#%%
#### Create data for models KNN RandomForest SVM
cat_cols = ['Period','Fuel_type', 'Detection','Method']
dummy_code = pd.get_dummies(fire[cat_cols],drop_first = False)
data1 = pd.concat([fire, dummy_code] ,axis=1)
drop_cols = ['NUMBER','number_of_fire','IA_Size','BH_Size','Status','Gap_BHIA','Response_time','Month','Gap_UCIA',
'Ex_Size', 'Gap','sqlogIA_Size','sqlogResp_time','sqFWI','sqISI','logNumber_of_fire','log_Ex_Size','log_BH_Size',
'Growth_Plot' ] + cat_cols
data1.drop(drop_cols,axis= 1 , inplace=True)
nums = data1[['FWI','ISI','logIA_Size','logResp_time']]
#Standardize variables
data1[['FWI','ISI','logIA_Size','logResp_time']] = (nums-nums.mean())/nums.std()
#%%
#Create data for logit model
cat_cols = ['Period','Fuel_type', 'Detection','Method']
dummy_code = pd.get_dummies(fire[cat_cols],drop_first = True)
data2 = pd.concat([fire, dummy_code] ,axis=1)
drop_cols = ['NUMBER','number_of_fire','IA_Size','BH_Size','Status','Gap_BHIA','Response_time','Month','Gap_UCIA',
'Ex_Size', 'Gap','sqlogIA_Size','sqlogResp_time','sqFWI','sqISI','logNumber_of_fire','log_Ex_Size',
'log_BH_Size',
'Growth_Plot' ] + cat_cols
data2.drop(drop_cols,axis= 1 , inplace=True)
nums = data2[['FWI','ISI','logIA_Size','logResp_time']]
data2[['FWI','ISI','logIA_Size','logResp_time']] = (nums-nums.mean())/nums.std()
data2.shape