#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_squared_log_error



# In[2]:


data = pd.read_csv('C:\\Users\\AASHIMA\\Desktop\\Python\\world.csv',decimal=',')
data.head()
#data.shape


# In[3]:


data.columns
print(data.isnull().sum())


# In[4]:


data.describe(include='all')


# In[5]:


data.groupby("Region")[['GDP ($ per capita)','Literacy (%)','Agriculture']].median()


# In[6]:


for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = data.groupby('Region')[col].median()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())&(data['Region']==region)] = guess_values[region]


# In[7]:


fig, ax=plt.subplots(figsize=(16,6))
top_gdp_countries=data.sort_values("GDP ($ per capita)",ascending=False).head(20)
#top_gdp_countries.head(20)
mean=pd.DataFrame({"Country":["World mean"],"GDP ($ per capita)":[data['GDP ($ per capita)'].mean()]})
#mean.tail()
gdps=pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean])
#gdps
sns.barplot(x='Country',y='GDP ($ per capita)',data=gdps,palette='Set3')
ax.set_xlabel(ax.get_xlabel(),labelpad=15)
ax.set_ylabel(ax.get_ylabel(),labelpad=20)
plt.xticks(rotation=90)
ax.xaxis.label.set_fontsize(30)
ax.yaxis.label.set_fontsize(30)
plt.show()


# In[8]:


plt.figure(figsize=(16,12))
sns.heatmap(data=data.iloc[:,2:].corr(),cmap='coolwarm',annot=True,fmt='.2f')
plt.show()


# In[9]:


fig,axes=plt.subplots(ncols=3,nrows=2,figsize=(20,12))
plt.subplots_adjust(hspace=0.4)

corr_to_gdp=pd.Series()
for col in data.columns.values[2:]:
    if((col!='GDP ($ per capita)')&(col!="Climate")):
        corr_to_gdp[col]=data["GDP ($ per capita)"].corr(data[col])
abs_corr_to_gdp=corr_to_gdp.abs().sort_values(ascending=False)
corr_to_gdp=corr_to_gdp.loc[abs_corr_to_gdp.index]

for i in range(2):
    for j in range(3):
        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=data,
                   ax=axes[i,j], fit_reg=False, marker='.')
        title = 'correlation='+str(corr_to_gdp[i*3+j])
        axes[i,j].set_title(title)
axes[1,2].set_xlim(0,102)
plt.show()


# In[10]:


data.columns.values[2:]


# # Some features, like phones, are related to the average GDP more linearly, while others are not. For example, High birthrate usually means low GDP per capita, but average GDP in low birthrate countries can vary a lot.
# 
# Let’s look at the countries with low birthrate (<14%) and low GDP per capita (<10000 $). They also have hight literacy, like other high average GDP countires. But we hope their other features can help distiguish them from those with low birthrate but high average GDPs, like service are not quite an importent portion in their economy, not a lot phone procession, some have negative net migration, and many of them are from eastern Europe or C.W. of IND. STATES, so the ‘region’ feature may also be useful.

# In[11]:


data.loc[(data["Birthrate"]<14)&(data["GDP ($ per capita)"]<10000)]


# In[12]:


LE=LabelEncoder()
data["Region_label"]=LE.fit_transform(data["Region"])
data["Climate_label"]=LE.fit_transform(data["Climate"])
data.head()


# In[13]:


data.Region.unique()


# In[14]:


data.Region_label.unique()


# In[15]:


train,test=train_test_split(data,test_size=0.3,shuffle=True)
training_features = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'Literacy (%)', 'Phones (per 1000)',
       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Region_label',
       'Climate_label','Service']
target = 'GDP ($ per capita)'
train_x=train[training_features]
train_y=train[target]
test_x=test[training_features]
test_y=test[target]


# In[16]:


#train_x
#train_y
#test_x
test_y


# In[17]:


model = LinearRegression()
model.fit(train_x, train_y)
print(model.score(test_x,test_y))


# In[18]:


train_pred_y=model.predict(train_x)
#train_pred_y
test_pred_y=model.predict(test_x)
#test_pred_y
train_pred_y=pd.Series(train_pred_y.clip(0,train_pred_y.max()),index=train_y.index)
#train_pred_y
test_pred_y = pd.Series(test_pred_y.clip(0, test_pred_y.max()), index=test_y.index)
#test_pred_y


# In[19]:


rmse_train=np.sqrt(mean_squared_error(train_pred_y,train_y))
rmse_train
msle_train=mean_squared_log_error(train_pred_y,train_y)
msle_train
rmse_test=np.sqrt(mean_squared_error(test_pred_y,test_y))
rmse_test
msle_test=mean_squared_log_error(test_pred_y,test_y)
msle_test
print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)


# In[20]:


#mode=RandomForestRegressor(n_estimators=50,max_depth=6,min_weight_fraction_leaf=0.05,max_features=0.8,random_state=42)
#model.fit(train_x,train_y)
#print(model.score(test_x,test_y))


# In[21]:


""""
train_pred_y=model.predict(train_x)
#train_pred_y
test_pred_y = model.predict(test_x)
#test_pred_y
train_pred_y = pd.Series(train_pred_y.clip(0, train_pred_y.max()), index=train_y.index)
test_pred_y= pd.Series(test_pred_y.clip(0, test_pred_y.max()), index=test_y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_y, train_y))
msle_train = mean_squared_log_error(train_pred_y, train_y)
rmse_test = np.sqrt(mean_squared_error(test_pred_y, test_y))
msle_test = mean_squared_log_error(test_pred_y, test_y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)
"""


# In[22]:


model = RandomForestRegressor(n_estimators = 5,
                             max_depth = 6,
                             min_weight_fraction_leaf = 0.05,
                             max_features = 0.8,
                             random_state = 42)
model.fit(train_x, train_y)
train_pred_Y = model.predict(train_x)
test_pred_Y = model.predict(test_x)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_y))
msle_train = mean_squared_log_error(train_pred_Y, train_y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_y))
msle_test = mean_squared_log_error(test_pred_Y, test_y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)


# In[23]:


plt.figure(figsize=(18,12))
train_test_Y=train_y.append(test_y)
#train_test_Y
train_test_pred_Y = train_pred_y.append(test_pred_y)
#train_test_pred_Y
data_shuffled=data.loc[train_test_Y.index]
#data_shuffled
label=data_shuffled["Country"]
colors = {'ASIA (EX. NEAR EAST)         ':'red',
          'EASTERN EUROPE                     ':'orange',
          'NORTHERN AFRICA                    ':'gold',
          'OCEANIA                            ':'green',
          'WESTERN EUROPE                     ':'blue',
          'SUB-SAHARAN AFRICA                 ':'purple',
          'LATIN AMER. & CARIB    ':'olive',
          'C.W. OF IND. STATES ':'cyan',
          'NEAR EAST                          ':'hotpink',
          'NORTHERN AMERICA                   ':'lightseagreen',
          'BALTICS                            ':'rosybrown'}
colors.items
for region, color in colors.items():
    X = train_test_Y.loc[data_shuffled['Region']==region]
    Y = train_test_pred_Y.loc[data_shuffled['Region']==region]
    ax = sns.regplot(x=X, y=Y, marker='.', fit_reg=False, color=color, scatter_kws={'s':200, 'linewidths':0}, label=region) 
plt.legend(loc=4,prop={'size': 12})  

ax.set_xlabel('GDP ($ per capita) ground truth',labelpad=40)
ax.set_ylabel('GDP ($ per capita) predicted',labelpad=40)
ax.xaxis.label.set_fontsize(24)
ax.yaxis.label.set_fontsize(24)
ax.tick_params(labelsize=12)

x = np.linspace(-1000,50000,100) # 100 linearly spaced numbers
y = x
plt.plot(x,y,c='gray')

plt.xlim(-1000,60000)
plt.ylim(-1000,40000)

for i in range(0,train_test_Y.shape[0]):
    if((data_shuffled['Area (sq. mi.)'].iloc[i]>8e5) |
       (data_shuffled['Population'].iloc[i]>1e8) |
       (data_shuffled['GDP ($ per capita)'].iloc[i]>10000)):
        plt.text(train_test_Y.iloc[i]+200, train_test_pred_Y.iloc[i]-200, label.iloc[i], size='small')


# In[24]:


data["Total GDP"]=data['GDP ($ per capita)']*data['Population']
top_gdp_countries=data.sort_values("Total GDP",ascending=False).head(10)
#top_gdp_countries
others=pd.DataFrame({'Country':["Others"],'Total GDP':[data["Total GDP"].sum()-top_gdp_countries["Total GDP"].sum()]})
#others
gdps = pd.concat([top_gdp_countries[['Country','Total GDP']],others],ignore_index=True)
gdps


# In[25]:


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(20,7),gridspec_kw={"width_ratios":[2,1]})
sns.barplot(x="Country",y="Total GDP",data=gdps,ax=axes[0],palette="Set3")
axes[0].set_xlabel("Country",labelpad=30,fontsize=16)
axes[0].set_ylabel("Total GDP",labelpad=30,fontsize=16)

colors = sns.color_palette("Set3", gdps.shape[0]).as_hex()
axes[1].pie(gdps['Total GDP'], labels=gdps['Country'],colors=colors,autopct='%1.1f%%',shadow=True)
axes[1].axis('equal')
plt.show()


# # Let’s compare the above ten countries’ rank in total GDP and GDP per capita.

# In[26]:


Rank1=data[['Country',"Total GDP"]].sort_values("Total GDP", ascending=False).reset_index()
Rank2=data[['Country',"GDP ($ per capita)"]].sort_values("GDP ($ per capita)", ascending=False).reset_index()
Rank1=pd.Series(Rank1.index.values+1,index=Rank1.Country)
Rank2=pd.Series(Rank2.index.values+1,index=Rank2.Country)
Rank_change = (Rank2-Rank1).sort_values(ascending=False)
print('rank of total GDP - rank of GDP per capita:')
Rank_change.loc[top_gdp_countries.Country]


# We see the countries with high total GDPs are quite different from those with high average GDPs.
# 
# China and India jump above a lot when it comes to the total GDP.
# 
# The only country that is with in top 10 (in fact top 2) for both total and average GDPs is the United States. 

# In[27]:


corr_to_gdp=pd.Series()
for col in data.columns.values[2:]:
    if((col!='Total GDP')&(col!='Climate')&(col!='GDP ($ per capita)')):
        corr_to_gdp[col]=data['Total GDP'].corr(data[col])
    abs_corr_to_gdp=corr_to_gdp.abs().sort_values(ascending=False)
    corr_to_gdp=corr_to_gdp.loc[abs_corr_to_gdp.index]
    print(corr_to_gdp)


# In[28]:


plot_data=top_gdp_countries.head(10)[["Country","Agriculture","Industry","Service"]]
plot_data=plot_data.set_index("Country")
#plot_data
ax=plot_data.plot.bar(stacked=True,figsize=(10,6))
ax.legend(bbox_to_anchor=(1, 1))
plt.show()


# In[29]:


plot_data = top_gdp_countries[['Country','Arable (%)', 'Crops (%)', 'Other (%)']]
plot_data = plot_data.set_index('Country')
ax=plot_data.plot.bar(stacked=True,figsize=(10,6))
ax.legend(bbox_to_anchor=(1, 1))
plt.show()


# In[ ]:




