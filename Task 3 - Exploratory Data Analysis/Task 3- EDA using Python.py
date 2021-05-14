#!/usr/bin/env python
# coding: utf-8

# # Grip- The Sparks Foundation
# 
# 
# 
# 
# 
# 
# <font color = green>*Task 3- Exploratory Data Analysis*</font>
# 

# ### To perform Exploratory Data Analysis on Dataset sample Superstore
# 

# 
# 
# #### Import libraries

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import warnings
warnings.filterwarnings("ignore")


# 
# #### Loading Dataset

# In[9]:


df=pd.read_csv("E:\Python\Task 3- EDA\SampleSuperstore.csv")


# In[10]:


df.head()


# #### Display Bottom 5 rows

# In[11]:


df.tail()


# In[12]:


df.shape


# #### Display Summary

# In[13]:


df.describe()


# #### Checking the Null values

# In[14]:


df.isnull().sum()


# 
# #### Information about dataset

# In[15]:


df.info()


# In[16]:


df.columns


# In[17]:


df.duplicated().sum()


# In[18]:


df.nunique()


# In[19]:


df['Postal Code'] = df['Postal Code'].astype('object')


# In[21]:


corr = df.corr()
sns.heatmap(corr, annot = True, cmap='Purples')


# In[22]:


df = df.drop(['Postal Code'], axis = 1)
sns.pairplot(df, hue = 'Ship Mode')


# In[23]:


df['Ship Mode'].value_counts()


# In[24]:


sns.countplot(x = df['Ship Mode'])


# In[25]:


df['Segment'].value_counts()


# In[26]:


sns.pairplot(df, hue= 'Segment')


# In[28]:


sns.countplot(x = df['Segment'])


# In[30]:


df['Category'].value_counts()


# In[31]:


sns.pairplot(df, hue = 'Category')


# In[32]:


df['Sub-Category'].value_counts()


# In[33]:


sns.countplot(x = df['Sub-Category'])


# In[36]:


plt.figure(figsize=(15,12))
df['Sub-Category'].value_counts().plot.pie()
plt.show()


# In[37]:


df['State'].value_counts()


# In[39]:


fig = plt.figure(figsize =(16,13))
fig.set(facecolor = 'yellow')
sns.countplot(x = df['State'], palette = 'icefire_r', order = df['State'].value_counts().index)
plt.xticks(rotation = 90)


# In[45]:


sns.set(rc = {'axes.facecolor' : 'pink', 'figure.facecolor' : 'skyblue'})
df.hist(figsize = (10,10))


# In[48]:


fig = plt.figure(figsize = (10,10))
fig.set(facecolor = 'cyan')
df['Region'].value_counts().plot.pie()


# In[50]:


fig,ab = plt.subplots(figsize = (18,10))
ab.set(facecolor = 'none')
ab.scatter(df['Sales'],df['Profit'])
ab.set_xlabel('Sales')
ab.set_ylabel('Profit')
ab.spines['bottom'].set_color('purple')
ab.spines['top'].set_color('purple')
ab.spines['right'].set_color('purple')
ab.spines['left'].set_color('purple')


# In[52]:


sns.set(rc = {'figure.facecolor' : 'pink'})
sns.lineplot(x = df['Discount'], y = df['Profit'], label = "Profit")


# In[53]:


df.groupby('Segment')[['Profit', 'Sales']].sum().plot.bar()


# In[55]:


plt.figure(figsize = (15,12))
plt.title('Segment wise sales in each region')
sns.barplot(x = df['Region'], y = df['Sales'], hue = df['Segment'])


# In[56]:


df.groupby('Region')[['Profit', 'Sales']].sum().plot.bar()


# In[57]:


ps = df.groupby('State')[['Sales', 'Profit']].sum().sort_values(by = 'Sales', ascending = False)
ps.plot.bar(figsize = (15,12))
plt.title('Profit/Loss & Sales across States')
plt.xlabel('States')
plt.ylabel('Profit/Loss & Sales')


# In[59]:


ts = df['State'].value_counts().nlargest(10)
ts


# In[60]:


df.groupby('Category')[['Profit', 'Sales']].sum().plot.bar(figsize = (10,8))
plt.xlabel('Category')
plt.ylabel('Profit/Loss & Sales')


# In[61]:


df.groupby('Sub-Category')[['Profit', 'Sales']].sum().plot.bar(figsize = (15,12))
plt.xlabel('Sub-Category')
plt.ylabel('Profit/Loss & Sales')


# 
# 
# # Observations:-
#   1. Maximum sales are from Binders , Papers,Furnishing,Phones, Storage,Art,Accessories amd Minimum from Copies, machines and suppliers.
#   2. Higher Number of buyers are from Calofornia and New York.
#   3. Most Customers tends to buy qauntity of 2 and 3.
#   4. Discount give maximum is of 0% to 20%.
#   5. There is **No Correlation** between Profit and Discount.
#   6. Profit and Sales are maximum in customer segments and minimum in home in segments.
#   7. Segments wise sales are almost same in every region.
#   8. Profits and sales are maximum in west region and minimum in the south region.
#   9. High profits are shown in california and new york.
#   10. Loss is for Texas, Ohio and Pennsylvania.
