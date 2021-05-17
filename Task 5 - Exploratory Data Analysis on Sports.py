#!/usr/bin/env python
# coding: utf-8

# # Grip - The Sparks Foundation
# 
# 
# 
# 
# 
# 
# 
# <font color = red>_Task 5 - Exploratory Data Analysis on Dataset Indian Premier League_</font>

# 
# 
# ### Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


matches = pd.read_csv(r'C:\Users\dell\Downloads\Indian Premier League\matches.csv')


# In[4]:


matches.head()


# In[5]:


matches.tail()


# 
# 
# ### Information of Data

# In[6]:


matches.info()


# 
# 
# ### Summary of the Data

# In[7]:


matches.shape


# In[8]:


matches.describe()


# 
# 
# ### Loading Second Dataset

# In[2]:


deliveries = pd.read_csv(r'C:\Users\dell\Downloads\Indian Premier League\deliveries.csv') 


# In[3]:


deliveries.head()


# In[4]:


deliveries.tail()


# 
# 
# ### Information of data

# In[5]:


deliveries.info()


# In[6]:


deliveries.shape


# 
# 
# 
# ### Summary of the Data
# 

# In[7]:


deliveries.describe()


# 
# 
# 
# 
# 
# ### Merging the two datasets

# In[16]:


merge = pd.merge(deliveries,matches, left_on='match_id', right_on ='id')


# In[17]:


merge.head(2)


# In[18]:


merge.info()


# In[19]:


merge.describe()


# In[20]:


matches.id.is_unique


# In[21]:


matches.describe(include = 'all')


# 
# 
# 
# ### Data Preprocessing

# In[22]:


matches.head()


# ### From Preprocessing we found that:
# 
# 
# 
# #### city has missing values
# 
# #### team1 and team2 columns have 14 distinct values but winner has 15 distinct values
# 
# #### umpire1 and umpire2 have 1 missing value each
# 
# #### city has 33 distinct values while venue has 35 distinct values
# 
# #### Filling in the missing values of city column
# 
# #### let's find the venues corresponding to which the values of city are empty
# 
# 

# In[23]:





matches[matches.city.isnull()][['city','venue']]


# In[24]:


matches.city = matches.city.fillna('Dubai')


# In[25]:


matches[(matches.umpire1.isnull()) | (matches.umpire2.isnull())]


# 
# 
# 
# #### Umpire3 column has close to 92% missing values. hence dropping that column

# In[26]:


matches = matches.drop('umpire3', axis = 1)


# 
# 
# 
# #### Let's find out venues grouped by cities to see which cities have multiple venues

# In[27]:


city_venue = matches.groupby(['city','venue']).count()['season']
city_venue_df = pd.DataFrame(city_venue)
city_venue_df


# 
# ### Observations
# #### So we need to keep one of them
# 
# #### Bengaluru and Bangalore both are in the data when they are same.
# 
# #### Chandigarh and Mohali are same and there is just one stadium Punjab Cricket Association IS Bindra Stadium, Mohali whose value has not been entered correctly.
# 
# ####  We need to have either Chandigarh or Mohali as well as correct name of the stadium there
# 
# 
# #### Mumbai has 3 stadiums/venues used for IPL
# 
# #### Pune has 2 venues for IPL
# 
# ### Exploratory Data Analysis:
# 
# #### Number of matches played in each season

# In[30]:


plt.figure(figsize=(15,5))
sns.countplot('season', data = matches)
plt.title("Number of matches played each season",fontsize=18,fontweight="bold")
plt.ylabel("Count", size = 25)
plt.xlabel("Season", size = 25)
plt.xticks(size = 20)
plt.yticks(size = 20)


# 
# 
# ### Number of players played in season

# In[31]:


matches.groupby('season')['team1'].nunique().plot(kind = 'bar', figsize=(15,5))
plt.title("Number of teams participated each season ",fontsize=18,fontweight="bold")
plt.ylabel("Count of teams", size = 25)
plt.xlabel("Season", size = 25)
plt.xticks(size = 15)
plt.yticks(size = 15)


# 
# 
#  #### 10 teams played in 2011 and 9 teams each in 2012 and 2013
# 
# #### This explains why 2011-2013 have seen more matches being played than other seasons
# 
# #### Venue which has hosted most number of IPL matches

# In[32]:


matches.venue.value_counts().sort_values(ascending = True).tail(10).plot(kind = 'barh',figsize=(12,8), fontsize=15, color='green')
plt.title("Venue which has hosted most number of IPL matches",fontsize=18,fontweight="bold")
plt.ylabel("Venue", size = 25)
plt.xlabel("Frequency", size = 25)


# #### Teams having maximum wins:

# In[4]:


winning_teams = matches[['season','winner']]


# 
# 
# #### Uploading dictionaries  to get winners

# In[5]:


winners_team = {}
for i in sorted(winning_teams.season.unique()):
    winners_team[i] = winning_teams[winning_teams.season == i]['winner'].tail(1).values[0]
    
winners_of_IPL = pd.Series(winners_team)
winners_of_IPL = pd.DataFrame(winners_of_IPL, columns=['team'])


# In[6]:


winners_of_IPL['team'].value_counts().plot(kind = 'barh', figsize = (15,5), color = 'darkblue')
plt.title("Winners of IPL across 11 seasons",fontsize=18,fontweight="bold")
plt.ylabel("Teams", size = 25)
plt.xlabel("Frequency", size = 25)
plt.xticks(size = 15)
plt.yticks(size = 15)


# 
# 
# 
# 
# ### MI and CSK have both won 3 times each followed by KKR who has won 2 times.
# 
# ### Hyderabad team has also won 2 matches under 2 franchise name - Deccan Chargers and Sunrisers Hyderabad.
# 
# ##### Does teams choosed to bat or field first, after winning toss?

# In[10]:


matches['toss_decision'].value_counts().plot(kind='pie', fontsize=14, autopct='%3.1f%%', 
                                               figsize=(10,7), shadow=True, startangle=135, legend=True, cmap='Reds')

plt.ylabel('Toss Decision')
plt.title('Decision taken by captains after winning tosses')


# 
# 
# 
# #### How does toss decision affects?

# In[11]:


matches['toss_win_game_win'] = np.where((matches.toss_winner == matches.winner),'Yes','No')
plt.figure(figsize = (15,5))
sns.countplot('toss_win_game_win', data=matches, hue = 'toss_decision')
plt.title("How Toss Decision affects match result", fontsize=18,fontweight="bold")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Winning Toss and winning match", fontsize = 25)
plt.ylabel("Frequency", fontsize = 25)


# 
# 
# 
# 
# #### Teams winning tosses and electng to field first have won most number of times.
# 
# #### Individual teams decision to choose bat or field after winning toss.

# In[12]:


plt.figure(figsize = (25,10))
sns.countplot('toss_winner', data = matches, hue = 'toss_decision')
plt.title("Teams decision to bat first or second after winning toss", size = 30, fontweight = 'bold')
plt.xticks(size = 10)
plt.yticks(size = 15)
plt.xlabel("Toss Winner", size = 35)
plt.ylabel("Count", size = 35)


# 
# 
# 
# #### Most teams field first after winning toss except for Chennai Super Kings who has mostly opted to bat first. Deccan Chargers and Pune Warriors also show the same trend.
# 
# #### Which player's performance has mostly led team's win?

# In[15]:


MoM= matches['player_of_match'].value_counts()
MoM.head(10).plot(kind = 'bar',figsize=(12,8), fontsize=15, color='Red')
plt.title("Top 10 players with most MoM awards",fontsize=18,fontweight="bold")
plt.ylabel("Frequency", size = 25)
plt.xlabel("Players", size = 25)


# 
# 
# 
# 
# #### Chris Gayle has so far won the most number of MoM awards followed by AB de Villiers.
# 
# #### Also, all top 10 are batsmen which kind of hints that in IPL batsmen have mostly dictated the matches.
# 
# #### How winning matches by fielding first varies across venues?

# In[14]:


new_matches = matches[matches['result'] == 'normal']   #taking all those matches where result is normal and creating a new dataframe
new_matches['win_batting_first'] = np.where((new_matches.win_by_runs > 0), 'Yes', 'No')
new_matches.groupby('venue')['win_batting_first'].value_counts().unstack().plot(kind = 'barh', stacked = True,
                                                                               figsize=(15,15))
plt.title("How winning matches by fielding first varies across venues?", fontsize=18,fontweight="bold")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Frequency", fontsize = 25)
plt.ylabel("Venue", fontsize = 25)


# In[ ]:




