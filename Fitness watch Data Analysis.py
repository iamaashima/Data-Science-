#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"
import plotly.express as px

data = pd.read_csv("C:\\Users\\AASHIMA\\Downloads\\python\\Apple-Fitness-Data.csv")


# In[2]:


data.head()


# In[3]:


print(data.isnull().sum())


# In[4]:


fig1=px.line(data,x="Time",y="Step Count",title="Step Count Over Time")
fig1.update_xaxes(rangeslider_visible=True)
fig1.show()


# # Now, let’s have a look at the distance covered over time:
# 
# 

# In[5]:


fig2=px.line(data,x="Time",y="Distance",title="Distance Over Time")
fig2.update_xaxes(rangeslider_visible=True)
fig2.show()


# # Now, let’s have a look at my energy burned over time:

# In[6]:


fig3=px.line(data,x="Time",y="Energy Burned",title="Energy Bunned Over Time")
fig3.update_xaxes(rangeslider_visible=True)
fig3.show()


# # Now, let’s have a look at my walking speed over time:

# In[7]:


fig4=px.line(data,x="Time",y="Walking Speed",title="Step Count Over Time")
fig4.update_xaxes(rangeslider_visible=True)
fig4.show()


# # Now, let’s calculate and look at the average step counts per day:

# In[8]:


average_step_count=data.groupby("Date")["Step Count"].mean().reset_index()
average_step_count


# In[9]:


fig5=px.bar(average_step_count,x="Date",y="Step Count",title="Average Step Count per Day")
fig5.update_xaxes(rangeslider_visible=True)
fig5.show()


# # Now, let’s have a look at my walking efficiency over time:

# In[10]:


data["Walking Efficiency"]=data["Distance"]/data["Step Count"]
fig1=px.line(data,x="Time",y="Walking Efficiency",title="Walking Efficiency Over Time")
fig1.update_xaxes(rangeslider_visible=True)
fig1.show()


# # Now, let’s have a look at the step count and walking speed variations by time intervals:

# In[11]:


# Create Time Intervals
time_intervals = pd.cut(pd.to_datetime(data["Time"]).dt.hour,
                        bins=[0, 12, 18, 24],
                        labels=["Morning", "Afternoon", "Evening"], 
                        )
data["Time Interval"]=time_intervals
fig7=px.scatter(data,x="Step Count",y="Walking Speed",color=time_intervals,title="Step Count and Walking Speed Variations by Time Interval",
                  trendline='ols')
fig7.show()


# # Now, let’s compare the daily average of all the health and fitness metrics:
# 
# 

# In[12]:


daily_avg_metrics=data.groupby("Date").mean().reset_index()

daily_avg_metrics_melted=daily_avg_metrics.melt(id_vars=["Date"],value_vars=["Step Count", "Distance", 
                                                              "Energy Burned", "Flights Climbed", 
                                                              "Walking Double Support Percentage", 
                                                              "Walking Speed"])
daily_avg_metrics_melted


# In[13]:


fig=px.treemap(daily_avg_metrics_melted,path=["variable"],values="value",color="variable",hover_data=
              ["value"],title="Daily Averages for Different Metrics")
fig.show()


# In[14]:


metrics_to_visualize = ["Distance", "Energy Burned", "Flights Climbed", 
                        "Walking Double Support Percentage", "Walking Speed"]

# Reshape data for treemap
daily_avg_metrics_melted = daily_avg_metrics.melt(id_vars=["Date"], value_vars=metrics_to_visualize)

fig = px.treemap(daily_avg_metrics_melted,
                 path=["variable"],
                 values="value",
                 color="variable",
                 hover_data=["value"],
                 title="Daily Averages for Different Metrics (Excluding Step Count)")
fig.show()


# In[15]:


daily_avg_metrics_melted


# # So this is how to perform Fitness Data Analysis using Python. Fitness Watch Data Analysis is a crucial tool for businesses in the health and wellness domain. By analyzing user data from fitness wearables, companies can understand user behaviour, offer personalized solutions, and contribute to improving users’ overall health and well-being.

# In[ ]:




