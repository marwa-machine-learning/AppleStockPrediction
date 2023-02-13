#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import plotly.graph_objects as go 
data = pd.read_csv("AAPL.csv") 
print(data.head())


# In[2]:


figure = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"])]) 
figure.update_layout(title = "Apple Stock Price Analysis", xaxis_rangeslider_visible=False) 
figure.show()


# In[3]:


print(data.corr())


# In[ ]:


from autots import AutoTS 
model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple') 
model = model.fit(data, date_col='Date', value_col='Close', id_col=None) 
prediction = model.predict() 
forecast = prediction.forecast 
print(forecast)


# In[ ]:




