#!/usr/bin/env python
# coding: utf-8

# #  Intern - Data Science

# ## Task 1 : Stock Prediction Using LSTM: 

# ----By Devalla Ganesh Babu

# Stock market prediction and forecasting are crucial tasks in the field of finance. Predicting the future trends and prices of stocks can assist investors, traders, and financial institutions in making informed decisions. In this project, we aim to predict and forecast the stock prices of the company "AMAZON(Amazon.csv)" using a Stacked Long Short-Term Memory (LSTM) model.

# In[2]:


# Import main libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for photing and viewing data
import matplotlib.pyplot as plt# plotting library


# In[3]:


df=pd.read_csv('F:/Bharat Intern/Amazon.csv')


# In[4]:


df.head()


# In[5]:


df.set_index('Date',inplace = True)# Set the date to be the index


# In[6]:


# resorting the data
df.index =  pd.to_datetime(df.index,format='%Y-%m-%d')


# In[7]:


df.head()


# # Now Plots

# In[8]:


df[['Open','Close','High','Low']].plot(figsize = (20,12))
plt.title('Amazon Stock at all time')


# In[9]:


df[['Open','Close']].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock price action')
plt.xlabel('Date')
plt.ylabel('Stock action')


# In[10]:


df[['Open','High']].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock price action')
plt.xlabel('Date')
plt.ylabel('Stock action')


# In[11]:


df[['Low','Close']].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock price action')
plt.xlabel('Date')
plt.ylabel('Stock action')


# In[12]:


df['Volume'].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock price action')
plt.xlabel('Date')
plt.ylabel('Stock action')


# In[13]:


df['Adj Close'].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock price action')
plt.xlabel('Date')
plt.ylabel('Stock action')


# # From the previous analysis and visualization, it can take the data from 2015 as the previous years doesn't important, not have a stock price variance

# In[14]:


Ama = df['2010':'2022']


Ama['Open'].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock Price Action form 2012 to 2021')


# In[15]:


Ama[['Open','High']].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock Price Action form 2012 to 2021')


# In[16]:


Ama['Adj Close'].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock Price Action form 2010 to 2022')


# In[17]:


Ama['Volume'].plot(figsize = (20,10), alpha = 1)
plt.title('Amazon Stock Price Action form 2010 to 2022')


# In[18]:


Ama.describe()


# # Augmented Dickey Fuller Test (ADF)

# ADF test is used to determine the presence of unit root in the series, and hence helps in understand if the series is stationary or not

# In[19]:


from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[20]:


print(adf_test(df['High']))


# In[21]:


print(adf_test(df['High'].resample('MS').mean()))


# In[22]:


Ama_diff = Ama['Open'].resample('MS').mean() - Ama['Open'].resample('MS').mean().shift(1)
Ama_open_diff = Ama_diff.dropna()
Ama_open_diff.plot()


print(adf_test(Ama_open_diff))


# # Kwiatkowski-Phillips-Schmidt-Shin Test (KPSS)

# another test for checking the stationarity of a time series

# In[23]:


from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


# In[24]:


kpss_test(Ama['High'])


# In[25]:


Ama["High_diff"] = Ama["High"] - Ama["High"].shift(1)
Ama["High_diff"].dropna().plot(figsize=(12, 8))


# In[26]:


kpss_test(Ama['High_diff'].dropna())


# In[27]:


kpss_test(Ama['High_diff'].resample('MS').mean().dropna())


# In[28]:


kpss_test(Ama['High_diff'].resample('MS').std().dropna())


# In[29]:


adf_test(Ama['High_diff'].dropna())


# # Data Preprocessing

# In[30]:


train_Ama = Ama['High'].iloc[:-4]

# Take ramdom  6 variables 

X_train=[]
y_train=[]

for i in range(2, len(train_Ama)):
    X_train.append(train_Ama[i-2:i])
    y_train.append(train_Ama[i])


# In[31]:


import math
train_len = math.ceil(len(train_Ama)*0.8)
train_len


# In[32]:


# For Model and apply RNN + LSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, TimeDistributed 


# In[33]:


X_train, y_train= np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[34]:


model=Sequential()
model.add(LSTM(50,activation='relu', input_shape=(X_train.shape[1],1)))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=100, verbose=2)


# In[36]:


losse = pd.DataFrame(model.history.history)
losse[['loss']].plot()


# In[37]:


test_data = train_Ama[train_len-2:]
X_val=[]
Y_val=[] 

for i in range(2, len(test_data)):
    X_val.append(test_data[i-2:i])
    Y_val.append(test_data[i])


# In[38]:


X_val, Y_val = np.array(X_val), np.array(Y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1],1))
prediction = model.predict(X_val)


# In[39]:


from sklearn.metrics import mean_squared_error
# Know the model error accuracy | the model accuracy 
lstm_train_pred = model.predict(X_train)
lstm_valid_pred = model.predict(X_val)
print('Train rmse:', np.sqrt(mean_squared_error(y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_val, lstm_valid_pred)))


# In[40]:


valid = pd.DataFrame(train_Ama[train_len:])
valid['Predictions']=lstm_valid_pred 
plt.figure(figsize=(16,8))
plt.plot(valid[['High','Predictions']])
plt.legend(['Validation','Predictions'])
plt.show()


# In[41]:


# data frame to see the percentage of error between real and predicted

variance = []
for i in range(len(valid)):
  
  variance.append(valid['High'][i]-valid['Predictions'][i])
variance = pd.DataFrame(variance)
variance.describe()


# In[42]:


train = train_Ama[:train_len]
valid = pd.DataFrame(train_Ama[train_len:])
valid['Predictions']=lstm_valid_pred

plt.figure(figsize=(16,8))
plt.title('Model LSTM')
plt.xlabel('Date')
plt.ylabel('Amazon Price USD')
plt.plot(train)
plt.plot(valid[['High','Predictions']])
plt.legend(['Train','Val','Predictions'])
plt.show()


# # Results

# The results of the Stock Market Prediction and Forecasting project depend on the specific implementation and training of the Stacked LSTM model. By training the model on historical stock price data, we can obtain predictions and forecasts for future stock prices.
# 
# The performance of the model can be analyzed using evaluation metrics such as mean absolute error (MAE) and root mean squared error (RMSE). Lower values of these metrics indicate better predictive accuracy. Additionally, visualizations, such as line plots, can be used to compare the predicted and actual stock prices, providing a visual assessment of the model's performance.
