import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override() 

from keras.models import load_model
import streamlit as st

start ='2017-12-01'
end = '2022-11-30'

st.title('Stock Trend Prediction')

st.sidebar.write("""
# Popular Stocks
### Apple Inc. (AAPL)
### Tesla, Inc. (TSLA)
### Microsoft Corporation (MSFT)
### Tata Motors Limited (TATAMOTORS.NS)
### Tata Steel Limited (TATASTEEL.NS)
### Alphabet Inc. (GOOG)
### Infosys Limited (INFY)
### Tata Consultancy Services Limited (TCS.NS)
""")

user_input=st.text_input('Enter Stock Ticker','TSLA')


df = pdr.get_data_yahoo(user_input, start="2017-12-01", end="2022-11-30")

#Describe Data
st.subheader('Data From December 2017 - November 2022')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(df.Open)
plt.plot(df.High)
plt.plot(df.Low)
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
st.pyplot(fig)

st.subheader('Volumn vs Time Chart')
fig=plt.figure(figsize=(12,5))
plt.plot(df.Volume)
plt.ylabel('Volume')
plt.xlabel('Days')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
fig=plt.figure(figsize=(12,5))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

# Create a dataframe with only the Close Stock Price Column
data_target = df.filter(['Close'])

# Convert the dataframe to a numpy array to train the LSTM model
target = data_target.values

# Splitting the dataset into training and test

# Target Variable: Close stock price value

training_data_len = math.ceil(len(target)* 0.75) # training set has 75% of the data
training_data_len

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_scaled_data = sc.fit_transform(target)


#load my model
model=load_model('my_model.keras')

#Testing Part
# Getting the predicted stock price
test_data = training_scaled_data[training_data_len - 180: , : ]

#Create the x_test and y_test data sets
X_test = []
y_test =  target[training_data_len : , : ]
for i in range(180,len(test_data)):
    X_test.append(test_data[i-180:i,0])

# Convert x_test to a numpy array
X_test = np.array(X_test)

#Reshape the data into the shape accepted by the LSTM
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print('Number of rows and columns: ', X_test.shape)

# Making predictions using the test dataset
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Final Graph

st.subheader('Prediction vs Original')
train = data_target[:training_data_len]
valid = data_target[training_data_len:]
valid['Predictions'] = predicted_stock_price
fig2=plt.figure(figsize=(10,5))
plt.title('Model')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig2)

#Describe Final Data
st.subheader('Result Data From December 2017 - November 2022')
st.write(valid)