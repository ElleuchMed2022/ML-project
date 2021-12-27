import streamlit as st 
import yfinance as yf 
from datetime import date
import pandas as pd 
from PIL import Image
# Importing the libraries 
import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , LSTM
import matplotlib.pyplot as plt 
#plt.style.use('fivethirtyeight')


from keras.models import load_model

import streamlit as st 
import pandas as pd
st.title("Stock Market Web Application ")
st.write("""
Visually show data on a stock! 

""")

image = Image.open("stock.jpg")
st.image(image, use_column_width = True )

#Create a sider Header
st.sidebar.header('User Input')
company=st.sidebar.text_input("company name" ,"AAPL")
start_date=st.sidebar.text_input("Start Date","2012-01-01")
end_date=st.sidebar.text_input("End Date",'2019-12-17')


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ("TSLA","GOOG","AMZN")
selected_stocks = st.selectbox("select dataset for prediction",stocks)
#n_years = st.slider("Years of prediction:",1,4)
#period = n_years*365
@st.cache
def load_data(ticker):
    if ticker == 'GOOG':
        data = pd.read_csv("goog.us.csv")
    elif ticker == 'TSLA':
        data = pd.read_csv("TSLA.csv")
    else :
        data = pd.read_csv("AMZN.csv")
    #data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
   
    return data
    
def get_comapny_name(symbole):
    if symbole == "GOOG" :
        return "Google"
    elif symbole == "TSLA" :
        return 'Tesla'
    elif symbole== 'AMZN':
        return 'Amazon'
    else :
        "None"


data_load_state = st.text("Load data...")
df = load_data(selected_stocks)
df= df.set_index('Date')
datemin=df.index.min()
datemax =df.index.max()
st.write("""
Visually show data on a stock! date range from  """+str(datemin)+""" to """+str(datemax))
data_load_state.text("Loding data..done!")

name = get_comapny_name(selected_stocks)
st.header(name)

st.subheader('Raw data')
st.write(df.tail())



#describe data 
st.subheader("Data Describe")
st.write(df.describe())


#visualization 


#fig = plt.figure(figsize=(12,6))
#plt.plot(df.Close)
#st.pyplot(fig)"""
genre = st.radio("choose your chart",('Open', 'Close', 'Volume'))
st.subheader(genre+" Price vs Time chart")
st.line_chart(df[genre])

st.subheader("Closing Price vs Time chart with 100MA ")
ma100=df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,"r")
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA & 200MA ")
ma200=df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200,"g",label="200MA")
plt.plot(ma100,"r",label="100Ma")
plt.plot(df.Close)
plt.legend()

st.pyplot(fig)


data = df.filter(['Close'])
dataset = data.values
#Get the number of rows to train the model on 
training_data_len = math.ceil(len(dataset)*.8)
#Scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set 
train_data = scaled_data[0:training_data_len, :]
x_train=[]
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
        
       
#convert the x_train and y_train to numpy array
x_train =np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#x_train.shape


#load my model
model=load_model("keras_model_web.h5")


#create the test dataset 
test_data = scaled_data[training_data_len-60:,:]
#create the data sets x_test and y_test
x_test=[]
y_test= dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
   
#convert the data to numpy array
x_test=np.array(x_test)


#reshape 
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))



#Get the models pridected values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)



#plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Prediction']= predictions
#visualization the data
st.subheader("Prediction vs original("+name+")")
fig2 = plt.figure(figsize=(12,6))
plt.title('Model')
plt.xlabel("Date ", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(["train","val", 'Prediction'],loc='lower right')

st.pyplot(fig2)


#get the quote 

apple_quote=web.DataReader(company,data_source='yahoo',start=start_date,end=end_date)
#create a new dataframe 
new_df= apple_quote.filter(['Close'])
#get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scale the data to be balues between 0 - 1 
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list 
x_test = []
x_test.append(last_60_days_scaled)
#convert the xtest data set to a numpy array 
x_test = np.array(x_test)
#Reshape the data 
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#get the predict scaled price 
pred_price = model.predict(x_test)
#undo the scaling
pred_price= scaler.inverse_transform(pred_price)
pred=st.sidebar.text_input("predected price",pred_price[0][0])

real=st.sidebar.text_input("Real price",pred_price[0][0])

