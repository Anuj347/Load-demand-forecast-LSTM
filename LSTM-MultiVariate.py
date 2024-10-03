"""

LSTM - Multi Variant
develop a model for us to forecast for next 2 days or next week or next hours
"""

#Sometimes we have independent variable or predictors which affects the result(target varaiable)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Electricity+Consumption.csv')
# we dont need date any more it is just for reference
#humidity and Temperature are predictor(independant variables) whereas electricity is our target variable

df.dropna(inplace =True)

#we should check correlation
import seaborn as sn
sn.heatmap(df.corr())

#correlation of electricity with humidity and temperature is weak but lets hope that
# even if it is poor that can help our model to perform better

training_set = df.iloc[:8712,1:4].values
test_set = df.iloc[8712:,1:4].values #last 48hrs for test

#in case of multivariate, it is a must to normalize

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)

test_set_scaled = sc.fit_transform(test_set)

test_set_scaled = test_set_scaled[:, 0:2]  
#just independent variable not electricity(target variable)

X_train = []
Y_train = []
WS = 24

for i in range(WS,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i,0:3])
    Y_train.append(training_set_scaled[i,2])   #since 2 is target variable
    

X_train , Y_train = np.array(X_train) , np.array(Y_train) 
#we need them in the form of array
#you can see there are now 24 features in X_train
#so when you give a value to Window size, your whole data is multiplied by that 
#amount since the no . of features increases

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],3))
#ensuring the shape of X_train before fitting it to the model

"""
Modelling

"""

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model = Sequential()

Model.add(LSTM(units=70,return_sequences= True, input_shape=(X_train.shape[1],3)))#more the units more time better result its hybrid
Model.add(Dropout(0.2)) #just droping sme random neurons to avoid overfitting

Model.add(LSTM(units=70,return_sequences= True))#more the units more time better result its hybrid
Model.add(Dropout(0.2)) 

Model.add(LSTM(units=70,return_sequences= True))
Model.add(Dropout(0.2))

Model.add(LSTM(units=70))
Model.add(Dropout(0.2))

Model.add(Dense(units =1)) #for output

Model.compile(optimizer='adam',loss='mean_squared_error')

#now we just need to fit the data

Model.fit(X_train,Y_train,epochs=80,batch_size=32) #these are good numbers

#first step after doing this is plot loss vs epoch no. because after that you will
#not have history

plt.plot(range(len(Model.history.history['loss'])),Model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show() #used for rendering

#after verifying from last process only save the model
Model.save('LSTM-Multivariate')

from keras.models import load_model
Model = load_model('LSTM-Multivariate')

prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,3)) # 3 because we have 3 variable 1-target 2-independent

for i in range(48): #48 because it is the range of prediction horizon
    First_prediction= Model.predict(Batch_New)[0]
    #sot the model takes input in batches
    prediction_test.append(First_prediction)
    
    New_var = test_set_scaled[i,:]
    
    New_var = New_var.reshape(1,2)
    
    New_test = np.insert(New_var,2,[First_prediction],axis=1)
    
    New_test = New_test.reshape(1,1,3)
    
    Batch_New = np.append(Batch_New[:,1:,:],New_test,axis=1)


prediction_test = np.array(prediction_test)


#we cannot use sc because it was used for 3 features hence it will give error

SI = MinMaxScaler(feature_range=(0,1))
y_Scale = training_set[:,2:3]
SI.fit_transform(y_Scale)

predictions = SI.inverse_transform(prediction_test)

real_values = test_set[:,2:3]

plt.plot(real_values,color='red',label='Actual Values')
plt.plot(predictions,color='blue',label='Predicted Values')
plt.xlabel('Hours')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
RMSE = math.sqrt(mean_squared_error(real_values,predictions))

from sklearn.metrics import r2_score
Rsquare = r2_score(real_values,predictions)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_pred))*100

MAPE = mean_absolute_percentage_error(real_values,predictions )
