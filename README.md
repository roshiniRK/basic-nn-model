# Developing a Neural Network Regression Model

### AIM

To develop a neural network regression model for the given dataset.

### THEORY

Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error.

### Neural Network Model

![WhatsApp Image 2024-08-24 at 15 25 44_855f4908](https://github.com/user-attachments/assets/f608dfb6-59c2-4278-8e09-7281027e2939)


### DESIGN STEPS

- STEP 1:Loading the dataset
  
- STEP 2:Split the dataset into training and testing
  
- STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
  
- STEP 4:Build the Neural Network Model and compile the model.
  
- STEP 5:Train the model with the training data.
  
- STEP 6:Plot the performance plot
  
- STEP 7:Evaluate the model with the testing data.

### PROGRAM

#### Name: ROSHINI R K
#### Register Number: 212222230123

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open("ex1DL").sheet1
df=worksheet.get_all_values()
print(df)
ds1=pd.DataFrame(df[1:],columns=df[0])
ds1=ds1.astype({'input':'float'})
ds1=ds1.astype({'output':'float'})
ds1.head()
x = ds1[['input']].values
y = ds1[['output']].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
xtrain=scaler.transform(x_train)
model=Sequential([Dense(8,activation="relu",input_shape=[1]),Dense(10,activation="relu"),Dense(1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain,y_train,epochs=2000)
cf=pd.DataFrame(model.history.history)
cf.plot()
xtrain=scaler.transform(x_test)
model.evaluate(xtrain,y_test)
n=[[17]]
n=scaler.transform(n)
model.predict(n)
```
### Dataset Information

#### DATASET.HEAD():
![image](https://github.com/user-attachments/assets/a7d1d93c-943c-434d-ab0f-12ab5ff50f28)
#### DATASET.INFO()
![image](https://github.com/user-attachments/assets/9a11303d-19df-47d9-9ebc-4eb59b14ea47)
#### DATASET.DESCRIBE()
![image](https://github.com/user-attachments/assets/38f7852f-bbc7-4ebd-b1a9-d398f7eb7f39)

### OUTPUT

#### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/962afafc-c381-4d47-8da0-f522b12f4f94)


#### Test Data Root Mean Squared Error 
![image](https://github.com/user-attachments/assets/ee571a1c-fc4c-4939-8357-e711aaa1eab5)

### RESULT
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
