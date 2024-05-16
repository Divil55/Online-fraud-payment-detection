import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("online payment fraud dataset.csv")
data.shape
data
data["type"].value_counts()
data.isnull()
data.isnull().sum()
data = data.drop(columns="nameOrig",axis=1)
data = data.drop(columns="nameDest",axis=1)
data = data.drop(columns="oldbalanceDest",axis=1)
data = data.drop(columns="newbalanceDest",axis=1)
data = data.drop(columns="isFlaggedFraud",axis=1)
data["type"].value_counts()
data.replace({"type":{"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5,}},inplace=True)
data["type"].value_counts()
x=data.drop(columns="isFraud",axis=1)
y=data["isFraud"]
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , stratify = y, random_state=1)
print(x_train.shape, x_test.shape, x.shape, y_train.shape)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
x_trained_data = model.predict(x_train)
x_trained_data.shape
train_data_accuracy = accuracy_score(x_trained_data,y_train)
print(train_data_accuracy)

#Plotting using Traning data

plt.plot(x_trained_data, y_train)
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.title("Online Payment Fraud Detection")
x_tested_data = model.predict(x_test)
test_data_accuracy = accuracy_score(x_tested_data, y_test)
print(test_data_accuracy)

#Plotting using Test data

plt.plot(x_tested_data, y_test)
plt.xlabel("x_test")
plt.ylabel("y_test")
plt.title("Online Payment Fraud Detection")

#Validating Payments

input_data = (1,1,1864.28,170136.00,160296.36)
input_numpy_data = np.asarray(input_data)

input_reshaped_data = input_numpy_data.reshape(1 , -1)
output_data = model.predict(input_reshaped_data)

if output_data[0] == 1:
    print("Fraud payment has been done")
else:
    print("Legal payment has been done")

input_data = (850002.52,181.00,339682.13,181.00,11668.14)
input_numpy_data = np.asarray(input_data)

input_reshaped_data = input_numpy_data.reshape(1 , -1)
output_data = model.predict(input_reshaped_data)

if output_data[0] == 1:
    print("Fraud payment has been done")
else:
    print("Legal payment has been done")
