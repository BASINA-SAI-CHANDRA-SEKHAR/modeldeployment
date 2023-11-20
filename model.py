import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("dataset.csv")
print(data)
print(data.shape)
x=data.drop('price',axis='columns').reshape(-1,1)
y=data.price.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
import pickle
pickle.dump(model,open("model.pkl","wb"))
