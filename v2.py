"""
Created on Sat Dec 23 17:24:47 2023

@author: ekizc
"""

import pandas as pd
import numpy as np
#%%
data = pd.read_excel("ikisibirlesim.xlsx")
#%%
from sklearn.preprocessing import LabelEncoder
le_home = LabelEncoder()
le_away = LabelEncoder()
data["Home"] = le_home.fit_transform(data["Home"])
data["Away"] = le_away.fit_transform(data["Away"])
#%%
x = data.drop(["Sıralama","Result","Target","GoalHome","GoalAway"],axis=1)
y= data["Target"]
#%%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=33)
#%%
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)
prednew = np.array([1, 16, 2.11, 2.55, 2.98, 1]).reshape(1, -1)
prednew = sc.transform(prednew)
#%%
from sklearn.ensemble import RandomForestClassifier
rm=RandomForestClassifier(n_estimators=100,random_state=65)
rm.fit(xtrain,ytrain)
yhead=rm.predict(xtest)
rm.score(xtest,ytest)
#%%
home_encoded = le_home.transform(["Bologna"])
away_encoded = le_away.transform(["Roma"])
#%%

#%%
# Make predictions
print(rm.predict(prednew))
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yhead, ytest)
#%%
import tensorflow as tf
#%%
ann=tf.keras.models.Sequential()
input_dim = xtrain.shape[1]  # xtrain veri setinizin özellik sayısı
#%%
ann.add(tf.keras.layers.Dense(units=10,activation="relu",input_dim=input_dim))
ann.add(tf.keras.layers.Dense(units=10,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann.summary()
history=ann.fit(xtrain,ytrain,batch_size=1,epochs=18,validation_data=(xtest,ytest))
#%%
print(ann.predict(prednew))

#batchsize ne kadar azsa o kadar iyi öğrenir ve daha çok bellek harcar


