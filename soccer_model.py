# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:57:18 2020

@author: natha
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

#from soccer_project import 

data_set_df=pd.read_csv(r'C:\Users\natha\Documents\DevProj\Soccer\dataset_Fifa.csv')
data_set=data_set_df.to_numpy()

train_set=data_set[:19000,:]
cv_set=data_set[19000:20000,:]
test_set=data_set[20000:,:]

train_set_data=train_set[:,1:-3]
train_set_labels=train_set[:,-3:]
train_set_ids=train_set[:,0].reshape((-1,1))

cv_set_data=cv_set[:,1:-3]
cv_set_labels=cv_set[:,-3:]
cv_set_ids=cv_set[:,0].reshape((-1,1))

test_set_data=test_set[:,1:-3]
test_set_labels=test_set[:,-3:]
test_set_ids=test_set[:,0].reshape((-1,1))

model=keras.models.Sequential([
    keras.layers.Dense(2048,activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(3,activation='softmax')])

opt=keras.optimizers.Adam()
loss_fn=keras.losses.CategoricalCrossentropy()

model.compile(optimizer=opt,loss=loss_fn,metrics=['accuracy'])

model.fit(train_set_data,train_set_labels,batch_size=1000,epochs=100)

model.evaluate(cv_set_data,cv_set_labels)

predictions=model.predict(test_set_data)
predictions_with_ids=np.concatenate((test_set_ids,test_set_labels,predictions),axis=1)

pd.DataFrame(predictions_with_ids).to_csv(r'C:\Users\natha\Documents\DevProj\Soccer\predictions_Fifa.csv',header=None,index=None)