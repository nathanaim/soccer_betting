# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:57:18 2020

@author: natha
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

data_set_df=pd.read_csv(r'C:\Users\natha\Documents\DevProj\Soccer\dataset_Fifa_test.csv')

home_formation_one_hot=pd.get_dummies(data_set_df['home_formation'],prefix='home_formation')
away_formation_one_hot=pd.get_dummies(data_set_df['away_formation'],prefix='away_formation')

match_api_id=data_set_df.match_api_id
data=data_set_df.iloc[:,1:-3]
labels=data_set_df.iloc[:,-3:]

data1=data.loc[:,:'home_defenceDefenderLineClass_Offside Trap']
data1=pd.concat([data1,home_formation_one_hot],axis=1)
data2=data.loc[:,'home_1overall_rating':'away_defenceDefenderLineClass_Offside Trap']
data2=pd.concat([data1,data2,away_formation_one_hot],axis=1)
data3=data.loc[:,'away_1overall_rating':]
data=pd.concat([data2,data3],axis=1)

def scale_relevant_cols(df_train,df_test,cols_list):
    sscaler=StandardScaler()
    for col in cols_list:
        scaled_col=sscaler.fit_transform(df_train[col].to_numpy().reshape(-1,1))
        scaled_col_test=sscaler.transform(df_test[col].to_numpy().reshape(-1,1))
        scaled_col=scaled_col.reshape(-1,)
        scaled_col_test=scaled_col_test.reshape(-1,)
        scaled_col=pd.Series(scaled_col)
        scaled_col_test=pd.Series(scaled_col_test)
        df_train[col]=scaled_col
        df_test[col]=scaled_col_test
    return df_train,df_test

cols_to_keep_home=['home_buildUpPlaySpeedClass_Balanced','home_buildUpPlaySpeedClass_Fast','home_buildUpPlaySpeedClass_Slow','home_buildUpPlayDribblingClass_Little','home_buildUpPlayDribblingClass_Lots','home_buildUpPlayDribblingClass_Normal','home_buildUpPlayPassingClass_Long','home_buildUpPlayPassingClass_Mixed','home_buildUpPlayPassingClass_Short','home_buildUpPlayPositioningClass_Free Form','home_buildUpPlayPositioningClass_Organised','home_chanceCreationPassingClass_Normal','home_chanceCreationPassingClass_Risky','home_chanceCreationPassingClass_Safe','home_chanceCreationCrossingClass_Little','home_chanceCreationCrossingClass_Lots','home_chanceCreationCrossingClass_Normal','home_chanceCreationShootingClass_Little','home_chanceCreationShootingClass_Lots','home_chanceCreationShootingClass_Normal','home_chanceCreationPositioningClass_Free Form','home_chanceCreationPositioningClass_Organised','home_defencePressureClass_Deep','home_defencePressureClass_High','home_defencePressureClass_Medium','home_defenceAggressionClass_Contain','home_defenceAggressionClass_Double','home_defenceAggressionClass_Press','home_defenceTeamWidthClass_Narrow','home_defenceTeamWidthClass_Normal','home_defenceTeamWidthClass_Wide','home_defenceDefenderLineClass_Cover','home_defenceDefenderLineClass_Offside Trap','home_formation_0','home_formation_1','home_formation_2','home_formation_3','home_formation_4','home_formation_5','home_formation_6','home_formation_7','home_formation_8','home_formation_9']
cols_to_keep_away=[i.replace('home','away') for i in cols_to_keep_home]
cols_to_keep=cols_to_keep_home
cols_to_keep.extend(cols_to_keep_away)

cols_to_change=[i for i in list(data.columns) if i not in cols_to_keep]



data_train,data_test,label_train,label_test=train_test_split(data,labels,test_size=0.05,shuffle=False)
data_train.reset_index(inplace=True,drop=True)
data_test.reset_index(inplace=True,drop=True)

data_train,data_test=scale_relevant_cols(data_train,data_test,cols_to_change)

def get_team_inputs(df):
    away_team_inputs=df.loc[:,'away_buildUpPlaySpeed':'away_formation_9']
    away_team_inputs_np=np.array(away_team_inputs)
    home_team_inputs=df.loc[:,'home_buildUpPlaySpeed':'home_formation_9']
    home_team_inputs_np=np.array(home_team_inputs)
    two_layers_teams=np.stack([home_team_inputs_np,away_team_inputs_np],axis=-1)
    return two_layers_teams

def get_players_inputs(df):
    away_players_inputs=df.loc[:,'away_1overall_rating':]
    away_players_inputs_np=np.array(away_players_inputs)
    away_players_inputs_np=away_players_inputs_np.reshape(away_players_inputs_np.shape[0],-1,11,order='F')
    home_players_inputs=df.loc[:,'home_1overall_rating':'home_11gk_reflexes']
    home_players_inputs_np=np.array(home_players_inputs)
    home_players_inputs_np=home_players_inputs_np.reshape(home_players_inputs_np.shape[0],-1,11,order='F')
    two_layers_players=np.stack([home_players_inputs_np,away_players_inputs_np],axis=-1)
    return two_layers_players


modelplayers_inputs=tf.keras.Input(shape=(38,11,2))
modelplayers_layer1=Conv2D(16,kernel_size=(2,2),padding='same',activation='relu')(modelplayers_inputs)
modelplayers_layer2=Conv2D(8,kernel_size=(3,3),padding='same',activation='relu')(modelplayers_layer1)
modelplayers_layer3=MaxPooling2D()(modelplayers_layer2)
modelplayers_layer4=Flatten()(modelplayers_layer3)
modelplayers_layer5=Dense(64,activation='relu')(modelplayers_layer4)

modelteams_inputs=tf.keras.Input(shape=(52,2))
modelteams_layer1=Dense(128,activation='relu')(modelteams_inputs)
modelteams_layer2=Flatten()(modelteams_layer1)
modelteams_layer3=Dense(32,activation='relu')(modelteams_layer2)

concat=Concatenate()([modelplayers_layer5,modelteams_layer3])
model_layer1=Dense(128,activation='relu')(concat)
model_layer2=Dense(64,activation='relu')(model_layer1)
out=Dense(3,activation='softmax')(model_layer2)

model=tf.keras.Model([modelplayers_inputs,modelteams_inputs],out)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

players_train=get_players_inputs(data_train)
players_test=get_players_inputs(data_test)

team_train=get_team_inputs(data_train)
team_test=get_team_inputs(data_test)

model_save=ModelCheckpoint('best_model.hdf5',save_best_only=True)

fitting_history=model.fit([players_train,team_train],label_train,shuffle=False,validation_split=0.05,epochs=20,batch_size=1000,callbacks=[model_save])

best_model=load_model('best_model.hdf5')
print(best_model.evaluate([players_test,team_test],label_test))
