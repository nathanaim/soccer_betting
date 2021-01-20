# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:37:26 2020

@author: natha
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.linear_model import LogisticRegression

con=sqlite3.connect(r'C:\Users\natha\Documents\DevProj\Soccer\SoccerDB.sqlite')

cur=con.cursor()

match=pd.read_sql_query('SELECT * FROM Match',con)
bookie_columns=match.columns[85:]
match_with_odds=match.dropna(subset=bookie_columns,how='all')

id_and_odds=match_with_odds.iloc[:,np.r_[6,85:len(match_with_odds.columns)]]

home_columns=['B365H','BWH','IWH','LBH','PSH','WHH','SJH','VCH','GBH','BSH']
draw_columns=['B365D','BWD','IWD','LBD','PSD','WHD','SJD','VCD','GBD','BSD']
away_columns=['B365A','BWA','IWA','LBA','PSA','WHA','SJA','VCA','GBA','BSA']

id_and_odds['odds_H']=id_and_odds[home_columns].max(axis=1)
id_and_odds.drop(home_columns,axis=1,inplace=True)
id_and_odds['odds_D']=id_and_odds[draw_columns].max(axis=1)
id_and_odds.drop(draw_columns,axis=1,inplace=True)
id_and_odds['odds_A']=id_and_odds[away_columns].max(axis=1)
id_and_odds.drop(away_columns,axis=1,inplace=True)

data_set_df=pd.read_csv(r'C:\Users\natha\Documents\DevProj\Soccer\dataset_Fifa_test.csv')

data_set_with_odds=data_set_df.merge(id_and_odds, on='match_api_id', how='inner')
data_set_with_odds.reset_index(inplace=True)
actual_returns_np=np.array(data_set_with_odds.iloc[:,-6:-3])*np.array(data_set_with_odds.iloc[:,-3:])
actual_returns_df=pd.DataFrame(actual_returns_np,columns=['returnH','returnD','returnA'])
data_set_with_odds=pd.concat([data_set_with_odds,actual_returns_df],axis=1)

odds=data_set_with_odds[['odds_H','odds_D','odds_A']]

def custom_loss_weighted(y_true,y_pred):
    loss=-K.sum(K.log(y_pred)*y_true)
    return loss

def average_return(y_true,y_pred):
    pred_indices=K.argmax(y_pred,axis=1)
    pred=K.one_hot(pred_indices,num_classes=3)
    ret=pred*y_true
    number_of_matches=K.sum(pred)
    overall_ret=K.sum(ret)
    avg_ret=overall_ret/number_of_matches
    return avg_ret

class strategic_return(tf.keras.callbacks.Callback):
    def __init__(self,players_train,team_train,players_test,team_test,odds_train,odds_test,label_test,label_train):
        self.label_test=label_test.to_numpy()
        self.label_train=label_train.to_numpy()
        self.odds_train=odds_train.to_numpy()
        self.odds_test=odds_test.to_numpy()
        self.players_train=players_train
        self.players_test=players_test
        self.team_train=team_train
        self.team_test=team_test
        
    def on_epoch_end(self,epoch,logs=None):
        players_test=self.players_test
        team_test=self.team_test
        label_test=self.label_test
        num_of_matches=players_test.shape[0]
        y_pred_test=self.model.predict([players_test,team_test])
        y_odds_test=1/y_pred_test
        odds_test=self.odds_test
        odds_diff=odds_test-y_odds_test
        bet_indices=np.argmax(odds_diff,axis=1)
        bet_indices=bet_indices.reshape(-1)
        bet_mask=np.eye(3)[bet_indices,:]
        ret=bet_mask*label_test
        overall_ret=np.sum(ret)
        avg_ret=overall_ret/num_of_matches
        print(' Average return on test_data is : '+str(avg_ret))
        

def average_return_strategic(odds):
    odds=tf.convert_to_tensor(odds.to_numpy())
    odds=K.cast(odds,'float32')
    def return_strat(y_true,y_pred):
        odds_pred=1/y_pred
        odds_diff=odds-odds_pred
        bet_indices=K.argmax(odds_diff,axis=1)
        bet_mask=K.one_hot(bet_indices,num_classes=3)
        num_of_matches=K.sum(bet_mask)
        ret=bet_mask*y_true
        overall_ret=K.sum(ret)
        avg_ret=overall_ret/num_of_matches
        return avg_ret
    return return_strat

def custom_loss_unweighted(y_true,y_pred):
    a=K.sum(y_pred)
    y_pred/=a
    res_indices=K.argmax(y_true,axis=1)
    res=K.one_hot(res_indices,num_classes=3)
    loss=-K.log(y_pred)*res
    return loss

def metrics_test(y_true,y_pred):
    number_of_matches=y_true.shape[0]
    pred_indices=K.argmax(y_pred,axis=1)
    pred=K.one_hot(pred_indices,num_classes=3)
    cross=pred*y_true
    number_right=K.sum(cross)
    acc=number_right/number_of_matches
    return acc


home_formation_one_hot=pd.get_dummies(data_set_with_odds['home_formation'],prefix='home_formation')
away_formation_one_hot=pd.get_dummies(data_set_with_odds['away_formation'],prefix='away_formation')

match_api_id=data_set_with_odds.match_api_id
data=data_set_with_odds.iloc[:,1:-9]
labels=data_set_with_odds.iloc[:,-3:]

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

odds_train=odds.iloc[:data_train.shape[0],:]
odds_test=odds.iloc[data_train.shape[0]:,:]

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

oddsbookies_input=tf.keras.Input(shape=(3,))
new_concat=Concatenate()([out,oddsbookies_input])

model=tf.keras.Model([modelplayers_inputs,modelteams_inputs],out)

model_ens=tf.keras.Model([modelplayers_inputs,modelteams_input,oddsbookies_input],new_out)

model.compile(optimizer='adam',loss=custom_loss_unweighted,metrics=['accuracy'])

players_train=get_players_inputs(data_train)
players_test=get_players_inputs(data_test)

team_train=get_team_inputs(data_train)
team_test=get_team_inputs(data_test)

model_save=ModelCheckpoint('best_model_test.hdf5',save_best_only=True)

label_train=label_train.astype('float32')
label_test=label_test.astype('float32')

fitting_history=model.fit([players_train,team_train],label_train,shuffle=False,validation_split=0.05,epochs=20,batch_size=1000,callbacks=[model_save,strategic_return(players_train,team_train,players_test,team_test,odds_train,odds_test,label_test,label_train)])

best_model=load_model('best_model_test.hdf5',compile=False)
best_model.compile(optimizer='adam',loss=custom_loss_unweighted,metrics=['accuracy',average_return])

print(best_model.evaluate([players_test,team_test],label_test))
