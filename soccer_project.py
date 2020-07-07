# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:15:45 2020

@author: natha
"""

import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


con=sqlite3.connect(r'C:\Users\natha\Documents\DevProj\Soccer\SoccerDB.sqlite')

cur=con.cursor()

team=pd.read_sql_query('SELECT * FROM Team',con)
match=pd.read_sql_query('SELECT * FROM Match',con)
players=pd.read_sql_query('SELECT * FROM Player',con)
players_attributes=pd.read_sql_query('SELECT * FROM Player_Attributes',con)
team_attributes=pd.read_sql_query('SELECT * FROM Team_Attributes',con)


#dropping bookies odds
match_clean=match.iloc[:,:77]

#dropping matches for which we don't have all the infos
match_clean.dropna(inplace=True)

#fonction pour récupérer les attributs fifa des joueurs d'un match
def get_players_attributes(match_api_id):
    date_string=match_clean[match_clean['match_api_id']==match_api_id]['date'].iloc[0]
    date=datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S')
    home_team_players_id=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'home_player_1':'home_player_11']
    away_team_players_id=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'away_player_1':'away_player_11']
    home_team_players_attributes=[]
    for i in range(home_team_players_id.shape[1]):
        player_api_id=home_team_players_id.iloc[0,i]
        player_attributes_all=players_attributes[players_attributes['player_api_id']==player_api_id]
        player_attributes_all=player_attributes_all[player_attributes_all['date'].map(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))<=date]
        player_attributes=player_attributes_all.iloc[0,4:].to_numpy()
        home_team_players_attributes.extend(player_attributes)
    away_team_players_attributes=[]
    for i in range(away_team_players_id.shape[1]):
        player_api_id=away_team_players_id.iloc[0,i]
        player_attributes_all=players_attributes[players_attributes['player_api_id']==player_api_id]
        player_attributes_all=player_attributes_all[player_attributes_all['date'].map(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))<=date]
        player_attributes=player_attributes_all.iloc[0,4:].to_numpy()
        away_team_players_attributes.extend(player_attributes)
    return home_team_players_attributes,away_team_players_attributes

#fonction pour recuperer les positions des joueurs pour un match
def get_players_positions(match_api_id):
    home_team_coordinates=[]
    away_team_coordinates=[]
    home_team_X_axis=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'home_player_X1':'home_player_X11'].iloc[0,:]
    home_team_coordinates.extend(home_team_X_axis)
    away_team_X_axis=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'away_player_X1':'away_player_X11'].iloc[0,:]
    away_team_coordinates.extend(away_team_X_axis)
    home_team_Y_axis=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'home_player_Y1':'home_player_Y11'].iloc[0,:]
    home_team_coordinates.extend(home_team_Y_axis)
    away_team_Y_axis=match_clean[match_clean['match_api_id']==match_api_id].loc[:,'away_player_Y1':'away_player_Y11'].iloc[0,:]
    away_team_coordinates.extend(away_team_Y_axis)
    
    return home_team_coordinates,away_team_coordinates

#fonction pour remplir les players_attributes manquants
def players_attributes_preprocessing(players_attributes):
    players_attributes['overall_rating']=players_attributes['overall_rating'].fillna(players_attributes['overall_rating'].mean())
    players_attributes['potential']=players_attributes['potential'].fillna(players_attributes['potential'].mean())
    players_attributes['defensive_work_rate'][players_attributes['defensive_work_rate'].map(lambda x:x not in ['medium','high','low'])]='medium'
    players_attributes['defensive_work_rate'].replace({'low':1,'medium':2,'high':3},inplace=True)
    players_attributes['attacking_work_rate'][players_attributes['attacking_work_rate'].map(lambda x:x not in ['medium','high','low'])]='medium'
    players_attributes['attacking_work_rate'].replace({'low':1,'medium':2,'high':3},inplace=True)
    players_attributes['preferred_foot'][players_attributes['preferred_foot'].map(lambda x:x not in ['left','right'])]='right'
    players_attributes['preferred_foot'].replace({'left':-1,'right':1},inplace=True)
    players_attributes.iloc[:,9:]=players_attributes.iloc[:,9:].fillna(players_attributes.iloc[:,9:].mean())
    return players_attributes

def get_match_result(match_api_id):
    home_team_goal=match_clean[match_clean['match_api_id']==match_api_id]['home_team_goal'].iloc[0]
    away_team_goal=match_clean[match_clean['match_api_id']==match_api_id]['away_team_goal'].iloc[0]
    if home_team_goal>away_team_goal:
        result=np.array([1,0,0])
    elif away_team_goal>home_team_goal:
        result=np.array([0,0,1])
    else:
        result=np.array([0,1,0])
    return result, home_team_goal, away_team_goal

def team_attributes_preprocessing(team_attributes):
    team_attributes['buildUpPlayDribbling']=team_attributes['buildUpPlayDribbling'].fillna(team_attributes['buildUpPlayDribbling'].mean())
    team_attributes['buildUpPlaySpeedClass']=team_attributes['buildUpPlaySpeedClass'].replace({'Slow':1,'Balanced':2,'Fast':3})
    team_attributes['buildUpPlayDribblingClass']=team_attributes['buildUpPlayDribblingClass'].replace({'Little':1,'Normal':2,'Lots':3})
    team_attributes['buildUpPlayPassingClass']=team_attributes['buildUpPlayPassingClass'].replace({'Short':1,'Mixed':2,'Long':3})
    team_attributes['buildUpPlayPositioningClass']=team_attributes['buildUpPlayPositioningClass'].replace({'Organised':1,'Free Form':2})
    team_attributes['chanceCreationPassingClass']=team_attributes['chanceCreationPassingClass'].replace({'Safe':1,'Normal':2,'Risky':3})
    team_attributes['chanceCreationCrossingClass']=team_attributes['chanceCreationCrossingClass'].replace({'Little':1,'Normal':2,'Lots':3})
    team_attributes['chanceCreationShootingClass']=team_attributes['chanceCreationShootingClass'].replace({'Little':1,'Normal':2,'Lots':3})
    team_attributes['chanceCreationPositioningClass']=team_attributes['chanceCreationPositioningClass'].replace({'Organised':1,'Free Form':2})
    team_attributes['defencePressureClass']=team_attributes['defencePressureClass'].replace({'Deep':1,'Medium':2,'High':3})
    team_attributes['defenceAggressionClass']=team_attributes['defenceAggressionClass'].replace({'Press':1,'Double':2,'Contain':3})
    team_attributes['defenceTeamWidthClass']=team_attributes['defenceTeamWidthClass'].replace({'Wide':1,'Normal':2,'Narrow':3})
    team_attributes['defenceDefenderLineClass']=team_attributes['defenceDefenderLineClass'].replace({'Cover':1,'Offside Trap':2})
    return team_attributes

def get_teams_attributes(match_api_id):
    date_string=match_clean[match_clean['match_api_id']==match_api_id]['date'].iloc[0]
    date=datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S')
    home_team_api_id=match_clean[match_clean['match_api_id']==match_api_id]['home_team_api_id'].iloc[0]
    away_team_api_id=match_clean[match_clean['match_api_id']==match_api_id]['away_team_api_id'].iloc[0]
    home_team_attributes_all=team_attributes[team_attributes['team_api_id']==home_team_api_id]
    home_team_attributes_all_test=home_team_attributes_all[home_team_attributes_all['date'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S')<=date)]
    if home_team_attributes_all_test.empty:
        home_team_attributes=home_team_attributes_all.iloc[0,4:].to_numpy()
    else:
        home_team_attributes_all_test.sort_values(by=['date'],ascending=False,inplace=True)
        home_team_attributes=home_team_attributes_all_test.iloc[0,4:].to_numpy()
    away_team_attributes_all=team_attributes[team_attributes['team_api_id']==away_team_api_id]
    away_team_attributes_all_test=away_team_attributes_all[away_team_attributes_all['date'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S')<=date)]
    if away_team_attributes_all_test.empty:
        away_team_attributes=away_team_attributes_all.iloc[0,4:].to_numpy()
    else:
        away_team_attributes_all_test.sort_values(by=['date'],ascending=False,inplace=True)
        away_team_attributes=away_team_attributes_all_test.iloc[0,4:].to_numpy()
    return home_team_attributes,away_team_attributes

def build_data_for_match(match_api_id):
    home_team_players_attributes,away_team_players_attributes=get_players_attributes(match_api_id)
    home_team_coordinates,away_team_coordinates=get_players_positions(match_api_id)
    home_team_attributes,away_team_attributes=get_teams_attributes(match_api_id)
    result, home_team_goal, away_team_goal=get_match_result(match_api_id)
    home_team_data=np.concatenate((home_team_attributes,home_team_coordinates,home_team_players_attributes))
    away_team_data=np.concatenate((away_team_attributes,away_team_coordinates,away_team_players_attributes))
    match_data=np.concatenate(([match_api_id],home_team_data,away_team_data))
    data_and_result=np.concatenate((match_data,result))
    return data_and_result

def build_data_set(match_api_id_list):
    error_count=0
    data_set=build_data_for_match(match_api_id_list.iloc[0])
    for i in range(1,len(match_api_id_list)):
        try:
            data_and_result=build_data_for_match(match_api_id_list.iloc[i])
            data_set=np.vstack((data_and_result,data_set))
        except:
            error_count+=1
            print("Number of errors : {}".format(error_count))
        if (i-error_count)%500==0:
            print("Dataset is now composed of {} examples".format(i-error_count))
    return data_set

players_attributes=players_attributes_preprocessing(players_attributes)
team_attributes=team_attributes_preprocessing(team_attributes)

match_api_id_list=match_clean['match_api_id'].iloc[:]

data_set=build_data_set(match_api_id_list)

pd.DataFrame(data_set).to_csv(r'C:\Users\natha\Documents\DevProj\Soccer\dataset_Fifa.csv',header=None,index=None)

