# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:12:13 2020

@author: natha
"""

import numpy as np
import pandas as pd
import sqlite3

con=sqlite3.connect(r'C:\Users\natha\Documents\DevProj\Soccer\SoccerDB.sqlite')

cur=con.cursor()

match=pd.read_sql_query('SELECT * FROM Match',con)
bookie_columns=match.columns[85:]
match_with_odds=match.dropna(subset=bookie_columns,how='all')
id_and_odds=match_with_odds.drop(match_with_odds.columns[0:6],axis=1).drop(match_with_odds.columns[7:85],axis=1)

predictions_with_ids_df=pd.read_csv(r'C:\Users\natha\Documents\DevProj\Soccer\predictions_Fifa.csv')
predictions_with_ids_df.columns=['match_api_id','1','N','2','predH','predD','predA']
#is of format (id,0/1,0/1,0/1,predH,predD,predA)

predictions_with_results_and_odds=predictions_with_ids_df.join(id_and_odds.set_index('match_api_id'),on='match_api_id',how='inner')
predictions_with_results_and_odds['predH']=1/predictions_with_results_and_odds['predH']
predictions_with_results_and_odds['predD']=1/predictions_with_results_and_odds['predD']
predictions_with_results_and_odds['predA']=1/predictions_with_results_and_odds['predA']

home_columns=['B365H','BWH','IWH','LBH','PSH','WHH','SJH','VCH','GBH','BSH']
draw_columns=['B365D','BWD','IWD','LBD','PSD','WHD','SJD','VCD','GBD','BSD']
away_columns=['B365A','BWA','IWA','LBA','PSA','WHA','SJA','VCA','GBA','BSA']
pred_columns=['predH','predD','predA']

predictions_with_results_and_odds['maxH']=predictions_with_results_and_odds[home_columns].max(axis=1)
predictions_with_results_and_odds['maxD']=predictions_with_results_and_odds[draw_columns].max(axis=1)
predictions_with_results_and_odds['maxA']=predictions_with_results_and_odds[away_columns].max(axis=1)

predictions_with_results_and_odds['difH']=predictions_with_results_and_odds['maxH']/predictions_with_results_and_odds['predH']
predictions_with_results_and_odds['difD']=predictions_with_results_and_odds['maxD']/predictions_with_results_and_odds['predD']
predictions_with_results_and_odds['difA']=predictions_with_results_and_odds['maxA']/predictions_with_results_and_odds['predA']

dif_columns=['difH','difD','difA']
predictions_with_results_and_odds['bestdif']=predictions_with_results_and_odds[dif_columns].idxmax(axis=1)
predictions_with_results_and_odds['bestbet']=predictions_with_results_and_odds[pred_columns].idxmax(axis=1)

predictions_with_results_and_odds['returnH']=predictions_with_results_and_odds['1']*predictions_with_results_and_odds['maxH']
predictions_with_results_and_odds['returnD']=predictions_with_results_and_odds['N']*predictions_with_results_and_odds['maxD']
predictions_with_results_and_odds['returnA']=predictions_with_results_and_odds['2']*predictions_with_results_and_odds['maxA']

win_per_match_using_dif=(predictions_with_results_and_odds['bestdif']=='difH')*predictions_with_results_and_odds['returnH']+(predictions_with_results_and_odds['bestdif']=='difD')*predictions_with_results_and_odds['returnD']+(predictions_with_results_and_odds['bestdif']=='difA')*predictions_with_results_and_odds['returnA']

win_per_match_using_bestbet=(predictions_with_results_and_odds['bestbet']=='predH')*predictions_with_results_and_odds['returnH']+(predictions_with_results_and_odds['bestbet']=='predD')*predictions_with_results_and_odds['returnD']+(predictions_with_results_and_odds['bestbet']=='predA')*predictions_with_results_and_odds['returnA']

overall_win_bestdif=np.sum(win_per_match_using_dif)
overall_win_bestbet=np.sum(win_per_match_using_bestbet)

print(overall_win_bestdif,overall_win_bestbet)