# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:58:31 2020

@author: natha
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve



con=sqlite3.connect(r'C:\Users\natha\Documents\DevProj\Soccer\SoccerDB.sqlite')

cur=con.cursor()

match=pd.read_sql_query('SELECT * FROM Match',con)
bookie_columns=match.columns[85:]
match_with_odds=match.dropna(subset=bookie_columns,how='all')
#id_and_odds=match_with_odds.drop(match_with_odds.columns[0:6],axis=1).drop(match_with_odds.columns[7:85],axis=1)
#the following line is preferred over the latter because it is ~10x faster
id_and_odds=match_with_odds.iloc[:,np.r_[6,85:len(match_with_odds.columns)]]

predictions_with_ids_df=pd.read_csv(r'C:\Users\natha\Documents\DevProj\Soccer\predictions_Fifa_cv_set.csv')
predictions_with_ids_df.columns=['match_api_id','1','N','2','predH','predD','predA']
#is of format (id,0/1,0/1,0/1,predH,predD,predA)

predictions_with_results_and_odds=predictions_with_ids_df.merge(id_and_odds,on='match_api_id',how='inner')

#turning my predictions into odds
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
predictions_with_results_and_odds['bestdif_col']=predictions_with_results_and_odds[dif_columns].idxmax(axis=1)
predictions_with_results_and_odds['bestbet_col']=predictions_with_results_and_odds[pred_columns].idxmin(axis=1)

predictions_with_results_and_odds['bestdif_dif']=predictions_with_results_and_odds[dif_columns].max(axis=1)
predictions_with_results_and_odds['bestbet_odds']=predictions_with_results_and_odds[pred_columns].min(axis=1)

predictions_with_results_and_odds['returnH']=predictions_with_results_and_odds['1']*predictions_with_results_and_odds['maxH']
predictions_with_results_and_odds['returnD']=predictions_with_results_and_odds['N']*predictions_with_results_and_odds['maxD']
predictions_with_results_and_odds['returnA']=predictions_with_results_and_odds['2']*predictions_with_results_and_odds['maxA']

#win_per_match_using_dif=(predictions_with_results_and_odds['bestdif']=='difH')*predictions_with_results_and_odds['returnH']+(predictions_with_results_and_odds['bestdif']=='difD')*predictions_with_results_and_odds['returnD']+(predictions_with_results_and_odds['bestdif']=='difA')*predictions_with_results_and_odds['returnA']

#win_per_match_using_bestbet=(predictions_with_results_and_odds['bestbet']=='predH')*predictions_with_results_and_odds['returnH']+(predictions_with_results_and_odds['bestbet']=='predD')*predictions_with_results_and_odds['returnD']+(predictions_with_results_and_odds['bestbet']=='predA')*predictions_with_results_and_odds['returnA']

#predictions_with_results_and_odds['bestbet_return']=predictions_with_results_and_odds.bestbet_col.apply(lambda x:x=='predA')*predictions_with_results_and_odds.returnA+predictions_with_results_and_odds.bestbet_col.apply(lambda x:x=='predD')*predictions_with_results_and_odds.returnD+predictions_with_results_and_odds.bestbet_col.apply(lambda x:x=='predH')*predictions_with_results_and_odds.returnH
#predictions_with_results_and_odds['bestdif_return']=predictions_with_results_and_odds.bestdif_col.apply(lambda x:x=='difA')*predictions_with_results_and_odds.returnA+predictions_with_results_and_odds.bestdif_col.apply(lambda x:x=='difD')*predictions_with_results_and_odds.returnD+predictions_with_results_and_odds.bestdif_col.apply(lambda x:x=='difH')*predictions_with_results_and_odds.returnH

def bestbet(x):
    if x.bestbet_col=='predH':
        return x.returnH
    elif x.bestbet_col=='predA':
        return x.returnA
    elif x.bestbet_col=='predD':
        return x.returnD

predictions_with_results_and_odds['bestbet_return']=predictions_with_results_and_odds.apply(bestbet,axis=1)

def bestdif(x):
    if x.bestdif_col=='difH':
        return x.returnH
    elif x.bestdif_col=='difA':
        return x.returnA
    elif x.bestdif_col=='difD':
        return x.returnD
    
predictions_with_results_and_odds['bestdif_return']=predictions_with_results_and_odds.apply(bestdif,axis=1)
dif_threshold=1.05
odds_threshold=2

dif_list=[i/100 for i in range(100,200,1)]
odds_list=[i/100 for i in range(200,300,1)]
return_dif_list=[]
return_odds_list=[]
number_of_bets_dif_list=[]
number_of_bets_odds_list=[]
for i in dif_list:
    number_of_bets=((predictions_with_results_and_odds.bestdif_dif>i)*1).sum()
    overall_return_bestdif=((predictions_with_results_and_odds.bestdif_dif>i)*(predictions_with_results_and_odds.bestdif_return)).sum()/number_of_bets
    return_dif_list.append(overall_return_bestdif)
    number_of_bets_dif_list.append(number_of_bets)
    
for i in odds_list:
    number_of_bets=((predictions_with_results_and_odds.bestbet_odds<i)*1).sum()
    overall_return_bestodds=((predictions_with_results_and_odds.bestbet_odds<i)*(predictions_with_results_and_odds.bestbet_return)).sum()/number_of_bets
    return_odds_list.append(overall_return_bestodds)
    number_of_bets_odds_list.append(number_of_bets)
    
plt.plot(odds_list,return_odds_list)
plt.xlabel('Max odds for betting')
plt.ylabel('Return obtained')
plt.show()
