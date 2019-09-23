from __future__ import print_function, division 

import os 
import sys

import numpy as np
import pandas as pd
import sklearn.linear_model
import math

import aiwolfpy
import aiwolfpy.ebifly

predictor = aiwolfpy.ebifly.Predictor_5()

def game_data_filter(df, day, phase='daily_initialize', agent=0):
    
    
    y = np.zeros(60)
    # werewolves, possessed
    werewolves = []
    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolves.append(i)
        elif role == "POSSESSED":
            possessed = i
            
    for i in range(60):
        if predictor.case5.case60_df["agent_"+str(possessed)][i] == 2:
            if predictor.case5.case60_df["agent_"+str(werewolves[0])][i] == 1:
                y[i] = 1
    
    # role
    role = "VILLAGER"
    if agent > 0:
        role = df["text"][agent - 1].split()[2]
    
    # filter by role
    if role in ["VILLAGER", "POSSESSED"]:
        df = df[df["type"].isin(["talk", "vote", "execute", "dead"])]
    elif role == "SEER":
        df = df[df["type"].isin(["talk", "vote", "execute", "dead", "divine"])]
    elif role == "WEREWOLF":
        df = df[df["type"].isin(["talk", "vote", "execute", "dead", "whisper", "attack", "attack_vote"])]

    # agent
    if agent == 0:
        agent = 1
    
    # filter by time
    if phase == 'daily_initialize':
        df = df[df["day"] < day]
    else:
        df = df[(df["day"] < day) | ((df["day"] == day) & (df["type"] == 'talk'))]
    
    predictor.initialize({"agent":agent, "roleMap":{str(agent):role}}, {})
    predictor.update_features(df.reset_index())
    predictor.update_df()
    
    return predictor.df_pred, y

log_path = "../log/gat2017log05/000/000.log"

x, y = game_data_filter(aiwolfpy.read_log(log_path), day=3, phase='vote')

# build data for 100 games
# takes several minutes
x_1000 = np.zeros((60000, 72))
y_1000 = np.zeros(60000)

ind = 0
for i in range(100):
    log_path = "../log/gat2017log05/000/" + "{0:03d}".format(i) + ".log"
    for d in range(3):
        x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d, phase='vote')
        x_1000[(ind*60):((ind+1)*60), :] = x
        y_1000[(ind*60):((ind+1)*60)] = y
        ind += 1
        x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d+1, phase='daily_initialize')
        x_1000[(ind*60):((ind+1)*60), :] = x
        y_1000[(ind*60):((ind+1)*60)] = y
        ind += 1

model = sklearn.linear_model.LogisticRegression()
model.fit(x_1000, y_1000)

np.set_printoptions(suppress=True)

print(np.array2string(model.coef_, separator=','))
