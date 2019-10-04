from __future__ import print_function, division 

import os 
import sys

import numpy as np
import pandas as pd
import sklearn.linear_model
import math

import aiwolfpy
import aiwolfpy.ebifly

predictor = aiwolfpy.ebifly.Predictor_15()

def game_data_filter(df, day, phase='daily_initialize', agent=0):
    
    
    y = np.zeros(5460)
    # werewolves, possessed
    werewolves = []
    for i in range(1, 16):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolves.append(i)
        elif role == "POSSESSED":
            possessed = i
            
    for i in range(5460):
        if predictor.case15.case5460_df["agent_"+str(possessed)][i] == 2:
            if predictor.case15.case5460_df["agent_"+str(werewolves[0])][i] == 1:
                if predictor.case15.case5460_df["agent_"+str(werewolves[1])][i] == 1:
                    if predictor.case15.case5460_df["agent_"+str(werewolves[2])][i] == 1:
                        y[i] = 1
    
    # role
    role = "VILLAGER"
    if agent > 0:
        role = df["text"][agent - 1].split()[2]
    
    # filter by role
    if role in ["VILLAGER", "POSSESSED"]:
        df = df[df["type"].isin(["talk", "vote", "execute", "dead"])]
    elif role == "MEDIUM":
        df = df[df["type"].isin(["talk", "vote", "execute", "dead", "identify"])]
    elif role == "SEER":
        df = df[df["type"].isin(["talk", "vote", "execute", "dead", "divine"])]
    elif role == "BODYGUARD":
        df = df[df["type"].isin(["talk", "vote", "execute", "dead", "guard"])]
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

#log_path = "../log/gat2017log05/000/000.log"

#x, y = game_data_filter(aiwolfpy.read_log(log_path), day=3, phase='vote')
"""
# build data for 100 games
# takes several minutes
x_1000 = np.zeros((60000, 72))
y_1000 = np.zeros(60000)

ind = 0
file_ok = 0
def_file_no = 115047
for i in range(100):
    # log_path = "../log/gat2017log05/000/" + "{0:03d}".format(i) + ".log"
    log_path = "../log/sample6/AIWolf20191002" + "{0:03d}".format(def_file_no+i) + ".log"
    
    for d in range(3):
        try:
            x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d, phase='vote')
            x_1000[(ind*60):((ind+1)*60), :] = x
            y_1000[(ind*60):((ind+1)*60)] = y
            ind += 1
            x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d+1, phase='daily_initialize')
            x_1000[(ind*60):((ind+1)*60), :] = x
            y_1000[(ind*60):((ind+1)*60)] = y
            ind += 1
        except FileNotFoundError:
            def_file_no += i + 1
            i -= 1
            print('detect')
            break
"""
# build data for 100 games
# takes several minutes
x_1000 = np.zeros((5460000, 60))
y_1000 = np.zeros(5460000)

ind = 0
filecount = 0
for i in range(999999):
    #log_path = "../log/sample13/AIWolf20191002" + "{0:03d}".format(def_file_no+i) + ".log"
    log_path = "../../Server/AIWolf-ver0.5.6/log/sample21/AIWolf20191002" + "{0:03d}".format(i) + ".log"
    for d in range(5):
        try:
            x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d, phase='vote')
            x_1000[(ind*5460):((ind+1)*5460), :] = x
            y_1000[(ind*5460):((ind+1)*5460)] = y
            ind += 1
            x, y = game_data_filter(aiwolfpy.read_log(log_path), day=d+1, phase='daily_initialize')
            x_1000[(ind*5460):((ind+1)*5460), :] = x
            y_1000[(ind*5460):((ind+1)*5460)] = y
            ind += 1
            if d == 0:
                filecount += 1
        except FileNotFoundError:
            break
    if filecount >= 100:
        break
    

model = sklearn.linear_model.LogisticRegression()
model.fit(x_1000, y_1000)

np.set_printoptions(suppress=True)

print(np.array2string(model.coef_, separator=','))
