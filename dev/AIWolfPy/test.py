from __future__ import print_function, division 

import numpy as np
import pandas as pd

import aiwolfpy
import aiwolfpy.cash

predictor = aiwolfpy.cash.Predictor_15()

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

log_path = "../log/gat2017log15/000/000.log"

x, y = game_data_filter(aiwolfpy.read_log(log_path), day=3, phase='vote')

print(x, y)