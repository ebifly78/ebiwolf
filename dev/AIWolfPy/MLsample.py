from __future__ import print_function, division

import os
import sys
import glob

import numpy as np
import pandas as pd
import sklearn.linear_model
import math

import aiwolfpy
import aiwolfpy.ebifly

from sklearn.svm import SVC

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

predictor = aiwolfpy.ebifly.Predictor_5()


def game_data_filter(df, day, phase='daily_initialize', agent=0):

    y = np.zeros(60)

    # werewolf, possessed
    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolf = i
        elif role == "POSSESSED":
            possessed = i

    for i in range(60):
        if predictor.case5.case60_df["agent_"+str(possessed)][i] == 2:
            if predictor.case5.case60_df["agent_"+str(werewolf)][i] == 1:
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
        df = df[df["type"].isin(
            ["talk", "vote", "execute", "dead", "whisper", "attack", "attack_vote"])]

    # agent
    if agent == 0:
        agent = 1

    # filter by time
    if phase == 'daily_initialize':
        df = df[df["day"] < day]
    else:
        df = df[(df["day"] < day) | (
            (df["day"] == day) & (df["type"] == 'talk'))]

    predictor.initialize(
        {"agentIdx": agent, "roleMap": {str(agent): role}}, {})
    predictor.update_features(df.reset_index())
    predictor.update_df()

    return predictor.df_pred, y


# build data for 100 games
# takes several minutes
match_num = 0
filenum = 1
file_list = glob.glob(
    '../../Server/AIWolf-ver0.5.6/log/file'+'{}'.format(filenum)+'/*.log')
for files in file_list:
    match_num += 1

days = 3
x_1000 = np.zeros((60*2*days*match_num, 1056))
# x_1000 = np.zeros((60*2*days*match_num, 72))
y_1000 = np.zeros(60*2*days*match_num)

ind = 0
filecount = 0
for i in range(match_num):
    # log_path = "../log/gat2017log05/000/" + "{0:03d}".format(i) + ".log"
    # log_path = '../../Server/AIWolf-ver0.5.6/log/file'+'{}'.format(filenum)+'/AIWolf20191002' + '{0:03d}'.format(i) + '.log'
    log_path = file_list[filecount]

    for d in range(days):
        x, y = game_data_filter(aiwolfpy.read_log(
            log_path), day=d, phase='vote')
        x_1000[(ind*60):((ind+1)*60), :] = x
        y_1000[(ind*60):((ind+1)*60)] = y
        ind += 1
        x, y = game_data_filter(aiwolfpy.read_log(
            log_path), day=d+1, phase='daily_initialize')
        x_1000[(ind*60):((ind+1)*60), :] = x
        y_1000[(ind*60):((ind+1)*60)] = y
        ind += 1
        if d == 0:
            filecount += 1


model = sklearn.linear_model.LogisticRegression()
# model = SVC(kernel='linear', random_state=None)
#model = RandomForestClassifier()
model.fit(x_1000, y_1000)

#joblib.dump(model, 'test.pkl')

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

print('file'+'{}'.format(filenum))
print(str(filecount)+'files')
print(np.array2string(model.coef_, separator=','))
print("modelData")
print(model.coef_)
print(x.columns.values)
