from __future__ import print_function, division

import os
import sys
import glob

import numpy as np
import pandas as pd
import sklearn.linear_model
import math

import aiwolfpy
import aiwolfpy.ebifly as ebi

from sklearn.svm import SVC

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)


predictor = aiwolfpy.ebifly.Predictor_5()


def game_data_filter(df, day, agent=0):

    y = np.zeros(60)
    z = np.zeros(60)

    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolf = i
        elif role == "POSSESSED":
            possessed = i
        elif role == "SEER":
            seer = i

    for i in range(60):
        if predictor.case5.case60_df["agent_"+str(werewolf)][i] == 1:
            z[i] = 1
            if predictor.case5.case60_df["agent_"+str(seer)][i] == 3:
                if predictor.case5.case60_df["agent_"+str(possessed)][i] == 2:
                    y[i] = 1

    # role
    role = "VILLAGER"
    if agent > 0:
        role = df["text"][agent - 1].split()[2]

    execute = df[(df["day"] < day) & (
        df["type"] == 'execute')]["agent"].values
    dead = df[(df["day"] < day) & (df["type"] == 'dead')]["agent"].values

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

    df = df[(df["day"] < day) | ((df["day"] == day) & (df["type"] == 'talk'))]

    statusMap = {'1': 'ALIVE', '2': 'ALIVE',
                 '3': 'ALIVE', '4': 'ALIVE', '5': 'ALIVE'}
    for i in execute:
        statusMap[str(i)] = 'DEAD'
    for i in dead:
        statusMap[str(i)] = 'DEAD'

    predictor.initialize(
        {"agentIdx": agent, "roleMap": {str(agent): role}, "statuMap": statusMap}, {})
    predictor.update_features(df.reset_index())
    predictor.update_df()

    return predictor.df_pred, y, z


# build data for 100 games
# takes several minutes
match_num = 0
foldernum = 20
file_list = []
folder_list = ['../../Server/AIWolf-ver0.5.6/log/file' +
               '{}'.format(foldernum)+'/']
# folder_list = glob.glob('../../Server/AIWolf-ver0.5.6/log/log_cedec2018/*')

for folder in folder_list:
    file_list += glob.glob(folder + '/*')
for files in file_list:
    match_num += 1

days = 3
x_data = np.zeros((60*days*match_num, predictor.n_para))
y_data = np.zeros(60*days*match_num)
z_data = np.zeros(60*days*match_num)

ind = 0
filecount = 0

for files in file_list:
    for d in range(days):
        x, y, z = game_data_filter(aiwolfpy.read_log(files), day=d)
        x_data[(ind*60):((ind+1)*60), :] = x
        y_data[(ind*60):((ind+1)*60)] = y
        z_data[(ind*60):((ind+1)*60)] = z
        ind += 1

        if d == 0:
            filecount += 1

# model = sklearn.linear_model.LogisticRegression()
# model = SVC(kernel='linear', random_state=None)
model = RandomForestClassifier()

model.fit(x_data, y_data)
joblib.dump(model, 'ebiwolf.pkl')

print('file'+'{}'.format(foldernum))
print(str(filecount)+'files')

# print("modelData")
# print(model.coef_)
# print(x.columns.values)
"""
fti = model.feature_importances_
ids = np.argsort(fti)[::-1]
print("modelData")
for i in range(len(ids)):
    for j in ids:
        if j == i:
            print('{0:20s} : {1:>.6f}'.format(
                x.columns.values[ids[i]], fti[ids[i]]))
        if fti[ids[i]] == 0:
            break
    else:
        continue
    break
"""
