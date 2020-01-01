from __future__ import print_function, division

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import math
import random
import csv
import itertools

import resource

import aiwolfpy
import aiwolfpy.ebifly as ebi

import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

resource.setrlimit(resource.RLIMIT_DATA, (10 * 1024**3, -1))


np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)


predictor = aiwolfpy.ebifly.Predictor_5()


def game_data_filter(df, day, pat, agent=0):

    y = np.zeros(60)

    count = 0
    for i in range(1, 6):
        df['agent'] = df['agent'].replace({i: i*11})
        df['text'] = df['text'].str.replace(str(i), str(i*11))
    for list1 in itertools.permutations(range(1, 6)):
        if pat <= count:
            for i in range(1, 6):
                df['agent'] = df['agent'].replace({i*11: list1[i-1]})
                df['text'] = df['text'].str.replace(str(i*11), str(list1[i-1]))
            break
        count += 1

    # for i in range(1, 6):
    #     df['agent'] = df['agent'].replace({i: i*11})
    # for i in range(1, 6):
    #     if i+add < 5:
    #         df['agent'] = df['agent'].replace({i*11: i+add})
    #     else:
    #         df['agent'] = df['agent'].replace({i*11: 1+add})

    # for i in range(1, 6):
    #     df['text'] = df['text'].str.replace(str(i), str(i*11))
    # for i in range(1, 6):
    #     if i+add < 5:
    #         df['text'] = df['text'].str.replace(str(i*11), str(i+add))
    #     else:
    #         df['text'] = df['text'].str.replace(str(i*11), str(1+add))

    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolf = df['agent'][i-1]
        elif role == "POSSESSED":
            possessed = df['agent'][i-1]
        elif role == "SEER":
            seer = df['agent'][i-1]

    for i in range(60):
        if predictor.case5.case60_df["agent_"+str(werewolf)][i] == 1:
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
        {"agentIdx": agent, "roleMap": {str(agent): role}, "statusMap": statusMap}, {})
    predictor.update_features(df.reset_index())
    predictor.update_df()

    return predictor.df_pred, y


def estimate(df, day, agent=0):

    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        if role == "WEREWOLF":
            werewolf = df['agent'][i-1]
        elif role == "POSSESSED":
            possessed = df['agent'][i-1]
        elif role == "SEER":
            seer = df['agent'][i-1]
        elif role == "VILLAGER":
            villager = df['agent'][i-1]
    agent = villager

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
    if statusMap[str(werewolf)] == 'DEAD':
        return -1

    predictor.initialize(
        {"agentIdx": agent, "roleMap": {str(agent): role}, "statusMap": statusMap}, {})
    predictor.update(df.reset_index())

    p = -1
    idx = 1
    p_5 = np.zeros((5, 4), dtype='float32')
    for i in range(60):
        for j in range(5):
            p_5[j, predictor.case5.case60[i, j]
                ] += predictor.p_60[i]
    p_mat = p_5
    for i in range(5):
        if p_mat[i, 1] > p:
            p = p_mat[i, 1]
            idx = i
    est = idx + 1

    # print('day = {}, agent = {}'.format(day, agent))
    # print(df)
    # print(statusMap)
    # print(predictor.p_60)
    # print(est == werewolf)

    if statusMap[str(agent)] == 'DEAD':
        return -1

    if est == werewolf:
        return 1
    else:
        return 0


start = time.time()

file_list = []
logdir = '../../Server/AIWolf-ver0.5.6/log/'
# folder_list = [logdir + 'file1/']
folder_list = glob.glob(logdir + 'gat2017log05/*')

for folder in folder_list:
    file_list += glob.glob(folder + '/*')

np.random.shuffle(file_list)

print("get_files = {}[sec]".format(time.time() - start))

for j in range(0, 6):
    for k in range(1, 10):
        if (k*10**j) > len(file_list):
            sys.exit()

        file_limit = (k*10**j)*2

        filing = file_list[:file_limit]
        file_num = len(filing)

        split_id = int(file_num * 0.5)
        train_file = filing[:split_id]
        test_file = filing[split_id:]

        train_num = len(train_file)
        test_num = len(test_file)

        days = 3
        expan = 1
        x_data = np.zeros((60*days*train_num*expan, predictor.n_para))
        y_data = np.zeros(60*days*train_num*expan)

        start1 = time.time()

        ind = 0
        for files in train_file:
            for i in range(expan):
                a = random.randint(0, 119)
                for d in range(days):
                    x, y = game_data_filter(
                        aiwolfpy.read_log(files), day=d, pat=a)
                    x_data[(ind*60):((ind+1)*60), :] = x
                    y_data[(ind*60):((ind+1)*60)] = y
                    ind += 1
                if (ind/days) % 100 == 0:
                    print('get_{}data = {}[sec]'.format(
                        int(ind/days), time.time() - start1))

        print("get_train_data = {}[sec]".format(time.time() - start1))

        times = 10
        # for i in range(times):
        start1 = time.time()
        i = 1
        a = int(10 ** (i/3))
        param = 'letest'
        # print(param + " = {}".format(a))
        # n_estimators = 30, max_depth = 10, min_samples_split = 2, max_features = 30   # default
        # n_estimators = 46, max_depth = 46, min_samples_split = 21, max_features = 100 # renew
        # model = RandomForestClassifier(
        #     n_estimators=30, max_depth=10, min_samples_split=2, max_features=30, n_jobs=-1, random_state=0)
        model = RandomForestClassifier(n_jobs=-1, random_state=0)

        model.fit(x_data, y_data)
        joblib.dump(model, 'RFC.pkl')
        print('fit_model = {}[sec]'.format(time.time() - start1))

        start1 = time.time()

        train = np.zeros((days, 2))
        test = np.zeros((days, 2))

        ind = 0
        for files in train_file:
            for d in range(1, days):
                y = estimate(aiwolfpy.read_log(files), day=d)
                if y >= 0:
                    train[d, 0] += y
                    train[d, 1] += 1
            if train[1, 1] % 100 == 0:
                print('est_train_{}files = {}[sec]'.format(
                    int(train[1, 1]), time.time() - start1))

        ind = 0
        for files in test_file:
            for d in range(1, days):
                y = estimate(aiwolfpy.read_log(files), day=d)
                if y >= 0:
                    test[d, 0] += y
                    test[d, 1] += 1
            if test[1, 1] % 100 == 0:
                print('est_test_{}files = {}[sec]'.format(
                    int(test[1, 1]), time.time() - start1))

        start2 = time.time()
        for d in range(1, days):
            out = open('csv/' + param + '_default_day{}.csv'.format(d), 'a')
            outer = csv.writer(out)
            outer.writerow(
                [train_num, train[d, 0]/train[d, 1], test[d, 0] / test[d, 1]])
        print("write_csv = {}[sec]".format(time.time() - start2))

        print("single_estimate = {}[sec]".format(
            time.time() - start1))

print("all_work = {}[sec]".format(time.time() - start))
