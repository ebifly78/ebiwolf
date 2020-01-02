from __future__ import print_function, division

import sys
import glob
import time

import numpy as np
import pandas as pd
import itertools

import random
import csv

import aiwolfpy
import aiwolfpy.ebifly

from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)


predictor = aiwolfpy.ebifly.Predictor_5()
days = 3


def xy_init(expan, train):
    n_feat = 60 * (days-1) * len(train) * expan
    x = np.zeros((n_feat, predictor.n_para))
    y = np.zeros(n_feat)

    return x, y


def split_file(files, limit):
    if limit > len(files):
        print("limit too big")
        sys.exit()

    files = files[:limit]
    split_id = int(len(files) * 0.5)
    train = files[:split_id]
    test = files[split_id:]

    return train, test


def gamedf_expansion(df):
    pat = random.randint(0, 119)
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

    return df


def gamedf_filter(df, day, myrole='VILLAGER'):
    agent = 0
    roleMap = {'WEREWOLF': '0', 'POSSESSED': '0',
               'SEER': '0', 'VILLAGER': '0', 'agent': '0'}
    for i in range(1, 6):
        role = df["text"][i - 1].split()[2]
        roleMap[role] = str(df['agent'][i-1])
        if role == "WEREWOLF":
            werewolf = df['agent'][i-1]
        elif role == "POSSESSED":
            possessed = df['agent'][i-1]
        elif role == "SEER":
            seer = df['agent'][i-1]
        elif role == "VILLAGER":
            villager = df['agent'][i-1]
        if role == myrole:
            agent = df['agent'][i-1]
            roleMap['agent'] = str(df['agent'][i-1])

    y = np.zeros(60)
    for i in range(60):
        if predictor.case5.case60_df["agent_"+str(werewolf)][i] == 1:
            if predictor.case5.case60_df["agent_"+str(seer)][i] == 3:
                if predictor.case5.case60_df["agent_"+str(possessed)][i] == 2:
                    y[i] = 1

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

    df = df[(df["day"] < day) | ((df["day"] == day) & (df["type"] == 'talk'))]

    statusMap = {'1': 'ALIVE', '2': 'ALIVE',
                 '3': 'ALIVE', '4': 'ALIVE', '5': 'ALIVE'}

    for i in execute:
        statusMap[str(i)] = 'DEAD'
    for i in dead:
        statusMap[str(i)] = 'DEAD'

    base_info = {"agentIdx": agent, "roleMap": {
        str(agent): role}, "statusMap": statusMap}
    gamedf = df.reset_index()

    return base_info, gamedf, y, roleMap


def get_data(df, day, myrole='VILLAGER'):
    base_info, gamedf, y, roleMap = gamedf_filter(df, day, myrole)
    predictor.initialize(base_info, {})
    predictor.update_features(gamedf)
    predictor.update_df()

    return predictor.df_pred, y


def estimate(df, day, myrole='VILLAGER'):
    base_info, gamedf, y, roleMap = gamedf_filter(df, day, myrole)

    if base_info['statusMap'][roleMap['WEREWOLF']] == 'DEAD':
        return -1
    if base_info['statusMap'][roleMap['agent']] == 'DEAD':
        return -1

    predictor.initialize(base_info, {})
    predictor.update(gamedf)

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

    werewolf = int(roleMap['WEREWOLF'])
    if est == werewolf:
        return 1
    else:
        return 0


def model_init(x, y, param, val):
    # RandomForestClassifier
    # h_para = [n_estimators, max_depth, min_samples_split, max_features]
    # h_para = [10, None, 2, 'auto'] # default
    # h_para = [46, 46, 21, 100] # renew
    # if param > 0:
    #     h_para[param-1] = val
    # model = RandomForestClassifier(
    #     n_estimators=h_para[0], max_depth=h_para[1],
    #     min_samples_split=h_para[2], max_features=h_para[3],
    #     n_jobs=-1)

    # model = RandomForestClassifier()
    # model = LogisticRegression()
    # model = LinearSVC()
    # model = DecisionTreeClassifier()
    model = MLPClassifier()

    model.fit(x, y)
    joblib.dump(model, 'MLP.pkl')


def data_active(file_list, expan):
    x, y = xy_init(expan, file_list)
    ind = 0
    for files in file_list:
        gamedf = aiwolfpy.read_log(files)
        for i in range(expan):
            df = gamedf_expansion(gamedf)
            for d in range(1, days):
                x[(ind*60):((ind+1)*60), :], y[(ind*60)
                   :((ind+1)*60)] = get_data(df, d)
                ind += 1
                n_data = int(ind / (days-1))
            if n_data % 100 == 0:
                print('get_{}data = {}[sec]'.format(
                    n_data, time.time() - start1))

    return x, y


def est_active(file_list):
    est = np.zeros((days, 2))
    ind = 0
    for files in file_list:
        for d in range(1, days):
            y = estimate(aiwolfpy.read_log(files), day=d)
            if y >= 0:
                est[d, 0] += y
                est[d, 1] += 1
        n_est = int(est[1, 1])
        if n_est % 100 == 0:
            print('est_{}files = {}[sec]'.format(
                n_est, time.time() - start1))

    return est


start0 = time.time()
start1 = time.time()

file_list = []
logdir = '../../Server/AIWolf-ver0.5.6/log/'
# folder_list = [logdir + 'file1/']
folder_list = glob.glob(logdir + 'gat2017log05/*')

for folder in folder_list:
    file_list += glob.glob(folder + '/*')

np.random.shuffle(file_list)

print("get_files = {}[sec]".format(time.time() - start1))

limit_list = []
for i in range(0, 4):
    for j in range(1, 10):
        limit_list.append(j * 10**i)
limit_list.append(10000)

expan_list = [1]
# param_RFC : 0=None, 1=n_estimators, 2=max_depth, 3=min_samples_split, 4=max_features
param_list = [0]
val_list = [2]
for limit in limit_list:
    for expan in expan_list:
        start1 = time.time()

        expan = 1
        if limit == 0:
            limit = len(file_list)
        else:
            limit = limit * 2

        train, test = split_file(file_list, limit)
        x, y = data_active(train, expan)

        print("get_train_data = {}[sec]".format(time.time() - start1))

        for param in param_list:
            for val in val_list:
                start1 = time.time()
                start3 = time.time()

                model_init(x, y, param, val)

                print('fit_model = {}[sec]'.format(time.time() - start1))
                start1 = time.time()

                print('estimate_train')
                est_train = est_active(train)
                start1 = time.time()
                print('estimate_test')
                est_test = est_active(test)

                print('estimate = {}[sec]'.format(time.time() - start1))
                start1 = time.time()

                for d in range(1, days):
                    out = open(
                        'csv/MLP_expan{}_param{}_val{}_day{}.csv'.format(expan, param, val, d), 'a')
                    outer = csv.writer(out)

                    recall_train = est_train[d, 0] / est_train[d, 1]
                    recall_test = est_test[d, 0] / est_test[d, 1]

                    outer.writerow([len(train), recall_train, recall_test])
                    out.close()
                print("write_csv = {}[sec]".format(time.time() - start1))

                print("single_estimate_{}files = {}[sec]\n".format(
                    len(train), time.time() - start3))

print("all_work = {}[sec]".format(time.time() - start0))
