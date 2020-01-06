from __future__ import print_function, division

import sys
import glob
import time
import random
import csv

import numpy as np
import pandas as pd
import itertools

import aiwolfpy
import aiwolfpy.ebifly

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB


np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)


predictor = aiwolfpy.ebifly.Predictor_5()
days = 3
log_list = []


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
            if predictor.case5.case60_df["agent_"+str(possessed)][i] == 2:
                if predictor.case5.case60_df["agent_"+str(seer)][i] == 3:
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
        str(agent): myrole}, "statusMap": statusMap}
    gamedf = df.reset_index()

    return base_info, gamedf, y, roleMap


def get_data(df, day, myrole='VILLAGER'):
    base_info, gamedf, y, roleMap = gamedf_filter(df, day, myrole)
    predictor.initialize(base_info, {})
    predictor.update_features(gamedf)
    predictor.update_df()

    return predictor.df_pred, y


def estimate(df, day, model_name, myrole='VILLAGER'):
    base_info, gamedf, y, roleMap = gamedf_filter(df, day, myrole)

    if base_info['statusMap'][roleMap['WEREWOLF']] == 'DEAD':
        return -1
    if base_info['statusMap'][roleMap['agent']] == 'DEAD':
        return -1

    predictor.initialize(base_info, {})
    predictor.update(gamedf, model_name)

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


def model_init(x, y, param, val, name):
    # RandomForestClassifier
    # h_para = [n_estimators, max_depth, min_samples_split, max_features]
    h_para = [10, None, 2, 'auto']  # default
    # h_para = [46, 46, 21, 100]  # renew
    if param > 0:
        h_para[param-1] = val
    elif param == 0:
        h_para = [random.randint(2, 1000) for i in range(4)]

    if name == 'RFCr':
        model = RandomForestClassifier(
            n_estimators=h_para[0], max_depth=h_para[1],
            min_samples_split=h_para[2], max_features=h_para[3],
            n_jobs=-1)
    elif name == 'RFC':
        model = RandomForestClassifier()
    elif name == 'LR':
        model = LogisticRegression()
    elif name == 'SVC':
        model = LinearSVC()
    elif name == 'DTC':
        model = DecisionTreeClassifier()
    elif name == 'MLP':
        model = MLPClassifier()
    elif name == 'BNB':
        model = BernoulliNB()

    model.fit(x, y)
    joblib.dump(model, name + '.pkl')

    return h_para


def data_active(file_list, expan):
    x, y = xy_init(expan, file_list)
    ind = 0
    for files in file_list:
        gamedf = aiwolfpy.read_log(files)
        for i in range(expan):
            df = gamedf_expansion(gamedf)
            for d in range(1, days):
                x[(ind*60):((ind+1)*60), :], y[(ind*60)                                               :((ind+1)*60)] = get_data(df, d)
                ind += 1
                n_data = int(ind / (days-1))
            if n_data % 100 == 0:
                print('get_{}data = {:.2f}[sec]'.format(
                    n_data, time.time() - start1))

    return x, y


def est_active(file_list, name):
    est = np.zeros((days, 2))
    ind = 0
    for files in file_list:
        for d in range(1, days):
            y = estimate(aiwolfpy.read_log(files), d, name)
            if y >= 0:
                est[d, 0] += y
                est[d, 1] += 1
        n_est = int(est[1, 1])
        if n_est % 100 == 0:
            print('est_{}files = {:.2f}[sec]'.format(
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

print("get_files = {:.2f}[sec]".format(time.time() - start1))


model_list = ['RFCr']

limit_list = []
# for i in range(2, 6):
#     for j in range(1, 10):
#         if j * 10**i <= 5000:
#             limit_list.append(j * 10**i)
for i in range(20):
    limit_list.append(2000)

expan_list = []
expan_list.append(1)

# param_RFC : 0=Random, 1=n_estimators, 2=max_depth, 3=min_samples_split, 4=max_features
param_list = []
param_list.append(0)

val_list = []
# for i in range(0, 6):
#     if j * 10**i >= 2 and j * 10**i <= 1000:
#         val_list.append(j * 10**i)
val_list.append(2)

for model_name in model_list:
    start4 = time.time()
    for limit in limit_list:
        for expan in expan_list:
            start1 = time.time()

            expan = 1
            if limit == 0:
                limit = len(file_list)
            elif limit >= len(file_list)/2:
                print('file_limit_error = {}'.format(limit))
                break
            else:
                limit = limit * 2

            train, test = split_file(file_list, limit)
            x, y = data_active(train, expan)

            print("get_train_data = {:.2f}[sec]".format(time.time() - start1))

            for param in param_list:
                for val in val_list:
                    start1 = time.time()
                    start3 = time.time()

                    h_para = model_init(x, y, param, val, model_name)

                    print('fit_model = {:.2f}[sec]'.format(
                        time.time() - start1))
                    start1 = time.time()
                    start5 = time.time()
                    print('estimate_train')
                    est_train = est_active(train, model_name)
                    start1 = time.time()
                    print('estimate_test')
                    est_test = est_active(test, model_name)

                    print('estimate = {:.2f}[sec]'.format(
                        time.time() - start5))
                    start1 = time.time()

                    for d in range(1, days):
                        # out = open(
                        #     'csv/' + model_name + '_expan{}_param{}_val{}_day{}.csv'.format(expan, param, val, d), 'a')
                        out = open(
                            'csv/' + model_name + '_expan{}_param{}_day{}.csv'.format(expan, param, d), 'a')
                        outer = csv.writer(out)

                        recall_train = est_train[d, 0] / est_train[d, 1]
                        recall_test = est_test[d, 0] / est_test[d, 1]

                        # outer.writerow([len(train), recall_train, recall_test])
                        outer.writerow(
                            h_para + [recall_train, recall_test])  # param0
                        out.close()
                    print("write_csv = {:.2f}[sec]".format(
                        time.time() - start1))

                    print("single_estimate_{}files = {:.2f}[sec]\n".format(
                        len(train), time.time() - start3))
                    log_list.append("{}_all_estimate = {:.2f}[sec]".format(
                        h_para, time.time() - start3))

    print(model_name +
          "_all_estimate = {:.2f}[sec]\n".format(time.time() - start4))
    log_list.append(
        model_name + "_all_estimate = {:.2f}[sec]".format(time.time() - start4))

for log in log_list:
    print(log)

print("all_work = {:.2f}[sec]\n".format(time.time() - start0))
