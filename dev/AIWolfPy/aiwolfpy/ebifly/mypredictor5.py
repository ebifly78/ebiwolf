import numpy as np
import pandas
import math
import inspect
import random

from .mytesnsor60 import Tensor60

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class Talking():
    def __init__(self):
        self.subject = '99'
        self.target = '99'
        self.role = ''
        self.species = ''
        self.action = ''
        self.operator = ''
        self.talk_number = ''
        self.sentence = []

    def print_text(self):
        print(self.subject)
        print(self.target)
        print(self.role)
        print(self.species)
        print(self.action)
        print(self.operator)
        print(self.talk_number)
        print(self.sentence)


class Predictor_5(object):

    def get_index(self, text):
        if text in self.case5.action:
            return self.case5.action.index(text)
        elif text in self.case5.object:
            return self.case5.object.index(text)
        return 0

    def parse_agent(self, content):
        if type(content) is str:
            content = content.strip('Agent[0')
            content = content.strip(']')
        return int(content)

    def commit_action(self, content):
        if content.subject == 'ANY':
            return
        else:
            content.subject = self.parse_agent(content.subject)
        if content.target == 'ANY':
            return
        else:
            content.target = self.parse_agent(content.target)

        if content.action in self.case5.action:
            a = self.get_index(content.action)
            o = self.get_index(content.role)
            self.x_para[content.subject - 1, content.target - 1, a, o] += 1

        for sentenses in content.sentence:
            self.commit_action(sentenses)

    def split_bracket(self, content, op_subject):
        list1 = []
        kakko = 0
        start = 0
        end = 0
        memo = 0
        memo2 = 0
        for index, item in enumerate(content):
            if item.count('(') > 0 and kakko == 0:
                memo2 = item.count('(')
                start = index
                content[index] = item[item.count('('):]
            kakko += item.count('(')
            memo = kakko
            kakko -= item.count(')')
            if item.count(')') > 0 and kakko == 0:
                end = index + 1
                content[index] = item[:len(
                    item) - item.count(')') + memo - memo2]
            if index != 0 and kakko == 0:
                list1.append(self.talk_content(content[start:end], op_subject))

        return list1

    def talk_content(self, content, op_subject):
        text = Talking()
        index = 0
        if 'Agent' in content[index] or 'ANY' in content[index]:
            text.subject = content[index]
            index += 1
        else:
            text.subject = op_subject

        text.action = content[index]
        # text.target = self.#parse_agent(content[index+1])
        if text.action == 'ESTIMATE':
            text.target = content[index+1]
            text.role = content[index+2]
            self.commit_action(text)
        elif text.action == 'COMINGOUT':
            text.target = content[index+1]
            text.role = content[index+2]
            self.commit_action(text)
        elif text.action == 'DIVINATION':
            text.target = content[index+1]
            self.commit_action(text)
        elif text.action == 'VOTE':
            text.target = content[index+1]
            self.commit_action(text)
        elif text.action == 'ATTACK':
            text.target = content[index+1]
            self.commit_action(text)
        elif text.action == 'DIVINED':
            text.target = content[index+1]
            text.species = content[index+2]
            self.commit_action(text)
        elif text.action == 'VOTED':
            text.target = content[index+1]
            self.commit_action(text)
        elif text.action == 'ATTACKED':
            text.target = content[index+1]
            self.commit_action(text)
        elif text.action == 'AGREE':
            text.talk_number = content[index+1]
            self.commit_action(text)
        elif text.action == 'DISAGREE':
            text.talk_number = content[index+1]
            self.commit_action(text)
        elif text.action == 'Over':
            self.commit_action(text)
            return text
        elif text.action == 'Skip':
            self.commit_action(text)
            return text
        else:
            text.action = ''
            index -= 1
        index += 1

        text.operator = content[index]
        if text.operator == 'REQUEST':
            text.target = content[index+1]
            op_subject = text.target
            text.sentence += self.split_bracket(content[index+2:], op_subject)
        elif text.operator == 'INQUIRE':
            text.target = content[index+1]
            op_subject = text.target
            text.sentence += self.split_bracket(content[index+2:], op_subject)
        elif text.operator == 'BECAUSE':
            text.sentence += self.split_bracket(content[index+1:], op_subject)
            self.commit_action(text.sentence[1])
        elif text.operator == 'DAY':
            text.talk_number = content[index+1]
            op_target = text.target
            text.sentence += self.split_bracket(content[index+2:], op_subject)
        elif text.operator == 'NOT':
            text.sentence += self.split_bracket(content[index+1:], op_subject)
        elif text.operator == 'AND':
            text.sentence += self.split_bracket(content[index+1:], op_subject)
        elif text.operator == 'OR':
            text.sentence += self.split_bracket(content[index+1:], op_subject)
        elif text.operator == 'XOR':
            text.sentence += self.split_bracket(content[index+1:], op_subject)
        else:
            text.operator = ''
            index -= 1
        index += 1

        # text.print_text()

        return text

    def __init__(self):
        self.case5 = Tensor60()

        # num of param
        self.n_role = len(self.case5.role)
        self.n_action = len(self.case5.action)
        self.n_object = len(self.case5.object)
        self.n_para = self.n_role * self.n_role * self.n_action * self.n_object

        # param
        self.param = np.zeros(
            (self.n_role, self.n_role, self.n_action, self.n_object))

        # my param
        coef = []
        for i in range(self.n_para):
            coef.append(random.random())

        param_data = [0., -0.69802066, -1.36021126, 0., 0., 0.,
                      -1.8637926, -1.47987468, 0., -0.09469956, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.14655044, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.36303599, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.78108527, 0., 0., 0., 0., 0.,
                      -0.17248876, 0., 0., 0., 0., 0.,
                      0.33613405, 0., 0., 0., 0., 0.,
                      0., 2.12293952, -1.14729554, 0., 0., 0.,
                      0., -0.46095134, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.3221005, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.14448648, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.72987778, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., -1.40352913, 0.82238703, 0., 0., 0.,
                      0., -3.40137883, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.17597479, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -1.1393652, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.74268157, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., -0.65053349, -1.37818737, 0., 0., 0.,
                      0., -1.48148427, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.02488798, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.32669799, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.73303741, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 4.36941294, -3.55782947, 0., 0., 0.,
                      0., 3.11505426, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.41526164, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.20911929, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      2.42942915, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -1.8502463, 0., 0., -1.09029864, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -6.15793274, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.17065503, 0., 0., 0., 0., 0.,
                      -5.69990876, 0., 0., 0., 0., 0.,
                      0., 1.12266822, -1.38523034, 0., 0., 0.,
                      0., 5.69028096, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.52557344, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.30258527, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      2.26032142, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 3.93496018, -3.11234062, 0., 0., 0.,
                      0., 2.51925634, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.42161247, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.19300495, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      2.40881448, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., -0.09072989, -3.54189737, 0., 0., 0.,
                      0., -0.90305735, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.47578682, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.58566071, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.11424103, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0.12400827, -0.29759426, 0., 0., 0.,
                      0., -1.50628116, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.81949068, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.58201162, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.27237135, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -1.18355391, 0., 0., 1.37531827, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -4.8187325, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.1755162, 0., 0., 0., 0., 0.,
                      0.10984745, 0., 0., 0., 0., 0.,
                      0., -0.10685336, -3.27677969, 0., 0., 0.,
                      0., -0.60035208, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.48347357, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.60356506, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.09411652, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., -0.66452012, -1.36430302, 0., 0., 0.,
                      0., -1.94710414, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.01926367, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.33605031, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.69914952, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 2.00707262, -1.03726526, 0., 0., 0.,
                      0., -0.62071346, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.30759749, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.15184632, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.70742537, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., -1.39928487, 0.81091306, 0., 0., 0.,
                      0., -2.5512744, 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.16223461, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -1.13844942, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.72354806, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -1.83837465, 0., 0., -0.07326819, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.31274008, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.15912047, 0., 0., 0., 0., 0.,
                      0.32405942, 0., 0., 0., 0., 0.]
        # param_data = coef

        num = 0
        for l in range(self.n_object):
            for k in range(self.n_action):
                for j in range(self.n_role):
                    for i in range(self.n_role):
                        self.param[i, j, k, l] = param_data[num]
                        num += 1

        self.x_para = np.zeros(
            (5, 5, self.n_action, self.n_object), dtype='float32')

    def initialize(self, base_info, game_setting):
        self.game_setting = game_setting
        self.base_info = base_info

        self.watshi_xxx = np.ones((60, self.n_role))

        # use for Machine Learning
        if inspect.currentframe().f_back.f_code.co_filename == 'agent_ebifly.py' or inspect.currentframe().f_back.f_code.co_filename == 'myagent.py':
            xv = self.case5.get_case60_df(
            )["agent_"+str(self.base_info['agentIdx'])].values
        else:
            xv = self.case5.get_case60_df(
            )["agent_"+str(self.base_info['agent'])].values

        for i in range(self.n_role):
            self.watshi_xxx[xv != i, i] = 0.0

        # initialize x_param
        self.x_para = np.zeros(
            (5, 5, self.n_action, self.n_object), dtype='float32')

    def update(self, gamedf):
        self.update_features(gamedf)
        self.update_df()
        self.update_pred()

    def update_features(self, gamedf):
        # read log
        for i in range(gamedf.shape[0]):
            # vote
            if gamedf.type[i] == 'vote' and gamedf.turn[i] == 0:
                a = self.get_index('vote')
                self.x_para[gamedf.idx[i] - 1, gamedf.agent[i] - 1, a, 0] += 1
            # execute
            elif gamedf.type[i] == 'execute':
                a = self.get_index('execute')
                self.x_para[gamedf.agent[i] - 1,
                            gamedf.agent[i] - 1, a, 0] = 1
            # attacked
            elif gamedf.type[i] == 'dead':
                a = self.get_index('dead')
                self.x_para[gamedf.agent[i] - 1,
                            gamedf.agent[i] - 1, a, 0] = 1
            # talk
            elif gamedf.type[i] == 'talk':
                content = gamedf.text[i].split()
                content = self.talk_content(content, gamedf.agent[i])

    def update_df(self):
        # update 60 dataframe
        self.names = []
        for i in range(self.n_role):
            for j in range(self.n_role):
                for k in range(self.n_action):
                    for l in range(self.n_object):
                        s = self.case5.role[i]
                        s += '_' + self.case5.action[k]
                        s += '_' + self.case5.role[j]
                        s += '_' + self.case5.object[l]
                        self.names.append(s)
        self.df_pred = self.case5.apply_tensor_df(self.x_para, self.names)

    def update_pred(self):
        l_para = self.param.reshape(self.n_para, 1)
        self.df_pred["pred"] = np.matmul(
            self.df_pred.values, l_para.reshape(-1, 1))
        self.p_60 = self.df_pred["pred"]

    def possible_60(self, myRole, statusMap):
        p = self.p_60 * self.watshi_xxx[:, myRole]
        for ind in range(60):
            i = np.where(self.case5.case60[ind, :] == 1)[0][0] + 1
            if statusMap[str(i)] != 'ALIVE':
                p[ind, 1] = 0.0
        return p
