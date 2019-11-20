from .tensor60 import Tensor60
import numpy as np
import math
import inspect
import random


class TalkContent():

    def split_bracket(self, content, num):
        list1 = []
        kakko = 0
        for index, item in enumerate(content[index:]):
            if item == '(' and kakko == 0:
                start = index
            if item == ')' and kakko == 0:
                end = index + 1
            list1.append(content[start:end])

            kakko += item.count('(')
            kakko -= item.count(')')
        return list1

    def __init__(self, content):
        self.subject = ''
        self.target = ''
        self.role = ''
        self.species = ''
        self.verb = ''
        self.operator = ''
        self.talk_number = ''
        self.sentence = []

        index = 0

        if 'Agent' in content[index] or 'ANY' in content[index]:
            self.subject = content[index]
            index += 1
        else:
            self.subject = 'UNSPEC'

        self.verb = content[index]
        index += 1
        if content[index] == 'ESTIMATE':
            self.target = content[index+1]
            self.role = content[index+2]
        elif content[index] == 'COMINGOUT':
            self.target = content[index+1]
            self.role = content[index+2]
        elif content[index] == 'DIVINATION':
            self.target = content[index+1]
        elif content[index] == 'VOTE':
            self.target = content[index+1]
        elif content[index] == 'ATTACK':
            self.target = content[index+1]
        elif content[index] == 'DIVINED':
            self.target = content[index+1]
            self.species = content[index+2]
        elif content[index] == 'VOTED':
            self.target = content[index+1]
        elif content[index] == 'ATTACKED':
            self.target = content[index+1]
        elif content[index] == 'AGREE':
            self.talk_number = content[index+1]
        elif content[index] == 'DISAGREE':
            self.talk_number = content[index+1]
        elif content[index] == 'Over':
            return None
        elif content[index] == 'Skip':
            return None
        else:
            self.verb = ''
            index -= 1

        self.operator = content[index]
        index += 1
        if content[index] == 'REQUEST':
            self.target = content[index+1]
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+2:], 1)))
        elif content[index] == 'INQUIRE':
            self.target = content[index+1]
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+2:], 1)))
        elif content[index] == 'BECAUSE':
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+1:], 2)))
        elif content[index] == 'DAY':
            self.talk_number = content[index+1]
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+2:], 1)))
        elif content[index] == 'NOT':
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+1:], 1)))
        elif content[index] == 'AND':
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+1:], 0)))
        elif content[index] == 'OR':
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+1:], 0)))
        elif content[index] == 'XOR':
            self.sentence.append(TalkContent(
                self.split_bracket(content[index+1:], 2)))
        else:
            self.operator = ''
            index -= 1
        print((self.sentence).verb)


class Predictor_5(object):

    def __init__(self):
        self.case5 = Tensor60()

        # num of param
        self.n_para_3d = 9
        self.n_para_2d = 6

        # param_linear
        self.para_3d = np.zeros((4, 4, self.n_para_3d))
        self.para_2d = np.zeros((4, self.n_para_2d))

        # my param
        coef = []
        for i in range(168):
            coef.append(random.random())
        param_data = [-0.0000172, 0., 0., -0.00004035, 0., 0.,
                      -0.00002156, 0., 0., -0.00000157, 0., 0.,
                      -0.00001756, 0., 0., 0., 0., 0.,
                      -0.00006442, 0., 0., -0.00005788, 0., 0.,
                      0.00000777, 0., 0., -0.00003353, 0., 0.,
                      0., 0., 0., 0.00003804, 0., 0.,
                      -0.00004614, 0., 0., -0.00004655, 0., 0.,
                      -0.00003882, 0., 0., 0., 0., 0.,
                      0.00029176, -0.00015814, -0.00003443, 0., 0., 0.,
                      0.00024926, -0.00089807, 0.00003039, 0., 0., 0.,
                      0.00018271, -0.00014811, 0.00007247, 0., 0., 0.,
                      0.00026703, -0.00015246, -0.0000621, 0., 0., 0.]
        # param_data = coef

        num = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    self.para_3d[i, j, k] = param_data[num]
                    num += 1

        for i in range(4):
            for j in range(6):
                self.para_2d[i, j] = param_data[num]
                num += 1

        self.x_3d = np.zeros((5, 5, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((5, self.n_para_2d), dtype='float32')

    def initialize(self, base_info, game_setting):
        # game_setting
        self.game_setting = game_setting

        # base_info
        self.base_info = base_info

        # initialize watashi_xxx
        self.watshi_xxx = np.ones((60, 4))

        # use for Machine Learning
        if inspect.currentframe().f_back.f_code.co_filename == 'agent_ebifly.py':
            xv = self.case5.get_case60_df(
            )["agent_"+str(self.base_info['agentIdx'])].values
        else:
            xv = self.case5.get_case60_df(
            )["agent_"+str(self.base_info['agent'])].values

        self.watshi_xxx[xv != 0, 0] = 0.0
        self.watshi_xxx[xv != 1, 1] = 0.0
        self.watshi_xxx[xv != 2, 2] = 0.0
        self.watshi_xxx[xv != 3, 3] = 0.0

        # initialize x_3d, x_2d
        self.x_3d = np.zeros((5, 5, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((5, self.n_para_2d), dtype='float32')

        """
        X_3d
        [i, j, 0] : agent i voted agent j (not in talk, action)
        [i, j, 1] : agent i divined agent j HUMAN
        [i, j, 2] : agent i divined agent j WEREWOLF
        [i, j, 3] : agent i agree agent j
        [i, j, 4] : agent i disagree agent j
        [i, j, 5] : agent i estimate agent j SEER
        [i, j, 6] : agent i estimate agent j VILLAGER
        [i, j, 7] : agent i estimate agent j POSSESSED
        [i, j, 8] : agent i estimate agent j WEREWOLF

        X_2d
        [i, 0] : agent i is executed
        [i, 1] : agent i is attacked
        [i, 2] : agent i comingout himself/herself SEER
        [i, 3] : agent i comingout himself/herself VILLAGER
        [i, 4] : agent i comingout himself/herself POSSESSED
        [i, 5] : agent i comingout himself/herself WEREWOLF
        """

    def update(self, gamedf):
        self.update_features(gamedf)
        self.update_df()
        self.update_pred()
        # self.mod_pred()

    def update_features(self, gamedf):
        # read log
        for i in range(gamedf.shape[0]):
            # vote
            if gamedf.type[i] == 'vote' and gamedf.turn[i] == 0:
                self.x_3d[gamedf.idx[i] - 1, gamedf.agent[i] - 1, 0] += 1
            # execute
            elif gamedf.type[i] == 'execute':
                self.x_2d[gamedf.agent[i] - 1, 0] = 1
            # attacked
            elif gamedf.type[i] == 'dead':
                self.x_2d[gamedf.agent[i] - 1, 1] = 1
            # talk
            elif gamedf.type[i] == 'talk':
                content = TalkContent(gamedf.text[i])
                print(content)
                """
                if content.subject == 'UNSPEC':
                    content.subject = gamedf.agent[i]

                if content.verb == 'COMINGOUT':
                    if content.role == 'SEER':
                        self.x_2d[content.subject - 1, 2:5] = 0
                        self.x_2d[content.subject - 1, 2] = 1
                    elif content.role == 'VILLAGER':
                        self.x_2d[content.subject - 1, 2:5] = 0
                        self.x_2d[content.subject - 1, 3] = 1
                    elif content.role == 'POSSESSED':
                        self.x_2d[content.subject - 1, 2:5] = 0
                        self.x_2d[content.subject - 1, 4] = 1
                    elif content.role == 'WEREWOLF':
                        self.x_2d[content.subject - 1, 2:5] = 0
                        self.x_2d[content.subject - 1, 5] = 1
                """
                """
                content = gamedf.text[i].split()
                # comingout
                if content[0] == 'COMINGOUT':
                    # self
                    if int(content[1][6:8]) == gamedf.agent[i]:
                        if content[2] == 'SEER':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 2] = 1
                        elif content[2] == 'VILLAGER':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 3] = 1
                        elif content[2] == 'POSSESSED':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 4] = 1
                        elif content[2] == 'WEREWOLF':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 5] = 1
                # divined
                elif content[0] == 'DIVINED':
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                    self.x_2d[gamedf.agent[i] - 1, 2] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 1] = 1
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 2] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 2] = 1
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 1] = 0

                elif content[0] == 'ESTIMATE':
                    # result
                    if content[2] == 'SEER':
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 1] = 1
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 2] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 2] = 1
                        self.x_3d[gamedf.agent[i] - 1,
                                  int(content[1][6:8])-1, 1] = 0
                # because
                elif content[0] == 'BECAUSE':
                    # remove reason
                    kakko = 0
                    first = 0
                    for index, item in enumerate(content):
                        kakko += item.count('(')
                        kakko -= item.count(')')
                        if kakko == 0 and index != 0:
                            first = index + 1
                            # print("because kakko detected")
                            # print(content)
                            break
                    content = content[first:]

                    # remove AND,OR
                    if content[0] == 'OR' or content[0] == 'XOR'or content[0] == 'AND' or content[0] == 'NOT':
                        kakko = 0
                        first = 0
                        for index, item in enumerate(content):
                            kakko += item.count('(')
                            kakko -= item.count(')')
                            if kakko == 1 and index != 0:
                                first = index + 1
                                break

                    content = self.remove_kakko(content)
                    # print(content)
                    for item in content:
                        pass
                        # if item ==
                """
                pass

    def update_df(self):
        # update 60 dataframe
        self.df_pred = self.case5.apply_tensor_df(self.x_3d, self.x_2d,
                                                  names_3d=["VOTE", "DIV_HM", "DIV_WW", "agree", "disagree",
                                                            "EST_SEER", "EST_VILLAGER", "EST_POSSESSED", "EST_WEREWOLF"],
                                                  names_2d=['executed', 'attacked', 'CO_SEER', 'CO_VILLAGER', 'CO_POSSESSED', 'CO_WEREWOLF'])

    def update_pred(self):
        # predict
        # Linear
        l_para = np.append(self.para_3d.reshape(
            (4*4*self.n_para_3d, 1)), self.para_2d.reshape((4*self.n_para_2d, 1)))
        self.df_pred["pred"] = np.matmul(
            self.df_pred.values, l_para.reshape(-1, 1))
        self.df_pred["pred"] = np.exp(-np.log(10)*self.df_pred["pred"])

        # average
        self.p_60 = self.df_pred["pred"] / self.df_pred["pred"].sum()

    def ret_pred(self):
        p = self.p_60
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes=[0, 0]).transpose()

    def ret_pred_wx(self, r):
        p = self.p_60 * self.watshi_xxx[:, r]
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes=[0, 0]).transpose()
