from .tensor60 import Tensor60
import numpy as np
import math
import inspect
import random


class Talking():
    def __init__(self):
        self.subject = '99'
        self.target = '99'
        self.role = ''
        self.species = ''
        self.verb = ''
        self.operator = ''
        self.talk_number = ''
        self.sentence = []

    def print_text(self):
        print(self.subject)
        print(self.target)
        print(self.role)
        print(self.species)
        print(self.verb)
        print(self.operator)
        print(self.talk_number)
        print(self.sentence)


class Predictor_5(object):

    def parse_agent(self, content):
        if type(content) is str:
            content = content.strip('Agent[0')
            content = content.strip(']')
        return int(content)

    def commit_verb(self, content):
        # content.print_text()
        if content.subject == 'ANY':
            return
        else:
            content.subject = self.parse_agent(content.subject)
        if content.subject == 'ANY':
            return
        else:
            content.target = self.parse_agent(content.target)
        # content.print_text()
        if content.verb == 'ESTIMATE':
            if content.role == 'SEER':
                self.x_3d[content.subject - 1, content.target - 1, 5:9] = 0
                self.x_3d[content.subject - 1, content.target - 1, 5] = 1
            elif content.role == 'VILLAGER':
                self.x_3d[content.subject - 1, content.target - 1, 5:9] = 0
                self.x_3d[content.subject - 1, content.target - 1, 6] = 1
            elif content.role == 'POSSESSED':
                self.x_3d[content.subject - 1, content.target - 1, 5:9] = 0
                self.x_3d[content.subject - 1, content.target - 1, 7] = 1
            elif content.role == 'WEREWOLF':
                self.x_3d[content.subject - 1, content.target - 1, 5:9] = 0
                self.x_3d[content.subject - 1, content.target - 1, 8] = 1
        elif content.verb == 'COMINGOUT':
            if content.role == 'SEER':
                self.x_2d[content.target - 1, 2:6] = 0
                self.x_2d[content.target - 1, 2] = 1
            elif content.role == 'VILLAGER':
                self.x_2d[content.target - 1, 2:6] = 0
                self.x_2d[content.target - 1, 3] = 1
            elif content.role == 'POSSESSED':
                self.x_2d[content.target - 1, 2:6] = 0
                self.x_2d[content.target - 1, 4] = 1
            elif content.role == 'WEREWOLF':
                self.x_2d[content.target - 1, 2:6] = 0
                self.x_2d[content.target - 1, 5] = 1
        elif content.verb == 'DIVINATION':
            pass
        elif content.verb == 'VOTE':
            # self.x_3d[content.subject - 1, content.target - 1, 9] = 1
            pass
        elif content.verb == 'ATTACK':
            pass
        elif content.verb == 'DIVINED':
            self.x_2d[content.subject - 1, 2:6] = 0
            self.x_2d[content.subject - 1, 2] = 1
            if content.role == 'HUMAN':
                self.x_3d[content.subject - 1, content.target - 1, 1] = 1
                self.x_3d[content.subject - 1, content.target - 1, 2] = 0
            elif content.role == 'WEREWOLF':
                self.x_3d[content.subject - 1, content.target - 1, 1] = 0
                self.x_3d[content.subject - 1, content.target - 1, 2] = 1
        elif content.verb == 'VOTED':
            pass
        elif content.verb == 'ATTACKED':
            pass
        elif content.verb == 'AGREE':
            self.x_3d[content.subject - 1, content.target - 1, 3] = 1
        elif content.verb == 'DISAGREE':
            self.x_3d[content.subject - 1, content.target - 1, 4] = 1
        elif content.verb == 'Over':
            pass
        elif content.verb == 'Skip':
            pass

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

        text.verb = content[index]
        # text.target = self.#parse_agent(content[index+1])
        if text.verb == 'ESTIMATE':
            text.target = content[index+1]
            text.role = content[index+2]
            self.commit_verb(text)
        elif text.verb == 'COMINGOUT':
            text.target = content[index+1]
            text.role = content[index+2]
            self.commit_verb(text)
        elif text.verb == 'DIVINATION':
            text.target = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'VOTE':
            text.target = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'ATTACK':
            text.target = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'DIVINED':
            text.target = content[index+1]
            text.species = content[index+2]
            self.commit_verb(text)
        elif text.verb == 'VOTED':
            text.target = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'ATTACKED':
            text.target = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'AGREE':
            text.talk_number = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'DISAGREE':
            text.talk_number = content[index+1]
            self.commit_verb(text)
        elif text.verb == 'Over':
            self.commit_verb(text)
            return text
        elif text.verb == 'Skip':
            self.commit_verb(text)
            return text
        else:
            text.verb = ''
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
        self.n_para_3d = 9
        self.n_para_2d = 6

        # param_linear
        self.para_3d = np.zeros((4, 4, self.n_para_3d))
        self.para_2d = np.zeros((4, self.n_para_2d))

        # my param
        coef = []
        for i in range(168):
            coef.append(random.random())
        param_data = [-0.00002605, 0., 0., 0., 0., 0.,
                      0., 0., 0., -0.00000409, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.00001614, 0., 0., 0., 0., 0.,
                      0., 0., 0., -0.00000404, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.00010474, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.00009952, 0., 0., 0., 0., 0.,
                      0., 0., 0., -0.00004602, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.00002241, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0.00000861, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0.00004657, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.0000456, 0., 0., 0., 0., 0.,
                      0., 0., 0., -0.00002542, 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      -0.00002329, 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0.,
                      0.00024544, -0.0000717, 0., 0., 0., 0.,
                      0.00015273, -0.00080324, 0., 0., 0., 0.,
                      0.00023144, -0.00001672, 0., 0., 0., 0.,
                      0.00020006, -0.00009929, 0., 0., 0., 0.]
        #param_data = coef

        num = 0
        amp = 10000
        for k in range(9):
            for j in range(4):
                for i in range(4):
                    self.para_3d[i, j, k] = param_data[num] * amp
                    num += 1

        for j in range(6):
            for i in range(4):
                self.para_2d[i, j] = param_data[num] * amp
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
                content = gamedf.text[i].split()
                content = self.talk_content(content, gamedf.agent[i])

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
