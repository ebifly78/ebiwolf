#!/usr/bin/env python
from __future__ import print_function, division

# this is main script
import numpy as np
import aiwolfpy
import aiwolfpy.contentbuilder as cb

# sample
import aiwolfpy.ebifly as ebi

myname = 'ebifly'


class PythonPlayer(object):

    def __init__(self, agent_name):
        self.myname = agent_name
        self.predicter_5 = ebi.Predictor_5()
        self.myRole = 0

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting

        self.predicter_5.initialize(base_info, game_setting)

        self.divined_list = []
        self.comingout = ''
        self.myresult = ''
        self.not_reported = False
        self.vote_declare = 0

    # def get_p60(self, base_info):
    #     if base_info['myRole'] == 'VILLAGER':
    #         self.myRole = 0
    #     elif base_info['myRole'] == 'WEREWOLF':
    #         self.myRole = 1
    #     elif base_info['myRole'] == 'POSSESSED':
    #         self.myRole = 2
    #     elif base_info['myRole'] == 'SEER':
    #         self.myRole = 3

    #     p_60 = self.predicter_5.possible_60(
    #         self.myRole, base_info['statusMap'])
    #     p_60 = p_60 / p_60.sum()

    #     return p_60

    def role_p(self, p_60):
        p_5 = np.zeros((5, 4), dtype='float32')
        for i in range(60):
            for j in range(5):
                p_5[j, self.predicter_5.case5.case60[i, j]] += p_60[i]

        return p_5

    def highest(self, role):
        p = -1
        idx = 1
        p_mat = self.role_p(self.predicter_5.p_60)
        for i in range(5):
            if p_mat[i, role] > p:
                p = p_mat[i, role]
                idx = i
        return idx + 1

    def lowest(self, role):
        p = 1
        idx = 1
        p_mat = self.role_p(self.predicter_5.p_60)
        for i in range(5):
            if p_mat[i, 1] < p:
                p = p_mat[i, 1]
                idx = i
        return idx + 1

    def update(self, base_info, diff_data, request):
        self.base_info = base_info

        # result
        if request == 'DAILY_INITIALIZE':
            for i in range(diff_data.shape[0]):
                # IDENTIFY
                if diff_data['type'][i] == 'identify':
                    self.not_reported = True
                    self.myresult = diff_data['text'][i]

                # DIVINE
                if diff_data['type'][i] == 'divine':
                    self.not_reported = True
                    self.DivWolf = 0
                    self.myresult = diff_data['text'][i]
                    res = self.myresult.split()
                    if res[2] == 'WEREWOLF':
                        res = res[1]
                        res = res.strip('Agent[0')
                        res = res.strip(']')
                        self.DivWolf = int(res)

            # POSSESSED
            if self.base_info['myRole'] == 'POSSESSED':
                self.not_reported = True

        # UPDATE
        self.predicter_5.update(diff_data)

    def dayStart(self):
        self.vote_declare = 0
        self.talk_turn = 0
        return None

    def talk(self):
        self.talk_turn += 1

        # 1.comingout anyway
        if self.base_info['myRole'] == 'SEER' and self.comingout == '':
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)

        # 2.report
        if self.base_info['myRole'] == 'SEER' and self.not_reported:
            self.not_reported = False
            return self.myresult
        elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
            self.not_reported = False
            # FAKE DIVINE
            # highest prob ww in alive agents
            idx = self.highest(1)
            self.myresult = 'DIVINED Agent[' + \
                "{0:02d}".format(idx) + '] ' + 'HUMAN'
            return self.myresult

        # 3.declare vote if not yet
        if self.vote_declare != self.vote():
            self.vote_declare = self.vote()
            return cb.vote(self.vote_declare)

        # 4. skip
        if self.talk_turn <= 10:
            return cb.skip()

        return cb.over()

    def vote(self):
        if self.base_info['myRole'] == "WEREWOLF":
            idx = self.highest(3)
        elif self.base_info['myRole'] == "POSSESSED":
            idx = self.highest(3)
        elif self.base_info['myRole'] == "SEER":
            idx = self.highest(1)
            if self.DivWolf != 0:
                idx = self.DivWolf
        else:
            idx = self.highest(1)
        return idx

    def attack(self):
        idx = self.lowest(1)
        return idx

    def divine(self):
        idx = 0
        if self.base_info['day'] == 0:
            if self.base_info['agentIdx'] != self.game_setting['playerNum']:
                idx = self.base_info['agentIdx']
            else:
                idx = 0

        # highest prob ww in alive and not divined agents provided watashi ningen
        p = -1
        p_mat = self.role_p(self.predicter_5.p_60)
        for i in range(5):
            if p_mat[i, 1] > p and i not in self.divined_list:
                p = p_mat[i, 1]
                idx = i
        self.divined_list.append(idx)
        return idx + 1

    def finish(self):
        return


agent = PythonPlayer(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
