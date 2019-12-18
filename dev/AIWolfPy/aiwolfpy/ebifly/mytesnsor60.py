from __future__ import print_function, division
import numpy as np
import pandas as pd


class Tensor60(object):

    def __init__(self):
        self.tensor60_3d = np.zeros((60, 4, 4, 5, 5), dtype='float32')
        self.case60 = np.zeros((60, 5), dtype='int32')

        self.role = ['VILLAGER', 'WEREWOLF', 'POSSESSED', 'SEER']
        self.species = ['HUMAN', 'WEREWOLF']
        self.verb = ['ESTIMATE', 'COMINGOUT', 'DIVINATION', 'VOTE',
                     'ATTACK', 'DIVINED', 'VOTED', 'ATTACKED']
        self.prot = ['vote', 'execute', 'dead']
        self.action = self.verb + self.prot
        self.object = self.role + self.species

        ind = 0
        for w in range(5):
            for p in range(5):
                for s in range(5):
                    if w != p and p != s and s != w:
                        self.case60[ind, w] = 1
                        self.case60[ind, p] = 2
                        self.case60[ind, s] = 3
                        ind += 1

        self.case60_df = pd.DataFrame(self.case60)
        self.case60_df.columns = ["agent_" + str(i) for i in range(1, 6)]

        for ind in range(60):
            for i in range(5):
                for j in range(5):
                    self.tensor60_3d[ind, self.case60[ind, i],
                                     self.case60[ind, j], i, j] = 1.0

    def get_case60(self):
        return self.case60

    def get_case60_df(self):
        return self.case60_df

    def apply_tensor_3d(self, x_para):
        return np.tensordot(self.tensor60_3d, x_para, axes=[[3, 4], [0, 1]])

    def apply_tensor_df(self, x_para, names):
        collected_df = pd.DataFrame(
            self.apply_tensor_3d(x_para).reshape((60, -1)))
        collected_df.columns = names
        return collected_df
