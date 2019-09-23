from .tensor60 import Tensor60
import numpy as np
import math

class Predictor_5(object):
    
    def __init__(self):
        self.case5 = Tensor60()
        
        # num of param
        self.n_para_3d = 3
        self.n_para_2d = 6
        
        # param_linear
        self.para_3d = np.zeros((4, 4, self.n_para_3d))
        """
        l09 = - math.log10(0.9)
        l05 = - math.log10(0.5)
        # werewolf might not vote possessed
        self.para_3d[1, 2, 0] = l05
        # possessed might not vote werewolf
        self.para_3d[2, 1, 0] = l09
        # possessed might not divine werewolf as werewolf
        self.para_3d[2, 1, 2] = l09
        # werewolf might not divine possessed as werewolf
        self.para_3d[1, 2, 2] = l05
        # werewolf would not divine werewolf as werewolf
        self.para_3d[1, 1, 2] = 1.0
        # Seer should not tell a lie
        self.para_3d[3, 0, 2] = 2.0
        self.para_3d[3, 2, 2] = 2.0
        self.para_3d[3, 1, 1] = 2.0
        """

        self.para_2d = np.zeros((4, self.n_para_2d))
        """
        # Seer should comingout correctly
        self.para_2d[3, 2] = -2.0
        self.para_2d[3, 5] = 2.0
        self.para_2d[3, 6] = 2.0
        # Possessed should comingout
        self.para_2d[2, 2] = -2.0
        self.para_2d[2, 6] = -1.0
        # villagers must not comingout
        self.para_2d[0, 2] = -2.0
        self.para_2d[0, 6] = -1.0
        # werewolf is alive
        self.para_2d[1, 0] = 100.0
        self.para_2d[1, 1] = 100.0
        """

        # my param
        param_data = [-0.48876856, 1.32325966,-2.39670778,-0.14712448,-2.7766485 , 2.32391299,
                        -0.5622551 , 0.86674804,-2.84946199,-0.49219547, 1.33906383,-2.39909599,
                        1.45915514, 0.41476269,-0.03100278, 0.        ,-0.43673512, 0.        ,
                        1.18398797, 0.25466381,-1.22750884, 1.45444443, 0.40961036,-0.02897342,
                        -0.07213567,-0.53800324, 0.11544423,-0.01796048,-0.66469015,-1.21257191,
                        0.        ,-0.7491663 , 0.        ,-0.07147983,-0.53479981, 0.11327178,
                        -0.47918485, 1.311403  ,-2.47759178,-0.13661977,-2.29967171, 2.24917958,
                        -0.55252677, 0.82057636,-2.43349497, 0.        , 0.38452864, 0.        ,
                        -0.48393659, 0.29507184,-1.48710682, 0.27663616,-2.48089234,-0.38545322,
                        -0.4129754 ,-3.72706974,-0.75448952,-0.62995894,-2.16836333,-1.13148981,
                        -0.1823445 ,-0.03449806, 3.41947274,-0.79649188, 4.39490806, 3.36059844,
                        -0.47933777, 0.29119958,-1.43782762, 0.25636475,-1.99020246,-0.3487788 ]
        
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
        #xv = self.case5.get_case60_df()["agent_"+str(self.base_info['agent'])].values

        # use for Python agent
        xv = self.case5.get_case60_df()["agent_"+str(self.base_info['agentIdx'])].values

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
                # comingout
                if content[0] == 'COMINGOUT':
                    # self
                    if int(content[1][6:8]) == gamedf.agent[i]:
                        if content[2] == 'SEER':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 2] = 1
                        elif content[2] == 'VILLAGER':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 3] = 1
                        elif content[2] == 'POSSESSED':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 4] = 1
                        elif content[2] == 'WEREWOLF':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 5] = 1
                # divined
                elif content[0] == 'DIVINED':
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                    self.x_2d[gamedf.agent[i] - 1, 2] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 0
                        
        
    def update_df(self):
        # update 60 dataframe
        self.df_pred = self.case5.apply_tensor_df(self.x_3d, self.x_2d, 
                                                   names_3d=["VOTE", "DIV_HM", "DIV_WW"], 
                                                   names_2d=['executed', 'attacked', 'CO_SEER', 'CO_VILLAGER', 'CO_POSSESSED', 'CO_WEREWOLF'])
        
    
    def update_pred(self):
        # predict
        # Linear
        l_para = np.append(self.para_3d.reshape((4*4*self.n_para_3d, 1)), self.para_2d.reshape((4*self.n_para_2d, 1)))
        self.df_pred["pred"] = np.matmul(self.df_pred.values, l_para.reshape(-1, 1))
        self.df_pred["pred"] = np.exp(-np.log(10)*self.df_pred["pred"])
        
        # average
        self.p_60 = self.df_pred["pred"] / self.df_pred["pred"].sum()
        
        
    def ret_pred(self):
        p = self.p_60
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes = [0, 0]).transpose()
        
    def ret_pred_wx(self, r):
        p = self.p_60 * self.watshi_xxx[:, r]
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes = [0, 0]).transpose()