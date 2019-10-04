from .tensor60 import Tensor60
import numpy as np
import math
import inspect

class Predictor_5(object):
    
    def __init__(self):
        self.case5 = Tensor60()
        
        # num of param
        self.n_para_3d = 3
        self.n_para_2d = 6
        
        # param_linear
        self.para_3d = np.zeros((4, 4, self.n_para_3d))
        self.para_2d = np.zeros((4, self.n_para_2d))
        
        # my param
        param_data = [-1.44451888, 0.        , 0.        , 0.49386889, 0.        , 0.        ,
  -1.30403434, 0.        , 0.        ,-1.44407568, 0.        , 0.        ,
   1.24035954, 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.89130139, 0.        , 0.        , 1.23994316, 0.        , 0.        ,
   0.23522107, 0.        , 0.        ,-0.17730285, 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.23516458, 0.        , 0.        ,
  -1.44304051, 0.        , 0.        , 0.49526556, 0.        , 0.        ,
  -1.30172934, 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.6057206 , 0.35697041,-1.1995996 , 0.        , 0.        , 0.        ,
  -0.59968686,-4.24262938,-0.40171481, 0.        , 0.        , 0.        ,
   0.47169936,-0.03241234, 2.37014808, 0.        , 0.        , 0.        ,
   0.60477882, 0.356686  ,-1.19911471, 0.        , 0.        , 0.        ]

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
            xv = self.case5.get_case60_df()["agent_"+str(self.base_info['agentIdx'])].values
        else:
            xv = self.case5.get_case60_df()["agent_"+str(self.base_info['agent'])].values

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