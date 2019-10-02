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
        self.para_2d = np.zeros((4, self.n_para_2d))
        
        # my param
        param_data = [ 0.24921325,-0.07492308,-0.90992294,-0.40840993,-1.19011885, 1.09578128,
  -1.06844736,-0.62145166, 0.87175914, 0.24037052,-0.07130385,-0.91410467,
   0.55335573,-1.31745731, 1.61472384, 0.        , 0.        , 0.        ,
   1.10996408,-0.12299003, 0.58396639, 0.54744138,-0.93853567, 1.58499902,
   0.12749461, 0.31199876, 0.00949664,-0.06356129, 0.14898545, 0.24503428,
   0.        , 0.        , 0.        , 0.10540031, 0.31983946, 0.01012104,
   0.27831923,-0.3132359 ,-1.20917923,-0.37176817,-0.95229107, 0.77812608,
  -1.01640282,-0.41544549, 0.54066437, 0.        , 0.        , 0.        ,
  -0.76354078, 0.33645175,-1.17714677, 0.        ,-1.708275  , 0.        ,
  -0.2315225 ,-2.0192374 , 0.03076725, 0.        , 1.50901701, 0.        ,
   0.80945028, 0.07787959, 3.8619232 , 0.        , 2.36843737, 0.        ,
  -0.7034654 , 0.35877206,-0.86350605, 0.        ,-1.32996873, 0.        ]
        
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