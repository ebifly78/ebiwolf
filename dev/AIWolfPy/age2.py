#!/usr/bin/env python
from __future__ import print_function, division 

# this is main script

import aiwolfpy
import aiwolfpy.contentbuilder as cb

# sample 
import aiwolfpy.ebifly

myname = 'ebifly'

class PythonPlayer(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        
        # predictor from sample
        # DataFrame -> P
        self.predicter_5 = aiwolfpy.ebifly.Predictor_5()
        
        
    def getName(self):
        return self.myname
        
    def initialize(self, base_info, diff_data, game_setting):
        # print(base_info)
        # print(diff_data)
        # base_info
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        
        # initialize
        if self.game_setting['playerNum'] == 5:   
            self.predicter_5.initialize(base_info, game_setting)
                
        ### EDIT FROM HERE ###     
        self.divined_list = []
        self.comingout = ''
        self.myresult = ''
        self.not_reported = False
        self.vote_declare = 0
        

        
    def update(self, base_info, diff_data, request):
        # print(base_info)
        # print(diff_data)
        # update base_info
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
                    self.myresult = diff_data['text'][i]
                
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
        elif self.base_info['myRole'] == 'MEDIUM' and self.comingout == '':
            self.comingout = 'MEDIUM'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        
        # 2.report
        if self.base_info['myRole'] == 'SEER' and self.not_reported:
            self.not_reported = False
            return self.myresult
        elif self.base_info['myRole'] == 'MEDIUM' and self.not_reported:
            self.not_reported = False
            return self.myresult
        elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
            self.not_reported = False
            # FAKE DIVINE
            # highest prob ww in alive agents
            p = -1
            idx = 1
            p0_mat = self.predicter_5.ret_pred_wx(2)
            for i in range(1, 6):
                p0 = p0_mat[i-1, 1]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
            self.myresult = 'DIVINED Agent[' + "{0:02d}".format(idx) + '] ' + 'HUMAN'
            return self.myresult
            
        # 3.declare vote if not yet
        if self.vote_declare != self.vote():
            self.vote_declare = self.vote()
            return cb.vote(self.vote_declare)
            
        # 4. skip
        if self.talk_turn <= 10:
            return cb.skip()
            
        return cb.over()

    def whisper(self):
        return cb.skip()
        
    def vote(self):
        if self.base_info['myRole'] == "WEREWOLF":
            p0_mat = self.predicter_5.ret_pred_wx(1)
            p = -1
            idx = 1
            for i in range(1, 6):
                p0 = p0_mat[i-1, 3]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
        elif self.base_info['myRole'] == "POSSESSED":
            p0_mat = self.predicter_5.ret_pred_wx(2)
            p = -1
            idx = 1
            for i in range(1, 6):
                p0 = p0_mat[i-1, 3]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
        elif self.base_info['myRole'] == "SEER":
            p0_mat = self.predicter_5.ret_pred_wx(3)
            p = -1
            idx = 1
            for i in range(1, 6):
                p0 = p0_mat[i-1, 1]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
        else:
            p0_mat = self.predicter_5.ret_pred_wx(0)
            p = -1
            idx = 1
            for i in range(1, 6):
                p0 = p0_mat[i-1, 1]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
        return idx

    def attack(self):
        # lowest prob ps in alive agents
        p = 1
        idx = 1
        p0_mat = self.predicter_5.ret_pred_wx(1)
        for i in range(1, 6):
            p0 = p0_mat[i-1, 2]
            if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 < p and i != self.base_info['agentIdx']:
                p = p0
                idx = i
        return idx
    
    def divine(self):
        # highest prob ww in alive and not divined agents provided watashi ningen
        p = -1
        idx = 1
        p0_mat = self.predicter_5.ret_pred_wx(3)
        for i in range(1, 6):
            p0 = p0_mat[i-1, 1]
            if self.base_info['statusMap'][str(i)] == 'ALIVE' and i not in self.divined_list and p0 > p:
                p = p0
                idx = i
        self.divined_list.append(idx)
        return idx
    
    
    def finish(self):
        pass
        
 

agent = PythonPlayer(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)