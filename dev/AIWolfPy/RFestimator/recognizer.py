from . import parse_content
import pandas as pd

class Recognizer(object):
    def __init__(self):
        self.agent_info = [dict(
            true_role="ERROR, DO INITIALIZE.",
            CO="none",
            got_white=set(),
            got_black=set(),
            identified_black=set(),
            identified_white=set(),
            is_alive="alive",
            voted_count=0,
            vote_COer_count=0,#!
            
            action_hist=[]
        ) for i in range(16)]
        self.game_info = dict(
            COseer=set(),
            executed_seer=set(),#OK
            attacked_seer=set(),#OK
            COmed=set(),
            executed_med=set(),#OK
            attacked_med=set(),#OK
            divined_black=set(),
            divined_white=set(),
            identified_black=set(),
            identified_white=set(),
            dead=set(),
            attacked=set(),
            executed=set(),
            #wolf=set()
        )
        self.day = 0
        self.turn = 0

    def black_or_white(self, agentID):
        agent = self.agent_info[agentID]
        got_white, got_black = len(agent["got_white"]), len(agent["got_black"])
        if not self.game_info["COseer"]:
            return "none"
        if got_white >= len(self.game_info["COseer"]):
            return "white_all"
        if got_white and not got_black:
            return "white"
        if got_black and not got_white:
            return "black"
        if got_black and got_white:
            return "panda"
        #if got_black >= len(self.game_info["COseer"]):
        #    return "black_all"
        return "none"

    def initialize_agent(self, line):
        att = line.split(",")
        self.agent_info[int(att[2])]["true_role"] = att[3].replace("\n", "")
        # print(self.agent_info[int(att[2])])

    def agent_class(self, agentID):
        agent = self.agent_info[agentID]
        return dict(CO=agent["CO"], divined=self.black_or_white(agentID), )#status=agent["is_alive"]

    def make_act_dict(self, act, agent_dict=None):
        return dict(act=act, agent=agent_dict)
    
    def update_info_strline(self,line):
        """
        strのlineを一行受け取って更新する
        """    
        day,stat,talkID,turn,agentID,action=line.replace("\n","").split(",")
        day,talkID,turn,agentID=int(day),int(talkID),int(turn),int(agentID)
        #Skip
        if action in ["Skip","Over"]:
            return 
        parsed_list = parse_content.parse_text(action)
        for interpreted in parsed_list:
            self.talk_recognize(line,stat, agentID, interpreted)
        #self.talk_recognize(s,stat,agentID,interpreted)
                
    
    def update_info(self, diff):
        for i in diff.iterrows():
            # print(i)
            to = 0
            line = i[1]
            who = int(line["agent"])
            #print(line["text"])
            parsed_list = parse_content.parse_text(line["text"])
            #print("parsed list==================")
            #print(parsed_list)
            #print("parsed list==================")
            # Diffを分解して渡す
            #self.talk_recognize(i,line,who,line["text"])
            for interpreted in parsed_list:
                self.talk_recognize(i,line["type"], who, interpreted)

    def talk_recognize(self, i,typ, agentID, interpreted):
        """
        Diffを受け取って1行ずつ更新
        """
        #print("TALK RECOGNIZER")
        #print(i,typ,agentID,interpreted)
        #print(line)
        if typ == "talk":
            content = interpreted.split(" ")
            #print("content:", content)

            # COMINGUOUTの場合
            if content[0] == "COMINGOUT":
                targetID = int(content[1][6:8])
                if targetID == agentID:
                    if content[2] == "SEER":
                        self.agent_info[agentID]["CO"] = "SEER"
                        ##print("info update", agentID, self.agent_info[agentID])
                    if content[2] == "MEDIUM":
                        self.agent_info[agentID]["CO"] = "MEDIUM"
                        ##print("info update", agentID, self.agent_info[agentID])

            elif content[0] == "DIVINED":
                targetID = int(content[1][6:8])
                # 異常AgentNo対策
                # if not 0 <targetID < self.player_size:
                #    return
                self.agent_info[agentID]["CO"] = "SEER"
                if content[2] == "WEREWOLF":
                    ##print(line, " -> divined", targetID, "as werewolf.")
                    self.agent_info[targetID]["got_black"].add(agentID)
                else:
                    #print(line, " -> divined", targetID, "as human.")
                    self.agent_info[targetID]["got_white"].add(agentID)
                ##print("info update", agentID, self.agent_info[agentID])
                ##print("info update", targetID, self.agent_info[targetID])
            elif content[0] == "IDENTIFIED":
                targetID = int(content[1][6:8])
                self.agent_info[agentID]["CO"] = "MEDIUM"
                if content[2] == "WEREWOLF":
                    ##print(line, " -> divined", targetID, "as werewolf.")
                    self.agent_info[targetID]["identified_black"].add(agentID)
                else:
                    #print(line, " -> divined", targetID, "as human.")
                    self.agent_info[targetID]["identified_white"].add(agentID)
        
        elif typ == "vote":
            targetID = agentID
            agentID= i[0] #idx
            #print("vote",agentID,"->",targetID)
            self.agent_info[targetID]["voted_count"] += 1
            
            if self.agent_info[targetID]["CO"] in ["SEER","MEDIUM","BODYGUARD"]:
                print("it is COed. recorded.")
                self.agent_info[agentID]["vote_COer_count"]+=1

        elif typ == "execute":
            self.agent_info[agentID]["is_alive"] = "executed"
        elif typ == "dead":
            self.agent_info[agentID]["is_alive"] = "attacked"
        # print(line)
        # print(line["text"].split())

    def zip_game_info(self):
        # self.game_info=dict(COseer=set(),COmed=set(),divined_black=set(),divined_white=set(),
        #                    identified_black=set(),identified_white=set(),dead=set(),wolf=set())
        ret=dict()
        for key,val in self.game_info.items():
            ret[key]=len(val)
        return ret

    def line2id(self, line):
        """
        talk一行を受け取り
        行動IDを返す
        """
        day, stat, talkID, turn, agentID, action = line.replace(
            "\n", "").split(",")
        day, talkID, turn, agentID = int(day), int(
            talkID), int(turn), int(agentID)
        # Skip
        if action in ["Skip", "Over"]:
            return "Over"

        # Comingout
        s = action.split(" ")
        if s[0] == "COMINGOUT":
            if int(s[1][6:8]) == agentID:
                return "COMINGOUT_"+s[2]

        # あとは対称指定系だけ残るはず

        # 行動
        if s[0] not in ["VOTE", "ESTIMATE", "DIVINED", "IDENTIFIED"]:
            #print("failed to catch:"+line)
            return "failed to catch"

        # 先に対象についての情報を作る
        # print(s[1])
        targetID = int(s[1][6:8])
        agent_class = self.agent_class(targetID)

        if s[0] == "VOTE":
            return "ESTIMATE_black", agent_class

        elif s[0] == "DIVINED":
            self.game_info["COseer"].add(agentID)
            if s[2] == "WEREWOLF":
                self.game_info["divined_black"].add(targetID)
            else:
                self.game_info["divined_white"].add(targetID)

        elif s[0] == "IDENTIFIED":
            self.game_info["COmed"].add(agentID)
            if s[2] == "WEREWOLF":
                self.game_info["identified_black"].add(targetID)
            else:
                self.game_info["identified_white"].add(targetID)

        judge = "_black" if s[2] == "WEREWOLF" else "_white"

        return s[0]+judge, agent_class

        #print("failed to catch:"+line)
        # return "failed to catch"

    def update_game_info(self,):
        """
        game_infoの更新
        """
        for agentID, agent in enumerate(self.agent_info[1:]):
            agentID+=1
            if agent["CO"] == "SEER":
                self.game_info["COseer"].add(agentID)
            if agent["CO"] == "MEDIUM":
                self.game_info["COmed"].add(agentID)
            if agent["got_white"]:
                self.game_info["divined_white"].add(agentID)
            if agent["got_black"]:
                self.game_info["divined_black"].add(agentID)
            if agent["identified_black"]:
                self.game_info["identified_black"].add(agentID)
            if agent["identified_white"]:
                self.game_info["identified_white"].add(agentID)
            if agent["is_alive"]=="attacked":
                self.game_info["attacked"].add(agentID)
                self.game_info["dead"].add(agentID)
                if agent["CO"]=="SEER":
                    self.game_info["attacked_seer"].add(agentID)
                if agent["CO"]=="MEDIUM":
                    self.game_info["attacked_med"].add(agentID)
            elif agent["is_alive"]=="executed":
                self.game_info["executed"].add(agentID)
                self.game_info["dead"].add(agentID)
                if agent["CO"]=="SEER":
                    self.game_info["executed_seer"].add(agentID)
                elif agent["CO"]=="MEDIUM":
                    self.game_info["executed_med"].add(agentID)
        
    def update_action_hist(self,line):
        #本番では搭載しない
        att=line.replace("\n","").split(",")
        if att[1]=="talk":
            day,stat,talkID,turn,agentID,action=att
            day,talkID,turn,agentID=int(day),int(talkID),int(turn),int(agentID)
            return_action=self.line2id(line)
        elif att[1]=="vote":
            day,stat,agentID,targetID=att
            agentID,targetID=int(agentID),int(targetID)
            turn=-1
            return_action=("VOTE",self.agent_class(targetID))
            self.vote(line)
        elif att[1]=="attack":
            day,stat,targetID,success=att
            targetID=int(targetID)
            turn=-1
            if success.startswith("fal"):
                return
            return_action=("ATTACK",self.agent_class(targetID))
            #agentIDが受け取れないのでこいつだけ特別
            d=dict(day=day,turn=turn,role="WEREWOLF",game_info=self.zip_game_info(),action=return_action)
            self.attack(line)
            for agent in self.agent_info:
                #print(agent["true_role"])
                if agent["true_role"]=="WEREWOLF":
                    agent["action_hist"].append(d)
                    return
        elif att[1]=="divine":
            day,stat,agentID,targetID,result=att
            agentID,targetID=int(agentID),int(targetID)
            turn=-1
            return_action=("DIVINE",self.agent_class(targetID))
        elif att[1]=="guard":
            day,stat,agentID,targetID,result=att
            agentID,targetID=int(agentID),int(targetID)
            turn=-1
            return_action=("GUARD",self.agent_class(targetID))
        
        d=dict(day=day,turn=turn,role=self.agent_info[agentID]["true_role"],game_info=self.zip_game_info(),action=return_action)
        self.agent_info[agentID]["action_hist"].append(d)
        
    def vote(self, line):
        # 投票を行動記録に追加
        day, v, agentID, targetID = line.replace("\n", "").split(",")
        agentID, targetID = int(agentID), int(targetID)

        self.agent_info[targetID]["voted_count"] += 1
        if self.agent_info[targetID]["CO"] in ["SEER","MEDIUM"]:
            self.agent_info[agentID]["vote_COer_count"]+=1

    def execute(self, line):
        attr = line.replace("\n", "").split(",")
        executed = int(attr[2])
        self.agent_info[executed]["is_alive"] = "executed"
        self.game_info["dead"].add(executed)

    def attack(self, line):
        attr = line.replace("\n", "").split(",")
        if attr[3] == "false":
            return
        attacked = int(attr[2])
        self.agent_info[attacked]["is_alive"] = "attacked"
        self.game_info["dead"].add(attacked)

        
from io import StringIO
def str2df(s):
    DATA=StringIO("day,type,idx,turn,agent,text\n"+s)
    tmp=pd.read_csv(DATA,sep=",")
    return tmp