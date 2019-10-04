#!/usr/bin/env python
from __future__ import print_function, division

# this is main script.
# simple version

import aiwolfpy
import aiwolfpy.contentbuilder as cb

from collections import Counter
import random
from pprint import pprint

#import predictor_v2 as predictor
import predictor_random as predictor

from jinro_natural import agent_natural as natural
from RFestimator import estimator
import util

myname = 'calups'

def isMostFrequent(l, me):
    """
    リスト内のme要素がそこで最頻出かどうかを調べる
    タイの場合は不定
    """
    try:
        from collections import Counter
        c = Counter(filter(lambda s: s != None, l))
        if c.most_common(1)[0][0] == int(me):
            return True
        return False
    except Exception:
        return False


class Agent(object):
    """
    とりあえず呼び出すエージェント
    役職が与えられるとself.behaviorにその役職の人格を入れる
    以後はそれを呼び出して色々する
    """

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        self.behavior = None

        #先に作っとく
        #self.predictor_5 = predictor.Predicter([str(i+1) for i in range(5)])
        #self.predictor_5.role_est("WEREWOLF",[str(i+1) for i in range(5)])
        self.natural_5=natural.SampleAgent(myname)

        self.predictor_15 = predictor.Predicter([str(i+1) for i in range(15)])
        self.predictor_15.role_est("WEREWOLF",[str(i+1) for i in range(15)])

        self.rfestimator=estimator.CalupsAI()

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):

        self.base_info = base_info

        # game_setting
        self.game_setting = game_setting
        self.remaining = len(base_info["remainTalkMap"])
        myRole = base_info["myRole"]

        if self.game_setting["playerNum"] == 5:
            self.natural_5.initialize(base_info, diff_data, game_setting)
            self.behavior = self.natural_5
            return
        elif self.game_setting["playerNum"] == 15:
            if myRole == "VILLAGER":
                self.behavior = VillagerBehavior(self.myname)
            elif myRole == "MEDIUM":
                self.behavior = MediumBehavior(self.myname)
            elif myRole == "BODYGUARD":
                self.behavior = BodyguardBehavior(self.myname)
            elif myRole == "SEER":
                self.behavior = SeerBehavior(self.myname)
            elif myRole == "POSSESSED":
                self.behavior = PossessedBehavior(self.myname)
            elif myRole == "WEREWOLF":
                #self.behavior = WerewolfBehavior(self.myname)
                """狼は狂人として動く"""
                self.behavior = PossessedBehavior(self.myname)
            else:
                print("CAUTION: valid role not found, so chosen villager behav.")
                self.behavior = VillagerBehavior(self.myname)

        self.behavior.initialize(base_info, diff_data, game_setting)
        self.behavior.init_predictor(self.predictor_15)
        self.behavior.init_rfestimator(self.rfestimator)
        print("=============INITIALIZE FINISHED============")

    def update(self, base_info, diff_data, request):
        try:
            self.behavior.update(base_info, diff_data, request)
        except Exception:
            pass

    def dayStart(self):
        try:
            self.behavior.dayStart()
        except Exception:
            pass

    def talk(self):
        #return self.behavior.talk()
        try:
            return self.behavior.talk()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return cb.over()
        # return cb.over()

    def whisper(self):
        #return self.behavior.whisper()
        try:
            return self.behavior.whisper()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return cb.over()

    def vote(self):
        #return self.behavior.vote()
        try:
            return self.behavior.vote()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return 1

    def attack(self):
        #return self.behavior.attack()
        try:
            return self.behavior.attack()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return 1

    def divine(self):
        #return self.behavior.divine()
        try:
            return self.behavior.divine()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return 1

    def guard(self):
        #return self.behavior.guard()
        try:
            return self.behavior.guard()
        except Exception:
            print("ERROR!!!!!!!!!!something wrong and returned tekito")
            return 1

    def finish(self):
        return self.behavior.finish()


class VillagerBehavior(object):
    """
    村人の振る舞い
    これをあらゆる役職の基底クラスにする
    """

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        # 村人であるはずの自分に黒出ししやがるなど、完全におかしいやつ
        self.absolute_dislike = []
        # 今日の投票先
        self.todays_vote = None

    def getName(self):
        """
        名前を返せば良い
        """
        return self.myname

    def init_rfestimator(self, predictor_15):
        """
        RFclassifier渡す
        """
        self.rfestimator = predictor_15
        self.rfestimator.initialize(
            self.base_info["agentIdx"], self.base_info["myRole"])

    def init_predictor(self,predictor_15):
        """
        predictorを決定
        """
        self.predictor=predictor_15

    def initialize(self, base_info, diff_data, game_setting):
        """
        新たなゲーム開始時に一度だけ呼ばれる
        前回のゲームデータのリセット等してもいいししなくてもいい
        """
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        self.player_size = len(self.base_info["remainTalkMap"].keys())
        # まだ誰にも占われてない人集合
        self.greys = set(self.base_info["remainTalkMap"].keys())
        print("GREYS: ", self.greys)

        

        # print(diff_data)
        # print(base_info)
        
        

    def update(self, base_info, diff_data, request):
        """
        initialize以外のすべてのメソッドの前に呼ばれる
        requestには要求が色々入ってる（DAILY_INITIALIZE,DAILY_FINISH,DIVINE,TALKなど）
        ベースインフォとでぃふデータを記録して予測器をアップデートする
        """
        print("REQUEST:", request)
        self.base_info = base_info
        self.diff_data = diff_data
        try:
            self.predictor.update(diff_data)
            self.predictor.update_agent_info(self.rfestimator.rec.agent_info)

            self.rfestimator.update(base_info, diff_data)
            self.rfestimator.rec.update_game_info()  #Gameinfoの更新
            print(self.rfestimator.rec.game_info)

        except Exception:
            print("ERROR!!!!FAILED TO UPDATE GAME INFO")

        # 一日のはじめにやること
        if request == "DAILY_INITIALIZE":
            # 誰が誰に投票しそうかリストの初期化
            self.hate_who = [None] * self.player_size
        for i in self.diff_data.iterrows():
            if i[1]["type"] == "talk":

                # 発言内容からESTIMATE WEREWOLFとVOTEを見つける
                self.talk_turn=i[1]["turn"]+1
                content = i[1]["text"].split()
                if content[0] == "ESTIMATE":
                    # 異常AgentNo対策
                    if int(content[1][6:8]) > self.player_size:
                        return
                    if content[2] == "WEREWOLF":
                        self.hate_who[i[1]["agent"] - 1] = int(content[1][6:8])
                elif content[0] == "VOTE":
                    self.hate_who[i[1]["agent"]-1] = int(content[1][6:8])
                    # print(self.hate_who)
                # 発言内容から占い情報を見つけて、グレーリストから削除
                if content[0] == "DIVINED":
                    # 異常No対策
                    if int(content[1][6:8]) > self.player_size:
                        return
                    #print("GREY LIST REMOVED", {int(content[1][6:8])})
                    # print(self.greys)
                    self.greys -= {str(int(content[1][6:8]))}


    def dayStart(self):
        self.talk_turn = 0
        self.todays_vote = None
        return None

    def talk(self):
        """
        村人は原則カミングアウトしない。
        1.その日までの結果を基に一番怪しいやつにVote宣言をする
        2.カミングアウトなどで一番怪しいやつが変化したら改めてvote宣言をする
        3.最後にVote宣言した相手に実際の投票をする
        """

        #各日4ターン目で投票先宣言
        if self.talk_turn==5:
            cand = self.rfestimator.decide_action("vote")
            #print("TURN",self.talk_turn)
            print(cand)
            return util.list2protocol(cand,"VOTE OR")

        # 何もなければOver
        if self.talk_turn < 5:
            return cb.skip()
        return cb.over()

    def whisper(self):
        """
        村人はwhisperを呼ばれることがない
        """
        return cb.over()

    def vote(self):
        """
        一応最後に仮投票先を更新してから投票
        """
        cand=self.rfestimator.decide_action("vote")
        self.todays_vote = self.predictor.role_est("WEREWOLF", cand)
        return self.todays_vote

    def attack(self):
        """
        村人は襲撃しない
        """
        return self.base_info['agentIdx']

    def divine(self):
        """
        村人は占わない
        """
        return self.base_info['agentIdx']

    def guard(self):
        """
        村人は守らない
        """
        return self.base_info['agentIdx']

    def finish(self):
        """
        ゲーム終了時に呼ばれる（？）
        """
        return None


class MediumBehavior(VillagerBehavior):
    """
    霊媒師の振る舞い
    """

    def __init__(self, agent_name):
        # 村人と同じ
        super().__init__(agent_name)
        # 霊媒結果
        self.result = []

    def initialize(self, base_info, diff_data, game_setting):
        # 村人と同じ
        super().initialize(base_info, diff_data, game_setting)
        # 結果の初期化
        self.result = []
        # ステルスモード
        self.stealth = True

    def update(self, base_info, diff_data, request):
        # 村人と同じ
        super().update(base_info, diff_data, request)

        # ===霊媒特殊処理
        # 結果がその日の初期化データだったら
        # 村側役職持ちの場合、前日夜に使った能力の結果が帰ってくる
        # それをresultに格納しておく
        if request == "DAILY_INITIALIZE":
            for i in range(diff_data.shape[0]):
                if diff_data["type"][i] == "identify":
                    self.result.append(diff_data["text"][i])
        print(self.result)

    def dayStart(self):
        # 村人と同じ
        super().dayStart()
        return None

    def talk(self):
        """
        霊媒のCO戦略
        定石：基本ステルス（村人と同じ）、ただし霊媒結果●が出たら即時COする

        基本は村人と同じ
        """
        
        print("TALK TURN:", self.talk_turn)

        # ===霊媒特殊処理ここから
        # ステルスモード解除してCO
        if self.stealth == True:
            # 1日目からロケットCO
            self.stealth = False
            return cb.comingout(self.base_info['agentIdx'], "MEDIUM")

        # 一度ステルスモード解除されていたら、resultを全部ぶちまける
        if self.stealth == False and len(self.result) > 0:
            # 最後の結果を取り出してIDENTIFIED発言
            last_res = self.result.pop(-1)
            return last_res

        # 何もなければOver
        return super().talk()

    def vote(self):
        # 村人と同じ
        return super().vote()

    def finish(self):
        return None


class BodyguardBehavior(VillagerBehavior):
    """
    狩人の振る舞い
    """

    def __init__(self, agent_name):
        # 村人と同じ
        super().__init__(agent_name)
        # 護衛記録
        self.result = []

    def initialize(self, base_info, diff_data, game_setting):
        # 村人と同じ
        super().initialize(base_info, diff_data, game_setting)
        # 結果の初期化
        self.result = []
        # ステルスモード
        self.stealth = True

    def update(self, base_info, diff_data, request):
        # 村人と同じ
        super().update(base_info, diff_data, request)

        # 結果がその日の初期化データだったら
        # 村側役職持ちの場合、前日夜に使った能力の結果が帰ってくる
        # それをresultに格納しておく
        # ===狩人特殊処理
        if request == "DAILY_INITIALIZE":
            for i in range(diff_data.shape[0]):
                if diff_data["type"][i] == "guard":
                    self.result.append(diff_data["text"][i])
        print(self.result)

    def dayStart(self):
        # 村人と同じ
        super().dayStart()
        return None

    def talk(self):
        """
        狩人のCO戦略
        基本的にステルス
        吊られそうになるか4日目になったらCO
        """
        """
        # ステルス解除、カミングアウト
        print("GUARD: "+str(self.base_info["day"]))
        if self.stealth == True and self.base_info["day"] >= 6:
            print("GUARD: stealth mode disabled")
            self.stealth = False
            return cb.comingout(self.base_info['agentIdx'], "BODYGUARD")

        # 吊られそうならカミングアウト
        print("HATRED LIST:", self.hate_who)
        if isMostFrequent(self.hate_who, int(self.base_info["agentIdx"])):
            self.stealth = False
            return cb.comingout(self.base_info["agentIdx"], "BODYGUARD")

        # ステルス解除後は履歴垂れ流し
        if self.stealth == False and len(self.result) > 0:
            # 最後の結果を取り出してguarded発言
            last_res = self.result.pop(0)
            print(last_res)
            return last_res
        """

        # 何もなければOver
        return super().talk()

    def vote(self):
        # 村人と同じ
        return super().vote()

    def guard(self):
        """
        狩人の護衛戦略
        唯一のSEER
        唯一のMEDIUM
        もっともらしいSEER
        白貰い
        ランダム
        """

        cand = self.rfestimator.decide_action("guard")
        med_est = self.predictor.role_est("MEDIUM", cand)
        return med_est

    def finish(self):
        return None


class SeerBehavior(VillagerBehavior):
    """
    占いの振る舞い
    """

    def __init__(self, agent_name):
        # 村人と同じ
        super().__init__(agent_name)
        # 占い記録
        self.result = []

    def initialize(self, base_info, diff_data, game_setting):
        # 村人と同じ
        super().initialize(base_info, diff_data, game_setting)
        # 結果の初期化
        self.result = []
        # ステルスモード
        self.stealth = True
        self.absolute_white=set()
        self.absolute_black=set()

    def update(self, base_info, diff_data, request):
        # 村人と同じ
        super().update(base_info, diff_data, request)

        # 結果がその日の初期化データだったら
        # 村側役職持ちの場合、前日夜に使った能力の結果が帰ってくる
        # それをresultに格納しておく
        # ===占い特殊処理
        if request == "DAILY_INITIALIZE":
            for i in range(diff_data.shape[0]):
                if diff_data["type"][i] == "divine":
                    self.result.append(diff_data["text"][i])
                    print("TRUE DIVINED:",diff_data["text"][i][14:16],"->",diff_data["text"][i][18:])
                    agentID=int(diff_data["text"][i][14:16])
                    result=diff_data["text"][i][18:]
                    if result.startswith("W"):
                        self.absolute_black.add(agentID)
                        self.rfestimator.rec.agent_info[agentID]["vote_COer_count"]+=100
                    else:
                        self.absolute_white.add(agentID)

        print(self.result)

    def dayStart(self):
        # 村人と同じ
        super().dayStart()
        return None

    def talk(self):
        """
        占いのCO戦略
        初日即CO
        """
        
        # ステルス解除、カミングアウト
        print("SEER: "+str(self.base_info["day"]))
        if self.stealth == True:
            print("SEER: stealth mode disabled")
            self.stealth = False
            return cb.comingout(self.base_info['agentIdx'], "SEER")

        # ステルス解除後は履歴垂れ流し
        if self.stealth == False and len(self.result) > 0:
            # 最後の結果を取り出してdivined発言
            last_res = self.result.pop(0)
            print(last_res)
            return last_res

        # 村人としての発言
        return super().talk()

    def vote(self):
        # 村人と同じ
        target=super().vote()
        if len(self.absolute_black)>0:
            return list(self.absolute_black)[0]
        if target in self.absolute_white:
            if len(self.greys)>0:
                return list(self.greys)[0]

        return target

    def divine(self):
        """
        占いの占い戦略
        1.まだ占われていない内で最も狼らしいやつを占う
        2.そんな対象がいなければ自分が占っていない内で
        """

        print("REQUEST:DIVINE")
        # 生存者の確認
        self.alive = []
        for i in range(self.player_size):
            # 1-origin
            i += 1
            if self.base_info["statusMap"][str(i)] == "ALIVE":
                self.alive.append(str(i))
        print("STILL ALIVES:", self.alive)

        cand = self.rfestimator.decide_action("divine")-{str(int(self.base_info["agentIdx"]))}
        if len(cand)==0:
            print("DIVINE len(candidate)==0, cand=not divined by me.")
            cand=self.alive-self.absolute_white-self.absolute_black-{str(int(self.base_info["agentIdx"]))}
        
        #cand = self.rfestimator.decide_action("divine")
        wolf_est = self.predictor.role_est("WEREWOLF", cand)
        print("DIVINE:", wolf_est)
        return wolf_est

    def finish(self):
        return None


class PossessedBehavior(VillagerBehavior):
    """
    狂人の振る舞い

    初日占いCO即黒出し、以後狼らしいやつから順にランダムで白
    """

    def __init__(self, agent_name):
        # 村人と同じ
        super().__init__(agent_name)
        # 占い記録
        self.result = []

    def initialize(self, base_info, diff_data, game_setting):
        # 村人と同じ
        super().initialize(base_info, diff_data, game_setting)
        # 結果の初期化
        self.result = []
        # ステルスモード
        self.stealth = True
        self.absolute_white=set()
        self.whisper_turn = 0
        #狼が狂人として振る舞うのに必要な変数
        self.wolfs = set()
        self.attack_success = True
        self.whisper_turn = 0

    def update(self, base_info, diff_data, request):
        # 村人と同じ
        super().update(base_info, diff_data, request)

        # 結果がその日の初期化データだったら
        # 村側役職持ちの場合、前日夜に使った能力の結果が帰ってくる
        # それをresultに格納しておく
        # ===狂人特殊処理
        if request == "DAILY_INITIALIZE" and self.base_info["day"] > 0:
            info=self.rfestimator.rec.game_info
            greys=set(range(1,16))-info["dead"]-info["COseer"]-info["COmed"]-info["divined_white"]-info["divined_black"]
            if len(greys)>0:
                wolf_est = self.predictor.role_est("WEREWOLF", greys)
            else:
                cand = set(range(1, 16))-info["dead"]
                wolf_est=self.predictor.role_est("WEREWOLF",cand)

            statement = 'DIVINED ' + util.int2agent(wolf_est)+ ' HUMAN'
            self.result.append(statement)
            print("POSSESSED: FALSE DIVINED")
            print(self.result)

    def dayStart(self):
        # 村人と同じ
        super().dayStart()
        self.talk_turn=0

        # 生存者の確認
        self.alive = []
        for i in range(self.player_size):
            # 1-origin
            i += 1
            if self.base_info["statusMap"][str(i)] == "ALIVE":
                self.alive.append(str(i))

        return None

    def talk(self):
        """
        狂人のCO戦略
        初日即占いとしてCO
        """
        """
        村人は３，狼は２
        """
        print("POSSESSEDtalk",self.talk_turn)
        if self.base_info["day"] == 1:
            if self.talk_turn == 0:
                self.stealth = False
                print("POSSESSED, false comingout")
                return cb.comingout(self.base_info['agentIdx'],"SEER")
            if self.talk_turn == 1:
                info = self.rfestimator.rec.game_info
                seers=info["COseer"]-set([self.base_info["agentIdx"]])
                """
                占い黒出しは無効化
                """
                #if len(seers)>0:
                #    target = self.predictor.role_est("SEER", seers)
                #    return cb.divined(target,"WEREWOLF")
                #else:
                #target = self.predictor.role_est("WEREWOLF", self.alive-seers-info["dead"]-self.absolute_white)
                #self.absolute_white.add(target)
                #return cb.divined(target,"HUMAN")

        # ステルス解除後は履歴垂れ流し
        if self.stealth == False and len(self.result) > 0:
            # 最後の結果を取り出してdivined発言
            last_res = self.result.pop(0)
            print(last_res)
            return last_res

        # 何もなければOver
        return super().talk()

    def vote(self):
        # 村人と同じ
        return super().vote()

    def finish(self):
        return None

    def whisper(self):
        """
        秘密会話
        攻撃したい先を言って、一致しなかったら相手に合わせる

        襲撃先選択
        1.最も占いらしいやつ
        2.昨日の護衛結果が失敗だったら最も狩人らしいやつ
        """
        self.whisper_turn += 1

        # 生存者の確認
        self.alive = []
        for i in range(self.player_size):
            # 1-origin
            i += 1
            if self.base_info["statusMap"][str(i)] == "ALIVE":
                self.alive.append(str(i))

        if self.base_info["day"]==0:
            if self.whisper_turn==1:
                return cb.comingout(self.base_info["agentIdx"],"SEER")
            else:
                return cb.over()
        # 候補者セット
        #cand = set(self.alive)-{str(int(self.base_info["agentIdx"]))}
        cand = self.rfestimator.decide_action("attack")
        print(set([self.base_info["agentIdx"]]),self.rfestimator.rec.game_info["dead"],self.wolfs)
        cand = cand - set([self.base_info["agentIdx"]])-self.rfestimator.rec.game_info["dead"]-self.wolfs
        # 襲撃先の吟味
        role = "SEER"
        if self.attack_success:
            self.todays_vote = int(self.predictor.role_est("SEER", cand))
            if self.whisper_turn == 1 and self.base_info["day"]<3:
                return cb.estimate(self.todays_vote, role)
            if self.whisper_turn == 2:
                return cb.attack(self.todays_vote)
        else:
            # 昨日失敗してたら狩人がまだ生きてるということなので
            role = "BODYGUARD"
            self.todays_vote = int(self.predictor.role_est("BODYGUARD", cand)) 
            if self.whisper_turn==1 and self.base_info["day"]<3:
                return cb.estimate(self.todays_vote, role)
            if self.whisper_turn==2:
                return cb.attack(self.todays_vote)

        return cb.over()

    def attack(self):
        """
        宣言した襲撃先をやる
        """
        self.whisper()

        return self.todays_vote

class WerewolfBehavior(VillagerBehavior):
    """
    狼の振る舞い
    ステルスモード時：ステルス
    吊られそうになったら霊媒CO、嘘の霊媒履歴を吐く
    """

    def __init__(self, agent_name):
        # 村人と同じ
        super().__init__(agent_name)
        # 占い記録
        self.result = []

    def initialize(self, base_info, diff_data, game_setting):
        # 村人と同じ
        super().initialize(base_info, diff_data, game_setting)
        # 偽霊媒結果の初期化
        self.result = []
        # 偽占い結果の初期化
        self.result_seer = []
        # ステルスモード
        self.stealth = True
        # 前日の襲撃は成功したか
        self.attack_success = True
        self.whisper_turn = 0

        #狼仲間の取得
        self.wolfs=set()
        print(self.base_info["roleMap"])
        for key,val in self.base_info["roleMap"].items():
            if val=="WEREWOLF":
                self.wolfs.add(int(key))
        print(self.wolfs)

    def update(self, base_info, diff_data, request):
        # 村人と同じ
        super().update(base_info, diff_data, request)

        # 襲撃結果を受け取る
        if request == "DAILY_INITIALIZE":
            for i in range(diff_data.shape[0]):
                if diff_data["type"][i] == "attack":
                    print(diff_data["text"][i])
                    if "false" in diff_data["text"][i]:
                        self.attack_success = False
                    else:
                        self.attack_success = True
        print(self.result)

        # 狼特殊処理
        # 占いと霊媒を引いたことにする
        # 生存者の確認
        self.alive = []
        for i in range(self.player_size):
            # 1-origin
            i += 1
            if self.base_info["statusMap"][str(i)] == "ALIVE":
                self.alive.append(str(i))
        if request == "DAILY_INITIALIZE":
            if self.base_info["day"] != 0:
                # ２日目までは夜は白を引いたことにする
                if self.base_info["day"] != 0:
                    print("WOLF DAY1 DATA", diff_data)

                    # グレーなしの場合
                    cand = self.rfestimator.decide_action("vote")
                    wolf_est = int(self.predictor.role_est("VILLAGER", cand))

                    # 可能ならグレーから選択
                    if len(self.greys) > 0:
                        print("GREY CHOSEN")
                        print(self.greys)
                        wolf_est = int(self.predictor.role_est("VILLAGER", list(
                            self.greys - {str(int(self.base_info["agentIdx"]))})))

                    state_seer = 'DIVINED Agent[' + \
                        "{0:02d}".format(wolf_est) + '] ' + 'HUMAN'
                    self.result_seer.append(state_seer)

                    state = 'IDENTIFIED Agent[' + \
                        "{0:02d}".format(wolf_est) + '] ' + 'HUMAN'
                    self.result.append(state)
                # 3日目夜は黒を引いたことにする
                else:
                    # グレーなしの場合
                    cand = set(self.alive)-{str(int(self.base_info["agentIdx"]))}
                    wolf_est = int(self.predictor.role_est("WEREWOLF", cand))

                    # 可能ならグレーから選択
                    if len(self.greys) > 0:
                        print("GREY CHOSEN")
                        print(self.greys)
                        wolf_est = int(self.predictor.role_est("WEREWOLF", list(
                            self.greys - {str(int(self.base_info["agentIdx"]))})))

                    state_seer = 'DIVINED Agent[' + \
                        "{0:02d}".format(wolf_est) + '] ' + 'HUMAN'
                    self.result_seer.append(state_seer)

                    state = 'IDENTIFIED Agent[' + \
                        "{0:02d}".format(wolf_est) + '] ' + 'WEREWOLF'
                    self.result.append(state)

    def dayStart(self):
        # 村人と同じ
        super().dayStart()
        self.whisper_turn = 0
        return None

    def talk(self):
        """
        狼のCO戦略
        ステルス？
        """
        # 村人としての発言
        ret_as_villager = super().talk()

        # ステルス解除、カミングアウト
        # は吊られそうにならない限り行わないことにする
        if self.stealth == True and False:
            self.stealth = False
            return cb.comingout(self.base_info['agentIdx'], "MEDIUM")

        # 吊られそうならカミングアウト
        print("HATRED LIST:", self.hate_who)
        if self.stealth and isMostFrequent(self.hate_who, int(self.base_info["agentIdx"])):
            self.stealth = False
            if self.player_size == 5:
                return cb.comingout(self.base_info["agentIdx"], "SEER")
            return cb.comingout(self.base_info["agentIdx"], "SEER")

        # 5人村なら初手占いCO
        #はしない
        if self.stealth and self.player_size == 5 and False:
            self.stealth = False
            return cb.comingout(self.base_info["agentIdx"], "SEER")

        if self.stealth == False and len(self.result_seer) > 0 and self.player_size == 5:
            # 最後の結果を取り出してdivined発言
            last_res = self.result_seer.pop(0)
            print(last_res)
            return last_res

        # 何もなければOver
        return ret_as_villager

    def vote(self):
        # 村人と同じ
        # 狼を回避するようにしたほうが良いだろう
        return super().vote()

    def whisper(self):
        """
        秘密会話
        攻撃したい先を言って、一致しなかったら相手に合わせる

        襲撃先選択
        1.最も占いらしいやつ
        2.昨日の護衛結果が失敗だったら最も狩人らしいやつ
        """
        self.whisper_turn += 1

        # 生存者の確認
        self.alive = []
        for i in range(self.player_size):
            # 1-origin
            i += 1
            if self.base_info["statusMap"][str(i)] == "ALIVE":
                self.alive.append(str(i))

        if self.base_info["day"]==0:
            if self.whisper_turn==1:
                return cb.comingout(self.base_info["agentIdx"],"VILLAGER")
            else:
                return cb.over()
        # 候補者セット
        #cand = set(self.alive)-{str(int(self.base_info["agentIdx"]))}
        cand = self.rfestimator.decide_action("attack")
        print(set([self.base_info["agentIdx"]]),self.rfestimator.rec.game_info["dead"],self.wolfs)
        cand = cand - set([self.base_info["agentIdx"]])-self.rfestimator.rec.game_info["dead"]-self.wolfs
        # 襲撃先の吟味
        role = "SEER"
        if self.attack_success:
            self.todays_vote = int(self.predictor.role_est("SEER", cand))
            if self.whisper_turn == 1 and self.base_info["day"]<3:
                return cb.estimate(self.todays_vote, role)
            if self.whisper_turn == 2:
                return cb.attack(self.todays_vote)
        else:
            # 昨日失敗してたら狩人がまだ生きてるということなので
            role = "BODYGUARD"
            self.todays_vote = int(self.predictor.role_est("BODYGUARD", cand)) 
            if self.whisper_turn==1 and self.base_info["day"]<3:
                return cb.estimate(self.todays_vote, role)
            if self.whisper_turn==2:
                return cb.attack(self.todays_vote)

        return cb.over()

    def attack(self):
        """
        宣言した襲撃先をやる
        """
        self.whisper()
        # とりあえず真占いっぽいやつ
        # self.alive.remove(str(self.base_info["agentIdx"]))
        #seer_est = self.predictor.role_est("SEER", self.alive)

        return self.todays_vote

    def finish(self):
        return None


class SampleAgent(object):

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        # print(base_info)
        # print(diff_data)

    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        # print(base_info)
        # print(diff_data)

    def dayStart(self):
        return None

    def talk(self):
        return cb.over()

    def whisper(self):
        return cb.over()

    def vote(self):
        return self.base_info['agentIdx']

    def attack(self):
        return self.base_info['agentIdx']

    def divine(self):
        return self.base_info['agentIdx']

    def guard(self):
        return self.base_info['agentIdx']

    def finish(self):
        return None


agent = Agent(myname)


# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
