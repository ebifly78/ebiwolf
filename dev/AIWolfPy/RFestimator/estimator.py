from sklearn import ensemble

import collections
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
from . import resources
from . import recognizer
from pprint import pprint

import aiwolfpy.contentbuilder as cb
# import AttrDict


class CalupsAI(object):
    def __init__(self):
        self.rec = recognizer.Recognizer()
        # RF学習済みモデルの読み込み
        self.clf = dict(
            vote=resources.trained_model_vote,
            divine=resources.trained_model_divine,
            attack=resources.trained_model_attack,
            guard=resources.trained_model_guard)
        self.role_factor = resources.role_factor

        self.base_info = None

        self.info = Info()

    def initialize(self, myID, myrole):
        self.info.role = myrole
        self.info.agentID = int(myID)
        self.info.agentIDstr = str(myID)
        self.rec = recognizer.Recognizer()

    def update(self, base_info, diff_info):
        """
        updateが呼ばれるたびに実行
        infoを更新する
        """
        self.base_info = base_info
        self.rec.update_info(diff_info)
        return

    def decide_action(self, mode):
        """
        recに保存された現在の状況を基にclfを動かし，行動を決定する
        VOTEと特殊行動時のみ．
        """
        self.rec.update_game_info()

            #
            # ほんとれ霊媒判定も必要？
            #
            #
            # 狼？
            #
        # X組み立て
        X_dict = dict(
            day=self.rec.day,
            turn=-1,
            role=self.info.role,
            game_info=self.rec.zip_game_info()
        )
        # dictを平らに
        X_flat = flatten_dict(X_dict)
        # Pandas経由で適切な形のベクトルに
        df = pd.DataFrame([X_flat])
        df.role, dump = pd.factorize(df.role)
        # X, dump, dump, dump = train_test_split(df, [0], test_size=0)
        X = df
        if mode in ["vote", "divine", "attack", "guard"]:

            # 予測実行！
            # TypeError: '<' not supported between instances of 'NoneType' and 'int'
            # モデルかXの形がおかしい？
            clf = self.clf[mode]
            print(mode)
            print(X_dict)
            print("feature,output:", clf.n_features_, clf.n_outputs_)
            #try:
            proba = clf.predict_proba(X)
            proba=proba[0]
            #except Exception:
            #    print("Nonetype < int error occured. toriaezu return ha all")
            #    return set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
            #print(mode, "PROBA:", proba)
            # 評価値の高い順に行動IDを並べる
            print(proba)
            action_sorted = np.array(proba).argsort()[::-1]
            print(action_sorted)
            
            # print(X)
            # 上から実行していく
            actions = clf.classes_[action_sorted]
            print(actions)
            # print("action_candidate;",actions[0])
            # print(self.rec.agent_info)
            # pprint(self.rec.agent_info)
            for i, action in enumerate(actions):
                print(action)
                # print(i, "th trial")
                #print(".", end="")
                ret = self.parse_action(action, mode)
                #print("decided action:", ret)
                if ret is not None:
                    return ret
            else:
                print("CAUTION!!!!!!all of actions were failed. something wrong.")
                return set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])-set([self.info.agentID])

        else:
            print("mode:", mode, "ignored.")
            return cb.over()

    def parse_action(self, action_str, mode):
        """
        actionを受け取る．
        actがmodeに対して適切であることを確認した上で，
        条件に一致するtargetIDを返す．
        """
        # act=action_str[0]
        try:
            dic = ast.literal_eval(action_str)
            # print(dic)
            # print(act)
            # print(dic)
            act = dic[0]
            target_class = dic[1]
            # print(act)
            # print(target_class)
        except ValueError:
            act = action_str
            print(act, "(cant be hapen?)")

        #print(mode, act)

        if mode == act.lower():
            targetID = self.search_valid_target(target_class)
            if targetID == None:
                # print("proper target not found. target_class=", target_class)
                return None
            return targetID
        else:
            print("mode-act does not match.")
            print(action_str)
            print(mode)

        return None

    def search_valid_target(self, target_class):
        valid_agents = set()
        # print(target_class)
        # for agentID in range(1, 15):
        #print("===", agentID, self.rec.agent_info[agentID])
        for agentID in range(1, 15):
            if self.rec.agent_info[agentID]["is_alive"] is not "alive":
                continue
            if agentID == self.info.agentID:
                continue
            #print("===")
            #print(agentID, self.rec.agent_class(agentID))
            # print(id(self.rec.agent_class(agentID)))
            if self.rec.agent_class(agentID) == target_class:
                valid_agents.add(agentID)
        print(len(valid_agents), end="\t")
        if not valid_agents:
            return None
        else:
            #print(target_class)
            return valid_agents


class Info():
    def __init__(self,):
        self.role = "VILLAGER"
        self.agentID = 0
        self.divine_info = dict()
        self.identify_info = dict()
        self.guard_info = dict()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


a = {'day': 1,
     'turn': 3,
     'role': 'WEREWOLF',
     'game_info': {'COseer': 3,
                   'COmed': 0,
                   'divined_black': 1,
                   'divined_white': 2,
                   'identified_black': 0,
                   'identified_white': 0,
                   'dead': 0,
                   'wolf': 0},
     'action': ('ESTIMATE_white',
                {'CO': None, 'divined': 'none', 'status': True})}

# c=CalupsAI()
# c.decide_action()
