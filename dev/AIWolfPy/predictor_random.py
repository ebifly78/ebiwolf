import numpy as np


class Predicter(object):
    """
    いろんな確率を考える予測機
    """

    def __init__(self, players_name):
        print("Random predictor succsessfully loaded")
        self.agent_info = None

    def update(self, diff_data):
        return

    def update_agent_info(self, agent_info):
        self.agent_info = agent_info
        return

    def role_est(self, role, still_alive):
        """
        役職持ちへの投票回数で狼らしさを判定する
        """
        if self.agent_info == None:
            return self.role_est_random(role, still_alive)
        if len(still_alive) == 0:
            print("caution, role_est stacked because of len==0 still alive.")
            return "1"
        cands = []
        for agent in list(still_alive):
            count = self.agent_info[int(agent)]["vote_COer_count"]
            cands.append((agent, count))
        cands = cands[::-1]
        cands = sorted(cands, key=lambda x: x[1])

        if role in ["WEREWOLF"]:
            print("ROLE_EST WEREWOLF", still_alive, "->", cands[-1])
            return cands[-1][0]
        else:
            print("ROLE_EST WEREWOLF", still_alive, "->", cands[0])
            return cands[0][0]

    def role_est_random(self, role, still_alive):
        """
        最もroleらしいやつのIDを返す
        """
        a = np.random.choice(list(still_alive))
        print("ROLE_EST:", role, ":", still_alive, "->", a)
        return a
