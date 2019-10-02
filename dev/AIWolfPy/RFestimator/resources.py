import os
import pickle
"""
学習済みモデルのときに作った
役職ーIDリストを読み込む
"""
excpath = os.path.dirname(os.path.abspath(__file__))
role_factor = []
trained_model = None

with open(os.path.join(excpath, "resources/role_factor.pickle"), "rb") as f:
    role_factor = pickle.load(f)
with open(os.path.join(excpath, "resources/trained_model_15_vote.pickle"), "rb") as f:
    trained_model_vote = pickle.load(f)
    trained_model_vote.n_jobs = 1
with open(os.path.join(excpath, "resources/trained_model_15_divine.pickle"), "rb") as f:
    trained_model_divine = pickle.load(f)
    trained_model_divine.n_jobs = 1
with open(os.path.join(excpath, "resources/trained_model_15_attack.pickle"), "rb") as f:
    trained_model_attack = pickle.load(f)
    trained_model_attack.n_jobs = 1
with open(os.path.join(excpath, "resources/trained_model_15_guard.pickle"), "rb") as f:
    trained_model_guard = pickle.load(f)
    trained_model_guard.n_jobs = 1
