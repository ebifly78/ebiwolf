import numpy as np
import pandas as pd


def int2agent(i):
    return "Agent["+'{:0=2}'.format(i)+"]"


def list2protocol(a, subject="VOTE", logic="OR"):
    """

    """
    a = sorted(list(a))
    a = list(map(lambda x: "(VOTE " + int2agent(x) + ") ", list(a)))
    logic += " "
    for i in a:
        logic += i
    return logic


#ret = list2protocol([1, 2, 10, 4], subject="VOTE", logic="OR")
# print(ret)
#print(np.random.choice([1, 2, 3, 5]))
