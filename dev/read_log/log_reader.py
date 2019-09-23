import csv
import glob


ALL_ROLE    = 0
WEREWOLF    = 1
HUMAN       = 2
VILLAGER    = 3
SEER        = 4
MEDIUM      = 5
BODYGURD    = 6
POSSESSED   = 7

class agent:
    def __init__(self):
        self.name = ''
        self.role = ''
        self.matches = [0, 0, 0, 0, 0, 0, 0, 0]
        self.win     = [0, 0, 0, 0, 0, 0, 0, 0]
    
    def register(self):
        self.win[ALL_ROLE] += 1
        if self.role == 'WEREWOLF':
            self.win[WEREWOLF] += 1
        else:
            self.win[HUMAN] += 1


agent = agent()
agent.name = 'ebifly'

i = 0
# file_list = glob.glob('../log/test5/*.log')
file_list = glob.glob('../../Server/AIWolf-ver0.5.6/log/test11/*.log')

for files in file_list:
    file = open(file_list[i], 'r')
    reader = csv.reader(file)

    for row in reader:

        # get myagent role
        if row[0] == '0' and row[1] == 'status':
            if row[5] == agent.name:
                agent.role = row[3]
                agent.matches[ALL_ROLE] += 1
                if agent.role == 'WEREWOLF':
                    agent.matches[WEREWOLF] += 1
                else:
                    agent.matches[HUMAN] += 1

        # get myagent win or lose
        if row[1] == 'result':
            if row[4] == agent.role:
                agent.register()

    file.close
    i += 1

def show_win_rate():
    if agent.matches[WEREWOLF] == 0:
        print('win_rate @ WEREWOLF = no matches')
    else:
        print('win_rate @ WEREWOLF = {}'.format(agent.win[WEREWOLF] / agent.matches[WEREWOLF] ))

    if agent.matches[HUMAN] == 0:
        print('win_rate @ VILLAGER = no matches')
    else:
        print('win_rate @ VILLAGER = {}'.format(agent.win[HUMAN] / agent.matches[HUMAN]))

    if agent.matches[ALL_ROLE] == 0:
        print('win_rate @ ALL      = no matches')
    else:
        print('win_rate @ ALL      = {}'.format(agent.win[ALL_ROLE] / agent.matches[ALL_ROLE]))

show_win_rate()
print(agent.matches[ALL_ROLE])