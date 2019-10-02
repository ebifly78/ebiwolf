import csv
import glob


ALL_ROLE    = 0
WEREWOLF    = 1
POSSESSED   = 2
VILLAGER    = 3
SEER        = 4
MEDIUM      = 5
BODYGURD    = 6
HUMAN       = 7
WOLF        = 8
role_name = ['ALL_ROLE', 'WEREWOLF', 'POSSESSED', 'VILLAGER', 'SEER', 'MEDIUM', 'BODYGURD', 'HUMAN', 'WOLF']

class agent:
    def __init__(self):
        self.name = ''
        self.role = ''
        self.id   = 0
        self.matches = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.win     = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    def register(self):
        if self.role == 'WEREWOLF':
            self.win[WEREWOLF] += 1
        elif self.role == 'POSSESSED':
            self.win[POSSESSED] += 1
        elif self.role == 'SEER':
            self.win[SEER] += 1
        elif self.role == 'MEDIUM':
            self.win[MEDIUM] += 1
        elif self.role == 'BODYGURD':
            self.win[BODYGURD] += 1
        else:
            self.win[VILLAGER] += 1


agent = agent()
agent.name = 'ebifly'

wolf_id = ['','','']
wolfcount = 0

atari = [0,0,0,0,0,0,0,0,0]
hazure = [0,0,0,0,0,0,0,0,0]

roles = ['','','','','','','','','','','','','','','','']

filecount = 0

i = 0
# file_list = glob.glob('../log/test5/*.log')
filenum = 2
file_list = glob.glob('../../Server/AIWolf-ver0.5.6/log/file'+'{}'.format(filenum)+'/*.log')

for files in file_list:
    file = open(file_list[i], 'r')
    reader = csv.reader(file)
    wolfcount = 0
    filecount += 1
    for row in reader:

        # get myagent role
        if row[0] == '0' and row[1] == 'status':
            if row[5] == agent.name:
                agent.role = row[3]
                agent.id = row[2]
                if agent.role == 'WEREWOLF':
                    agent.matches[WEREWOLF] += 1
                elif agent.role == 'POSSESSED':
                    agent.matches[POSSESSED] += 1
                elif agent.role == 'SEER':
                    agent.matches[SEER] += 1
                elif agent.role == 'MEDIUM':
                    agent.matches[MEDIUM] += 1
                elif agent.role == 'BODYGURD':
                    agent.matches[BODYGURD] += 1
                else:
                    agent.matches[VILLAGER] += 1

        # get myagent win or lose
        if row[1] == 'result':
            if row[4] == agent.role:
                agent.register()

        # get werewolf
        if row[0] == '0' and row[1] == 'status':
            if row[3] == 'WEREWOLF':
                wolf_id[wolfcount] = row[2]
                wolfcount += 1
                
        
        # get agent role
        if row[0] == '0' and row[1] == 'status':
            roles[int(row[2])] = row[3]
        
        # check estimate
        if row[1] == 'vote' and row[2] == agent.id and agent.role != 'WEREWOLF' and agent.role != 'POSSESSED':
            if row[3] == wolf_id[0] or row[3] == wolf_id[1] or row[3] == wolf_id[2]:
                atari[int(row[0]) - 1] += 1
            else:
                hazure[int(row[0]) - 1] += 1


    file.close
    i += 1

for i in range(1,2):
    agent.win[WOLF] += agent.win[i]
    agent.matches[WOLF] += agent.matches[i]
for i in range(3,6):
    agent.win[HUMAN] += agent.win[i]
    agent.matches[HUMAN] += agent.matches[i]
agent.win[ALL_ROLE] = agent.win[WOLF] + agent.win[HUMAN]
agent.matches[ALL_ROLE] = agent.matches[WOLF] + agent.matches[HUMAN]

print('file'+'{}'.format(filenum))
print(str(filecount)+'files')

for i in range(9):
    try:
        print(role_name[i]+' {}, {:.3f}'.format(agent.matches[i], agent.win[i]/agent.matches[i]))
    except ZeroDivisionError:
        print(role_name[i]+' no_matches')


for i in range(1,5):
    try:
        print('vote_day'+str(i)+' {:.3f}'.format(atari[i-1]/(atari[i-1]+hazure[i-1])))
    except ZeroDivisionError:
        pass