ind = 0
case60 = [[0] * 5 for i in range(5460)]
# case5460
for w in range(5):
    for p in range(5):
        for s in range(5):
            if w != p and p != s and s != w:
                case60[ind][w] = 1
                case60[ind][p] = 2
                case60[ind][s] = 3
                ind += 1

for i in range(60):
    print(case60[i][0], case60[i][1], case60[i][2], case60[i][3], case60[i][4])