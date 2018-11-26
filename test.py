import numpy as np
import pandas as pd
import sys
'''
x = np.arange(16).reshape(4, 4)
x = np.mat(x)
print(type(x))
print(x)

x1 = np.squeeze(np.asarray(x[:,0]))
print(type(x1))
print(x1)
print(np.shape(x1))

#np.squeeze(np.asarray(M))
'''

# partition matrix for k fold
'''
x = np.arange(32).reshape(16, 2)
print(x)
k = 6
rows, _ = np.shape(x)
partList = []
import math
segSize = math.floor((rows + 1)/k)
print(segSize)
rowCtr = 0
for i in range(k):
#while rowCtr < rows:
    xPart = None
    if (rowCtr + segSize <= rows) and (partList.__len__() != (k - 1)):
        xPart = x[rowCtr:rowCtr+segSize,:]
        rowCtr += segSize
    else:
        xPart = x[rowCtr:,:]
        rowCtr += segSize
    partList.append(xPart)

for p in partList:
    print(p, '\n')
'''

def method1(a,b):
    print(a + b)

def method2(method, a,b):
    method(a,b)

method2(method1, 2,3)

sys.exit(1)
'''
resultTuple = (1,0,-1) # 1- win, 0 - draw, -1 - loss

for r in resultTuple:
    print(r)

matchIdCol = ['829513','829513','829514','829514','829515','829515','829516','829516']
teamIdCol = [13,162,184,12,12,31,26,18]
yTestReg = [0]*8
#teamRatings = [7, 6, 6, 7, 6.7, 6.6, 7.28, 6.76]
teamRatings = [1,2,3,4,5,6,7,8]

teamRatings = pd.DataFrame({'match_id':matchIdCol,'team_id':teamIdCol, 'actual_rating':yTestReg, \
                        'predicted_rating': teamRatings})

print(teamRatings)
yTestReg = [0]*4
teamIdCol1 = [13,184,12,26]
teamIdCol2 = [162,12,31,18]
matchIdCol = ['829513','829514','829515','829516']
matchRatings = pd.DataFrame({'match_id':matchIdCol,'home_team_id':teamIdCol1,'away_team_id':teamIdCol2,\
                             'home_team_rating':yTestReg, \
                        'away_team_rating': yTestReg, 'blah': yTestReg, 'otherfeatures': yTestReg})


pd.set_option('display.expand_frame_repr', False)

print(matchRatings, '\n')

print('-- ',teamRatings.loc[(teamRatings['match_id'] == '829515') & (teamRatings['team_id'] == 12)])
print(type(teamRatings.loc[(teamRatings['match_id'] == '829515') & (teamRatings['team_id'] == 12)]))
df = teamRatings.loc[(teamRatings['match_id'] == '829515') & (teamRatings['team_id'] == 12)]
print('--- ',df['predicted_rating'].values[0])



i = 0
#print('---   ', teamRatings.loc[(teamRatings['match_id'] == matchRatings['match_id'][mRowCtr]) &
#                            (teamRatings['team_id'] == int(matchRatings['home_team_id'][mRowCtr]))])
mRowCtr = 0
a = matchRatings['match_id'][mRowCtr]
b = int(matchRatings['home_team_id'][mRowCtr])



mRowCtr = 0
while mRowCtr < (matchRatings.shape[0]):

    l1 = matchRatings['match_id'][mRowCtr]
    r1 = int(matchRatings['home_team_id'][mRowCtr])
    print('l1, r1', l1, r1)
    dfLhs = teamRatings.loc[(teamRatings['match_id'] == l1) & (teamRatings['team_id'] == r1)]
    print('dfLhs ', dfLhs)
    print('dfLhs ', dfLhs['predicted_rating'].values[0])
    #matchRatings.set_value(mRowCtr, 'home_team_rating', teamRatings['predicted_rating'][i])
    matchRatings.set_value(mRowCtr, 'home_team_rating', dfLhs['predicted_rating'].values[0])
    #i += 1

    l2 = matchRatings['match_id'][mRowCtr]
    r2 = int(matchRatings['away_team_id'][mRowCtr])
    dfRhs = teamRatings.loc[(teamRatings['match_id'] == l2) & (teamRatings['team_id'] == r2)]

    matchRatings.set_value(mRowCtr, 'away_team_rating', dfRhs['predicted_rating'].values[0])
    mRowCtr += 1
    #i += 1

print('\n',matchRatings)
'''