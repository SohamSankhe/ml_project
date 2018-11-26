import numpy as np
import math
import random

def getPartitions(x, k):
    rows, _ = np.shape(x)
    partList = []
    segSize = math.floor((rows + 1) / k)
    rowCtr = 0
    #while rowCtr < rows:
    for i in range(k):
        xPart = None
        if (rowCtr + segSize <= rows) and (partList.__len__() != (k - 1)):
            xPart = x[rowCtr:rowCtr + segSize, :]
            rowCtr += segSize
        else:
            xPart = x[rowCtr:, :]
            rowCtr += segSize
        partList.append(xPart)
    return partList


def getPartitionsList(x, k):
    rows = x.__len__()
    partList = []
    segSize = math.floor((rows + 1) / k)
    #print('segSize: ', segSize)
    rowCtr = 0
    #while rowCtr < rows:
    for i in range(k):
        xPart = None
        #if rowCtr + segSize <= rows:
        if (rowCtr + segSize <= rows) and (partList.__len__() != (k - 1)):
            #print(partList.__len__())
            xPart = x[rowCtr: rowCtr + segSize]
            rowCtr += segSize
        else:
            xPart = x[rowCtr:]
            rowCtr += segSize
        partList.append(xPart)

    return partList

#x = np.arange(16)
#getPartitionsList(x, 6)

def divideListForTrnTest(lst, trainingLimit):

    trainingLimit = lst.__len__() - int(lst.__len__() * trainingLimit)
    print('trainingLimit: ', trainingLimit)

    trainingList = lst[:trainingLimit]
    testingList = lst[trainingLimit:]

    print('np.shape(trainingList): ', np.shape(trainingList))
    print('np.shape(testingList): ', np.shape(testingList))
    # print(trainingList)
    # print(testingList)

    return trainingList, testingList


def getCommaSepForm(lst):
    'Get elements in comma separated form'

    csString = "\'"
    csString += "\',\'".join(map(str, lst))
    csString += "\'"

    return csString


def getTargetVariables(oldY):
    'Convert Y from 0,1,-1 to -1,1'

    Y = []
    for i in range(0, oldY.__len__()):
        if (oldY[i] == 0) or (oldY[i] == -1):
            Y.append(0)
        else:
            Y.append(1)
    return Y

def getRandomClassRes(yMatchClassTest):

    # random for reference
    randomClass = []
    items = [1, 0, -1] # pick among these

    for i in range(yMatchClassTest.__len__()):
        randVal = random.choice(items)
        randomClass.append(randVal)

    correctCtr = 0
    for i in range(yMatchClassTest.__len__()):
        if randomClass[i] == yMatchClassTest[i]:
            correctCtr += 1

    print(randomClass)

    correctCtr = correctCtr / yMatchClassTest.__len__()
    print('Random func acc: ', correctCtr)

    return correctCtr*100
