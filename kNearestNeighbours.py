from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math

def checkAccuracy(xTest, yTest, wList):

    predProbabilites = []

    for w in wList:
        pred = w[1].predict_proba(xTest)
        predProbabilites.append(pred)

    # print side by side for test
    prediction = []

    #print('Knn prob predictions')
    for i in range(predProbabilites[0].__len__()):
        #print(predProbabilites[0][i], ' - ', predProbabilites[1][i], ' - ', predProbabilites[2][i])

        if (predProbabilites[0][i][1] > predProbabilites[1][i][1]) and (predProbabilites[0][i][1] > predProbabilites[2][i][1]):
            prediction.append(1)  # win
        elif (predProbabilites[1][i][1] > predProbabilites[0][i][1]) and (predProbabilites[1][i][1] > predProbabilites[2][i][1]):
            prediction.append(0)  # draw
        else:
            prediction.append(-1)  # lose

    print('\nPredictions: ', prediction)
    print(yTest.T)

    correctCtr = 0
    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr


def checkAccuracyNew(xTest, yTest, wList):

    predProbabilites = []

    for w in wList:
        pred = w.predict_proba(xTest)
        predProbabilites.append(pred)

    # print side by side for test
    prediction = []

    #print('Knn prob predictions')
    for i in range(predProbabilites[0].__len__()):
        print(predProbabilites[0][i], ' - ', predProbabilites[1][i], ' - ', predProbabilites[2][i])

        if (predProbabilites[0][i][1] > predProbabilites[1][i][1]) and (predProbabilites[0][i][1] > predProbabilites[2][i][1]):
            prediction.append(1)  # win
        elif (predProbabilites[1][i][1] > predProbabilites[0][i][1]) and (predProbabilites[1][i][1] > predProbabilites[2][i][1]):
            prediction.append(0)  # draw
        else:
            prediction.append(-1)  # lose

    # print('\nPredictions: ', prediction)
    # print(yTest.T)

    correctCtr = 0
    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr

def getTargetVarMultiClass(oldY, res):
    Y = []
    for i in range(0, oldY.__len__()):
        if oldY[i] == res:
            Y.append(1)
        else:
            Y.append(0)
    return Y

def getkValue(len):

    k = int(math.sqrt(len))
    if k % 2 == 0:
        k = k+1

    return k

def classifyKnn(xTraining, yTraining, xTest, yTest):

    wList = []
    resultTuple = (1, 0, -1)  # 1- win, 0 - draw, -1 - loss
    sqrtK = getkValue(np.shape(xTraining)[0])

    for r in resultTuple:

        yTrn = getTargetVarMultiClass(yTraining, r)

        kList = [3, sqrtK, sqrtK - 2]
        kResults = []

        for k in kList:

            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(xTraining, yTrn)

            yTst = getTargetVarMultiClass(yTest, r)
            kAcc = neigh.score(xTest, yTst)
            print('Knn accuracy training: ', kAcc, ' - ', k)

            kRes = (k, neigh, kAcc)
            kResults.append(kRes)

        kResults = sorted(kResults, key= lambda obj: obj[2], reverse=True)

        wList.append(kResults[0])

    return wList


def classifyKnnNew(xTraining, yTraining, xTest, yTest):

    wList = []
    resultTuple = (1, 0, -1)  # 1- win, 0 - draw, -1 - loss
    sqrtK = getkValue(np.shape(xTraining)[0])
    kList = [3, 5, 7, 9, 15, 21, 25, sqrtK, sqrtK - 2]
    kResults = []

    for k in kList:
        wList = []
        for r in resultTuple:
            yTrn = getTargetVarMultiClass(yTraining, r)
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(xTraining, yTrn)

            yTst = getTargetVarMultiClass(yTest, r)
            kAcc = neigh.score(xTest, yTst)
            print('Knn accuracy training: ', kAcc, ' - ', k)
            wList.append(neigh)

        acc = checkAccuracyNew(xTest, yTest, wList)
        kRes = (wList, k , kAcc)
        kResults.append(kRes)

    kResults = sorted(kResults, key=lambda obj: obj[2], reverse=True)

    return kResults[0][0], kResults

