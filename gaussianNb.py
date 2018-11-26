from sklearn.naive_bayes import GaussianNB
import numpy as np


def checkAccuracy(xTest, yTest, wList):

    predProbabilites = []

    for w in wList:
        pred = w.predict_proba(xTest)
        #print('w.classes_ ', w.classes_)
        #print('pred: \n', pred)
        predProbabilites.append(pred)

    # print side by side for test
    prediction = []

    #print('GNB prob predictions')
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


def getTargetVarMultiClass(oldY, res):
    Y = []
    for i in range(0, oldY.__len__()):
        if oldY[i] == res:
            Y.append(1)
        else:
            Y.append(0)
    return Y


def classifyGaussianNb(xTraining, yTraining, xTest, yTest):

    wList = []
    resultTuple = (1, 0, -1)  # 1- win, 0 - draw, -1 - loss

    for r in resultTuple:

        yTrn = getTargetVarMultiClass(yTraining, r)
        gnb = GaussianNB().fit(xTraining, yTrn)

        yTst = getTargetVarMultiClass(yTest, r)
        print('G Naive bayes accuracy training: ', gnb.score(xTest, yTst))

        #print('gnb.predict(xTest): \n', gnb.predict(xTest))
        #print(yTst)
        #print(np.shape(gnb.predict_log_proba()))

        wList.append(gnb)

    return wList