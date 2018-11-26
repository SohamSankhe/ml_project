import utils
import numpy as np

def kFoldClassification(classifierMethod, xTraining, yTraining, xTest, yTest):
    print('-- ', np.shape(yTraining))
    findings = []
    kList = [2, 3, 10]
    for k in kList:
        partitionXList = utils.getPartitions(xTraining, k)
        partitionYList = utils.getPartitionsList(yTraining, k)
        #partitionYList = utils.getPartitions(yTraining, k)

        for i in range(0, k):  # take partition i as test
            xTst = partitionXList[i]
            yTst = partitionYList[i]

            # rest is training
            xTrnList = []  # holds selected partitions to merge later
            xTrn = []  # merged training set
            yTrn = []  # merged test set

            for j in range(0, k):  # combine training partitions
                if i != j:
                    xTrnList.append(partitionXList[j])  # list of matrices
                    yTrn.extend(partitionYList[j])
                    #yTrn.append(partitionYList[j])

            # get xTrn from xTrnList
            totalRows = 0
            for xPart in xTrnList:
                totalRows += np.shape(xPart)[0]

            xTrn = np.zeros((totalRows, np.shape(xTraining)[1]))

            # merge training partitions
            rowCtr = 0
            for m in xTrnList:
                noRows = np.shape(m)[0]
                xTrn[rowCtr:rowCtr + noRows, :] = m
                rowCtr += noRows

            if np.shape(xTrn)[0] != yTrn.__len__():
                print('-----Error: ', np.shape(xTrn)[0], yTrn.__len__())

            # call classifier
            print('----- ', np.shape(yTrn))
            _,results = classifierMethod(xTrn, yTrn, xTst, yTst)
            #t = (wList, hyperParam, acc)

            for r in results:
                lst = [k]
                lst.extend(list(r))
                findings.append(lst)


    findings = sorted(findings, key= lambda obj:obj[3], reverse = True)
    print('findings[0][2]: ', findings[0][3], ' ', findings[0][0])
    print('findings[-1][2]: ', findings[-1][3], ' ', findings[-1][0])

    wList = findings[0][1] # get best w
    #print('wList: \n', wList)
    '''
    xTest = np.squeeze(np.asarray(xTest))
    yTest = np.squeeze(np.asarray(yTest))
    acc = checkAccuracyMultiClass(xTest, yTest.transpose(), wList)
    print('Accuracy of multi class logistic on training test: ', acc * 100, '%')
    '''

    return wList, findings