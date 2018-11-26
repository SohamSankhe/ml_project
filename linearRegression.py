from numpy.linalg import inv
import numpy as np
from numpy import array
from sklearn.linear_model import LinearRegression
import utils

class RidgeRegResults():  # class to store findings for k fold ridge
    def __init__(self):
        self.k = 0
        self.lamb = 0
        self.testIndex = 0  # to id hold out partition
        self.sseTraining = 0
        self.sseTest = 0
        self.theta = []

    def setResults(self, k, lamb, testIndex, sseTraining, sseTest, theta):
        self.k = k
        self.lamb = lamb
        self.testIndex = testIndex
        self.sseTraining = sseTraining
        self.sseTest = sseTest
        self.theta = theta
        return

    # test method
    def printData(self):
        print('\n-----')
        print(' k = ', self.k, ' testIndex = ', self.testIndex, ' lambda = ', self.lamb, ' sseTest: ',
              self.sseTest, ' sseTraining: ', self.sseTraining)
        #print('theta: ', self.theta)
        return


def calculateError(X, theta, Y):
    estimation = X.dot(theta)

    error = []
    for i in range(0, Y.__len__()):
        er = estimation[i] - Y[i]
        error.append(er * er)

    return sum(error) / Y.__len__()

def calculateErrorSkLearn(estimation, Y):

    error = []
    for i in range(0, Y.__len__()):
        er = estimation[i] - Y[i]
        error.append(er * er)

    return sum(error) / Y.__len__()

def calculateThetaClosedForm(X, Y):
    X_Transpose = X.transpose()

    theta = inv(X_Transpose.dot(X)).dot(X_Transpose).dot(Y)

    return theta


def getGradient(X, theta, Y, learningRate):
    X_transpose = X.transpose()
    estimation = X.dot(theta)
    error = estimation - Y

    gradient = X_transpose.dot(error) / (2 * Y.__len__())
    gradient = learningRate * gradient

    return gradient



def getConvergence(newTheta, oldTheta):
    'L1 error for thetas in gradient descent'

    diff = newTheta - oldTheta

    l1Norm = 0

    for n in diff:
        l1Norm += abs(n)

    return l1Norm


def calculateThetaRidge(X, Y, lamb):
    X_Transpose = X.transpose()

    # make regularization matrix for lambda
    xCol = np.shape(X)[1]
    regMatrix = np.eye(xCol)
    regMatrix = lamb * regMatrix
    regMatrix_Transpose = regMatrix.transpose()

    theta = inv(X_Transpose.dot(X) + regMatrix_Transpose.dot(regMatrix)).dot(X_Transpose).dot(Y)

    return theta


def calculateThetaGradientDescent(X, yTraining, n):
    maxIterations = 10000
    epsilon = 0.0001

    oldTheta = []
    newTheta = np.ones(n + 1)  # starting with 1s as initial theta

    for i in range(0, maxIterations):

        oldTheta = newTheta

        newTheta = newTheta - getGradient(X, newTheta, yTraining, 0.001)

        error = getConvergence(newTheta, oldTheta)

        # print('Iteration: ', i, ' Convergence: ', error)

        if error <= epsilon:
            print('Converged')
            break

    return newTheta


def getPhi(X, n):
    rows = X.__len__()
    cols = n + 1

    phi = np.ones(shape=(rows, cols))  # all 1s
    phi[:, 1] = X  # 2nd row is X

    for ind in range(2, n + 1):
        phi[:, ind] = pow(phi[:, 1], ind)

    return phi


def ridgeRegression(xTraining, yTraining,xTest, yTest, lambdaList, kList):

    print('Ridge Regression\n')

    findings = []

    for k in kList:

        # partition as per K
        #partitionXList = np.vsplit(xTraining, k) # row wise partition

        partitionXList = utils.getPartitions(xTraining, k)
        #partitionYList = np.array_split(yTraining, k)
        partitionYList = utils.getPartitionsList(yTraining, k)

        for i in range(0, k):  # take partition i as test

            # select partition i as test set
            xTst = partitionXList[i]
            yTst = partitionYList[i]

            # rest is training
            xTrnList = [] # holds selected partitions to merge later
            xTrn = [] # merged training set
            yTrn = [] # merged test set

            for j in range(0, k):  # combine training partitions
                if i != j:
                    xTrnList.append(partitionXList[j]) # list of matrices
                    yTrn.extend(partitionYList[j])

            # get xTrn from xTrnList
            totalRows = 0
            for xPart in xTrnList:
                totalRows += np.shape(xPart)[0]

            xTrn = np.zeros((totalRows, np.shape(xTraining)[1]))

            # merge training partitions
            rowCtr = 0
            for m in xTrnList:
                noRows = np.shape(m)[0]
                xTrn[rowCtr:rowCtr+noRows, :] = m
                rowCtr += noRows

            # X, Y, xTst, yTest

            # for each hyper param, check error
            for lamb in lambdaList:
                theta = calculateThetaRidge(xTrn, yTrn, lamb)
                sseTraining = calculateError(xTrn, theta, yTrn)
                sseTest = calculateError(xTst, theta, yTst)

                # store findings in object
                resObj = RidgeRegResults()
                resObj.setResults(k, lamb, i, sseTraining, sseTest, theta)
                findings.append(resObj)


    '''
    if not not findings:
        for f in findings:
            f.printData()
    '''

    # get optimal lambda, sseTest, sseTraining, theta - sort on sseTest
    findings = sorted(findings, key=lambda linkObj: linkObj.sseTest)

    print('Findings length = ', findings.__len__())

    print('Error in ridge reg: ', findings[0].sseTest)
    print('Error in ridge reg last: ', findings[-1].sseTest)

    reg = LinearRegression().fit(xTraining, yTraining)
    reg.score(xTraining, yTraining)
    print('sklearn training: ', calculateErrorSkLearn(reg.predict(xTraining), yTraining))
    print('sklearn testing: ', calculateErrorSkLearn(reg.predict(xTest), yTest))


    return findings[0].theta, findings


def main():
    lambdaList = [0.01]
    kList = [2,3]

    x = np.arange(24).reshape(6, 4)
    #x = np.mat(x)
    #x = np.squeeze(np.asarray(x))

    y = [1, 2, 3, 4, 5, 6]
    #y = np.mat(y)
    y = array(y)
    print(np.shape(x))
    print(np.shape(y))
    print(type(x))
    print(type(y))
    # ridgeRegression(xTraining, yTraining, xTest, yTest, lambdaList, kList, nList = 0)
    print('theta: \n', ridgeRegression(x,y.transpose(),lambdaList,kList))

    return


def linearReg(xTraining, yTraining, xTest, yTest, nList = 0):
    print('Linear Regression\n')

    X = xTraining

    print('With closed form:')

    theta = calculateThetaClosedForm(X, yTraining)
    print('Theta: ', theta)

    sseTraining = calculateError(X, theta, yTraining)

    print('Error for training: ', sseTraining)

    #sseTest = calculateError(phiXTest, theta, yTest)
    sseTest = calculateError(xTest, theta, yTest)

    print('Error for testing: ', sseTest)
    print('\n')
    '''
    print('With gradient descent:')

    theta = calculateThetaGradientDescent(X, yTraining, n)

    print('Theta: ', theta)

    sseTraining = calculateError(X, theta, yTraining)

    print('Error for training: ', sseTraining)

    sseTest = calculateError(phiXTest, theta, yTest)

    print('Error for testing: ', sseTest)
    print('\n')
    '''
    return theta



# main()


'''
def main():
    # Linear regression

    # load data
    dataSet = sio.loadmat('./dataset1.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']
    n = [2, 3]

    linearReg(xTraining, yTraining, xTest, yTest, n)

    # Ridge regression

    dataSet = sio.loadmat('./dataset2.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    lambdaList = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1]
    kList = [2, 10, yTraining.__len__()]
    nList = [2, 5]

    ridgeRegression(xTraining, yTraining, xTest, yTest, lambdaList, kList, nList)

    return

'''
#main()

