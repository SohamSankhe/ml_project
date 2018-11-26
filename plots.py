import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, splrep, splev

def plotXvsYOld(lst,givenLabel = 'Logistic regression'):
    xCoord = []
    yCoord = []

    for t in lst:
        xCoord.append(t[0])
        yCoord.append(t[1])

    plt.title(givenLabel)
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy %')
    plt.axis([xCoord[0], xCoord[-1], 0, 100])
    plt.plot(xCoord, yCoord, '-o')
    plt.show()
    return



def plotXvsY(lst, givenLabel = 'Logistic regression'):
    x = []
    y = []

    for t in lst:
        x.append(t[0])
        y.append(t[1])

    # test
    #x = [1, 2, 3, 4, 5, 6]
    #y = [40, 51, 60, 48, 50, 49]

    tck = splrep(x, y)
    xnew = np.linspace(x[0], x[-1])
    ynew = splev(xnew, tck)
    plt.title(givenLabel)
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy %')
    plt.axis([x[0], x[-1], 0, 100])
    plt.plot(xnew, ynew)
    plt.plot(x, y, 'o')
    plt.show()

    return

def algoAcc():
    # random, log reg, svm, nb, knn
    y = [30, 68, 46, 52, 42]
    x = np.arange(5)

    fig, ax = plt.subplots()
    #ax.yaxis.set_major_formatter(formatter)
    plt.bar(x, y)
    ax.set_ylim([0, 100])
    plt.xticks(x, ('random', 'logistic reg', 'svm', 'gaussian nb', 'kNN'))
    plt.show()

    return


#plotXvsY([])


