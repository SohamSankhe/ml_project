import matplotlib.pyplot as plt


def plotXvsY(lst):
    xCoord = []
    yCoord = []

    for t in lst:
        xCoord.append(t[0])
        yCoord.append(t[1])
    plt.plot(xCoord, yCoord, '-o')
    plt.show()
    return



