from numpy import *
import operator

def createDataSet():
    group = array([ [1.0, 2.0], [1.0,4.0], [4.0, 1.0], [4.0, 2.0] ])
    labels = ['Action', 'Action', 'Romantic', 'Romantic']
    return group, labels

def calcDistance(inX, dataSet, labels, k):
    # shape is a tuple that gives dimensions of the array
    # shape[0] returns the number of rows, here will return 4
    # dataSet = [[1.0, 2.0], [1.0,4.0], [4.0, 1.0], [4.0, 2.0]]
    dataSetSize = dataSet.shape[0]  # dataSetSize = 4

    # numpy.tile(A, reps): construct an array by repeating A the number of times given by reps
    # if reps has length d, the result will have dimension of max(d, A.ndim)
    # tile(inX, (dataSetSize,1)) will return [[2,3],
    #[2,3],
    #[2,3],
    #[2,3]]
    # diffMat = [[2-1,3-4],[2-1,3-2][2-4,3-1][2-4,3-2]]
    # diffMat is [ [1, 1], [1, -1], [-2, 2], [-2, 1] ]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # **2 means square
    sqDiffMat = diffMat ** 2

    # sqDistances = x^2 + y^2
    # Distances = [[1^2+1^2],
    #[1^2+(-1)^2],
    #[(-2)^2+2^2],
    #[(-2)^2+1^2]].
    sqDistances = sqDiffMat.sum(axis=1)
    # distance is equal to the square root of the sum of the squares of the coordinates
    # distance = [2, 2, 8, 5]
    distances = sqDistances ** 0.5

    # numpy.argsort() returns the indices that would sort an array
    # here returns [0 1 3 2]
    sortedDistIndices = distances.argsort()
    return sortedDistIndices

def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    classCount = {}
    # classCount maps the lables to occurance counts i.e taking the occurance of each element in the class



    # iterate k times from the closest item . Keeping track of how many times those lables appear.
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        # increase +1 on the selected label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # classCount dictionary : {'Action': 2, 'Romantic': 1}
    # sort ClassCount Descending order

    return sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)


def classify0(inX, dataSet, labels, k):
    # calculate the distance between inX and the current point
    sortedDistIndices = calcDistance(inX, dataSet, labels, k)
    # take k items with lowest distances to inX and find the majority class among k items
    sortedClassCount = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
    # sortedClassCount is now [('Action', 2)], therefore return Action
    return sortedClassCount[0][0]


group, labels = createDataSet()
result = classify0([2.0, 3.0], group, labels,3)
print (result)                    # this is a python 2.7 code , So if you run it in python 3+ then you need to put "" to run it .
