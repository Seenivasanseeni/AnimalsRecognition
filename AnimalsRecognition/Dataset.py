import json
import os
from collections import deque
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
import  numpy as np

def makeLabelsDict(labels):
    labelsDict = {}
    state = 0
    for label in labels:
        labelsDict[label] = state
        state += 1
    return labelsDict


def makeQueue(dir):
    if(isinstance(dir,list)):
        return  deque(dir)
    Q = deque()
    for file in os.listdir(dir):
        Q.append(os.path.join(dir, file))
    return Q


def randomDataPath(Q):
    index = random.randrange(0, len(Q))
    path = Q[index]
    return path


class Dataset():

    def __init__(self, configLocation):
        self.configLocation = configLocation
        with open(configLocation) as file:
            self.config = json.load(file)

        self.root = self.config["root"]
        self.trainDataPath = os.path.join(self.root, self.config["train"])
        self.testDataPath = os.path.join(self.root, self.config["test"])
        self.trainDataQueue = makeQueue(self.trainDataPath)
        self.testDataQueue = makeQueue(self.testDataPath)
        self.labels = self.config["labels"]
        self.numLabels = len(self.labels)
        self.labelsDict = makeLabelsDict(self.labels)

        # images regarding
        self.imageSize = self.config["imageSize"]
        self.channels = self.config["channels"]

        return

    def oneHot(self, y):
        y_oneHot = [0] * self.numLabels
        y_oneHot[self.labelsDict[y]] = 1
        return y_oneHot

    def processImage(self, image):
        image = resize(image, output_shape=[self.imageSize, self.imageSize, self.channels])
        return image

    def getLabel(self, path):
        file = path.rsplit("/")[-1]
        label = file.split(".", 1)[0]
        return label

    def getData(self, path, train=True):
        image = plt.imread(path)
        if (not train):
            return image

        label = self.getLabel(path)
        return image, label

    def makeData(self, train=True):
        if (not train):  # make test Data
            path = randomDataPath(self.testDataQueue)
            image = self.getData(path, train)
            image = self.processImage(image)
            return image

        path = randomDataPath(self.trainDataQueue)

        image, output = self.getData(path)

        image = self.processImage(image)
        output = self.oneHot(output)
        return image, output

    def makeBatchData(self, batchSize=100,train=True):
        batch = {"images": [], "outputs": []}
        if (not train):
            batch = {"images": []}

        for _ in range(batchSize):
            if (train):
                image, output = self.makeData()
                batch["images"].append(image)
                batch["outputs"].append(output)
            else:
                image = self.makeData(train=False)
                batch["images"].append(image)

        return batch

    def getClass(self, oneHotClass):
        state = oneHotClass.index(max(oneHotClass))  # it return the index of the element that has max value
        for key in self.labelsDict:
            value = self.labelsDict[key]
            if value == state:
                return key
        raise Exception("Invalid Output One Hot Vector")



def train_test_split(paths):
    index=int(len(paths) * 75/100)
    return paths[:index],paths[index:]

def validFileFormat(file):
    parts=file.split(".")
    try:
        temp=int(parts[0]) #invalid file will raise exception
    except:
        return False
    return True

def filterFiles(files):
    return [file for file in files if validFileFormat(file)]


class MicrosoftDataset():

    def __init__(self, configLocation):
        self.configLocation = configLocation
        with open(configLocation) as file:
            self.config = json.load(file)

        self.root = os.path.join(os.path.expanduser("~"),
            self.config["root"])

        self.labels=os.listdir(self.root)
        self.paths=[]

        for label in self.labels:
            dirPath=os.path.join(self.root,label)
            for file in os.listdir(dirPath):
                if(validFileFormat(file)):
                    self.paths.append(os.path.join(dirPath,file))

        random.shuffle(self.paths)
        self.trainPaths,self.testPaths=train_test_split(self.paths)
        self.trainDataQueue=makeQueue(self.trainPaths)
        self.testDataQueue=makeQueue(self.testPaths)

        self.numLabels = len(self.labels)
        self.labelsDict = makeLabelsDict(self.labels)

        # images regarding
        self.imageSize = self.config["imageSize"]
        self.channels = self.config["channels"]

        return

    def oneHot(self, y):
        y_oneHot = [0] * self.numLabels
        y_oneHot[self.labelsDict[y]] = 1
        return y_oneHot

    def processImage(self, image):
        image = resize(image, output_shape=[self.imageSize, self.imageSize, self.channels])
        return image

    def getLabel(self, path):
        file = path.rsplit("/")[-2]
        label = file
        return label

    def getData(self, path, train=True):
        image = plt.imread(path)
        if (not train):
            return image

        label = self.getLabel(path)
        return image, label

    def makeData(self, train=True):
        if (not train):  # make test Data
            path = randomDataPath(self.testDataQueue)
            image = self.getData(path, train)
            image = self.processImage(image)
            return image

        path = randomDataPath(self.trainDataQueue)

        image, output = self.getData(path)

        image = self.processImage(image)
        output = self.oneHot(output)
        return image, output

    def makeBatchData(self, batchSize=100,train=True):
        train=True # it is needed in testing cases
        batch = {"images": [], "outputs": []}
        if (not train):
            batch = {"images": []}

        for _ in range(batchSize):
            try:
                if (train):
                    image, output = self.makeData()
                    batch["images"].append(image)
                    batch["outputs"].append(output)
                else:
                    image = self.makeData(train=False)
                    batch["images"].append(image)
            except:
                print("ERROR: exception raised when  handling ")
        return batch

    def getClass(self, oneHotClass):
        state = oneHotClass.index(max(oneHotClass))  # it return the index of the element that has max value
        for key in self.labelsDict:
            value = self.labelsDict[key]
            if value == state:
                return key
        raise Exception("Invalid Output One Hot Vector")





if(__name__=="__main__"):
    MicrosoftDataset("../Conf/dataset.json")

