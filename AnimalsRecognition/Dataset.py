import json


def makeLabelsDict(labels):
    labelsDict = {}
    state = 0
    for label in labels:
        labelsDict[label] = state
        state += 1
    return labelsDict


class Dataset():

    def __init__(self, configLocation):
        self.configLocation = configLocation
        with open(configLocation) as file:
            self.config = json.load(file)

        self.root = self.config["root"]
        self.labels = self.config["labels"]
        self.numLabels = len(self.labels)
        self.labelsDict = makeLabelsDict(self.labels)
        return

    def oneHot(self, y):
        y_oneHot = [0] * self.numLabels
        y_oneHot[self.labelsDict[y]] = 1
        return y_oneHot

    def processImage(self, image):
        return image

    def makeData(self):
        image, output = None, None  # Todo 1 get the data
        image = self.processImage(self, image)
        output = self.oneHot(output)
        return image, output

    def makeBatchData(self, batchSize=100):
        batch = {"images": [], "outputs": []}
        for _ in range(batchSize):
            image, output = self.makeData()
            batch["images"].append(image)
            batch["outputs"].append(output)
        return batch

    def getClass(self, oneHotClass):
        state = oneHotClass.index(oneHotClass.(max(oneHotClass))) + 1 # it return the index of the element that has max value
        for key in self.labelsDict:
            value = self.labelsDict[key]
            if value == state:
                return key
        raise Exception("Invalid Output One Hot Vector")
