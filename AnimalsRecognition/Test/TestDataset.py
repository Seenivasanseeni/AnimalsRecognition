from .. import Dataset
import matplotlib.pyplot as plt


def main():
    d=Dataset.Dataset("Conf/dataset.json")

    image,label=d.makeData()

    plt.imshow(image)
    plt.show()