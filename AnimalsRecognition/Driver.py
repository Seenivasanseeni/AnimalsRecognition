
from . import Model,Dataset

import numpy as np

import matplotlib.pyplot as plt

def train(Mod,Ds,num=10):

    for i in range(num):
        batch=Ds.makeBatchData(50)
        acc,lo=Mod.train(batch["images"],batch["outputs"])
        print("TimeStep %d accuracy %f Loss %f" %(i,acc,lo))
    return

def visualize(Mod,Ds):
    batch=Ds.makeBatchData(batchSize=1,train=False)
    units=Mod.visualize(batch["images"])
    print(np.shape(units))
    return

def main():
    Mod=Model.Model()
    Mod.createCompGraph()
    Mod.intializeModel()
    Ds=Dataset.Dataset(configLocation="Conf/dataset.json")
    train(Mod,Ds,num=500)
    visualize(Mod,Ds)
    return



if __name__ == '__main__':
    main()
