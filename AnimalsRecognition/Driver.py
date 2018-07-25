
from . import Model,Dataset

import numpy as np

import matplotlib.pyplot as plt

def train(Mod,Ds,num=10):

    for i in range(num):
        batch=Ds.makeBatchData(100)
        acc,lo=Mod.train(batch["images"],batch["outputs"])
        print("TimeStep %d accuracy %f Loss %f" %(i,acc,lo))
    return

def modTrain(Mod,Ds,num=10):
    batch=Ds.makeBatchData(1000)
    import  numpy as np
    print(np.shape(batch["images"]))
    for _ in range (num):
        print("TRAIN")
        for i in range(10):

            acc, lo = Mod.train(batch["images"][i * 100:(i + 1) * 100], batch["outputs"][i * 100:(i + 1) * 100])
            print("Timestep %d accuracy %f loss %f " %(i, acc, lo))


    print("SEEN DATA")
    for i in range(10):
        acc, lo = Mod.train(batch["images"][i * 100:(i + 1) * 100], batch["outputs"][i * 100:(i + 1) * 100])
        print("Timestep %d accuracy %f loss %f "%(i, acc, lo))

    return

def visualize(Mod,Ds):
    batch=Ds.makeBatchData(batchSize=1,train=False)
    units=Mod.visualize(batch["images"])
    print("Shape is ",np.shape(units))
    return

def clearTensorFiles():
    import  os
    if "logs" in os.listdir("."):
        #os.removedirs("logs")
        print("Removed logs dir")
    return

def main():
    clearTensorFiles()
    Mod=Model.Model()
    Mod.createCompGraph()
    Mod.intializeModel()
    Ds=Dataset.Dataset(configLocation="Conf/dataset.json")
    modTrain(Mod,Ds,num=10)
    visualize(Mod,Ds)
    return



if __name__ == '__main__':
    main()
