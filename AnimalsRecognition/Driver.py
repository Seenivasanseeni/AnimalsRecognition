
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
    trainBatch=Ds.makeBatchData(1000,train=True)
    testBatch=Ds.makeBatchData(100,train=False)

    for epoch in range(num):
        for i in range(9):
            train_acc,train_lo=Mod.train(trainBatch["images"][i*(100):(i+1)*100],trainBatch["outputs"][i*100:(i+1)*100])
            test_acc,test_lo=Mod.test(testBatch["images"],testBatch["outputs"])
            print("epoch {} iter {} train loss{} train acc {} test loss {} test accu {}".format(epoch,i,train_lo,train_acc,test_lo,test_acc))

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
    Ds=Dataset.MicrosoftDataset(configLocation="Conf/dataset.json")
    modTrain(Mod,Ds,num=100)
    return



if __name__ == '__main__':
    main()
