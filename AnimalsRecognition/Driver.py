
from . import Model,Dataset


def train(Mod,Ds,num=10):

    for i in range(num):
        batch=Ds.makeBatchData(50)
        acc,lo=Mod.train(batch["images"],batch["outputs"])
        print("TimeStep %d accuracy %f Loss %f" %(i,acc,lo))
    return

def visulaize(Mod,Ds):
    batch=Ds.makeBatchData(batchSize=1,train=False)
    units=Mod.visualize(batch["images"])
    return

def main():
    Mod=Model.Model()
    Mod.createCompGraph()
    Mod.intializeModel()
    Ds=Dataset.Dataset(configLocation="Conf/dataset.json")
    train(Mod,Ds,num=10)
    return



if __name__ == '__main__':
    main()