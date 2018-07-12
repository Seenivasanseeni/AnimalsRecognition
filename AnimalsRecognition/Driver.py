from . import Model,Dataset
import matplotlib.pyplot as plt

def train(Mod,Ds):

    num=1000
    for i in range(num):
        batch=Ds.makeBatchData(10)
        plt.imshow(batch["images"][0])
        plt.show()
        acc,lo=Mod.train(batch["images"],batch["outputs"])
        print("TimeStep %d accuracy %f Loss %f" %(i,acc,lo))
    return




def main():
    Mod=Model.Model()
    Mod.createCompGraph()
    Mod.intializeModel()
    Ds=Dataset.Dataset(configLocation="Conf/dataset.json")
    train(Mod,Ds)
    return






if __name__ == '__main__':
    main()