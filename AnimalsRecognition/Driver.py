from . import Model,Dataset


def train(Mod,Ds):

    num=1000
    for i in range(num):
        batch=Ds.makeBatchData(1000)
        acc,lo=Mod.train(batch["input"],batch["output"])
        print("TimeStep %d accuracy %f Loss %f" %(i,acc,lo))
    return




def main():
    Mod=Model.Model()
    Mod.createCompGraph()
    Mod.intializeModel()
    Ds=Dataset.Dataset(configLocation="../conf/data.json")
    train(Mod,Ds)
    return






if __name__ == '__main__':
    main()