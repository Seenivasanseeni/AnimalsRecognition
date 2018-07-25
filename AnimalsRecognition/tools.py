import os

def makeLogDir():
    os.makedirs("logs",exist_ok=True)
    os.makedirs("logs/1",exist_ok=True)
    return
