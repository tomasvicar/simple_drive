from train import train
# from test_ import test_
from config import Config


config = Config()


model_name = train(config)

# auc,acc = test_(model_name)






