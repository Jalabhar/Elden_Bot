import numpy as np
import Elden_Bot_8 as Bot
import time

time.sleep(3.0)
X = np.load('EldenBrain.npy')
S = Bot.run_bot(X)
print(S)
