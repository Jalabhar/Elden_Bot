import numpy as np
import Elden_Bot_Meta as Bot
import time

time.sleep(3.0)
W = np.load('MetaBrain.npy')
S = Bot.run_bot(W)
print(S)
