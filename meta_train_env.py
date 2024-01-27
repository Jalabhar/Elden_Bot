import pandas as pd
import numpy as np
import Elden_Bot_Meta as Bot
import graph_optimizer as GO


lim = [
    [1, 2, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1],
    [24, 32, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
]
n_dims = len(lim[0])
step_size = np.ones(n_dims)
step_size[6:] = 0
obj = Bot.run_bot
# a,b = opt.optimize(F,lim,n_iter=10,n_parts=20,n_keep=5)
a, b, c = GO.optimize(
    obj,
    lim,
    base_solution=None,
    n_iterations=50,
    n_parts=10,
    n_neighbors=3,
    init_mode="uniform",
    passo=step_size,
)
print(b)
