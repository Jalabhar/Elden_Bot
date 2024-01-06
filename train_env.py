import pandas as pd
import numpy as np
import Elden_Bot as Bot
import optimizer_base as opt
import graph_optimizer as GO

env_steps = 1000
probs_depth = 10
stack_depth = 10
n_filters = 4
n_hidden_1 = 100
n_hidden_2 = 50
n_hidden_3 = 25
kernel_size = 3
n_channels = 3
scale_factor = 10
base_width = 791
base_height = 440
width = 80  # int(np.ceil(base_width / scale_factor))
height = 44  # int(np.ceil(base_height / scale_factor))
controls_df = pd.read_csv("Custom_Controls.csv")
control_options = controls_df["Control"].values
# control_options = np.array(control_options)
n_controls = len(control_options)
A = stack_depth * n_channels
B = n_filters * kernel_size * kernel_size
C = n_hidden_1 * width // 4 * height // 4
D = n_hidden_1
E = n_hidden_1 * n_hidden_2
F = n_hidden_2
G = n_hidden_2 * n_hidden_3
H = n_hidden_3
I = n_hidden_3 * n_controls
J = n_controls
K = n_controls * probs_depth
L = n_controls
M = n_filters * kernel_size * kernel_size
n_dims = A + B + C + D + E + F + G + H + I + J + K + L + M
print(n_dims)
lim = [-1.0 * np.ones(n_dims), 1.0 * np.ones(n_dims)]
obj = Bot.run_bot
# a,b = opt.optimize(F,lim,n_iter=10,n_parts=20,n_keep=5)
a, b, c = GO.optimize(obj, lim, env_steps,base_solution=None, n_iterations=50, n_parts=12, n_neighbors=3)
print(a)
