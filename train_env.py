import pandas as pd
import numpy as np
import Elden_Bot as Bot
import graph_optimizer as GO
probs_depth = 30
stack_depth = 30
n_filters = 16
n_hidden_1 = 500
n_hidden_2 = 200
n_hidden_3 = 80
kernel_size = 3
n_channels = 3
scale_factor = 5
width = ((791//(scale_factor)))
height = ((440//(scale_factor)))
controls_df = pd.read_csv('Custom_Controls.csv')
control_options = list(controls_df['Control'].values)
control_options = np.array(control_options)
n_controls = len(control_options)
A = stack_depth*n_channels
B = n_filters*kernel_size*kernel_size
C =n_hidden_1*(width+2)*(height+2)
D = n_hidden_1
E = n_hidden_1*n_hidden_2
F = n_hidden_2
G = n_hidden_2*n_hidden_3
H = n_hidden_3
I = n_hidden_3*n_controls
J = n_controls
K = n_controls*probs_depth
L = n_controls
M = n_filters
n_dims = (A+B+C+D+E+F+G+H+I+J+K+L+M)
lim = [-1*np.ones(n_dims),np.ones(n_dims)]
F = Bot.run_bot
a,b,c = GO.optimize(F,lim,n_parts=10,n_iterations=300)
print(b)
