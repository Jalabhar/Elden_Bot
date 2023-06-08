'''pop based optimizer, takes any function, works with discrete,
   continuous, combinatorial and mixed problems'''
import numpy as np
from scipy.spatial.distance import cdist
import time

def distance_matrix(points, cat_vars):
    '''gets the distance between all point pairs'''
    cat_points = points*cat_vars
    spacial_points = points*(1-cat_vars)
    euclidean_distances = cdist(
        spacial_points, spacial_points, metric='euclidean')
    cat_distances = cdist(cat_points, cat_points, metric='hamming')
    prox_matrix = euclidean_distances+cat_distances
    return prox_matrix



def get_constraints(eval_point, constraints=None):
    '''imposes all constraint functions on the problem'''
    constraint_breaks = 0
    if constraints is not None:
        for func in constraints:
            parameters = constraints[func]
            if len(parameters) == 2:
                args = parameters[2]
                constraint_values = func(eval_point, args)
            else:
                constraint_values = func(eval_point)
            constraint_thresh = parameters[0]
            sign = parameters[1]
            constraint_breaks += (sign*constraint_thresh >
                                  sign*constraint_values)
    return constraint_breaks


def limit_pos(eval_point, limite, passo, permut):
    '''guarantees that tried points comply with any restrictions'''
    num_dim = len(limite[0])
    for i in range(num_dim):
        s_var = passo[i]
        if s_var > 0.0:
            grid = round(eval_point[i] / s_var)
            eval_point[i] = grid * s_var
    if permut:
        uni = []
        b = list(range(num_dim))
        p = list(eval_point)
        p += b
        for v in p:
            if v not in uni:
                uni.append(v)
        eval_point = np.asarray(uni)
    for i in range(num_dim):
        if eval_point[i] > limite[1][i]:
            dif = eval_point[i]-limite[1][i]
            eval_point[i] = limite[0][i]+dif
        elif eval_point[i] < limite[0][i]:
            dif = limite[0][i]-eval_point[i]
            eval_point[i] = limite[1][i]-dif
    return eval_point


def optimize(objective_function, lim, *args,  n_parts=20,n_iterations=200,
             n_neighbors=3, global_search_threshold=1e-1,
             Const_funcs=None, passo=None,permut=False, cat_vars_index=None,
             return_history=False):
    '''Runs the actual optimization process'''
    lim = np.asarray(lim)
    n_dims = lim.shape[1]
    values = []
    constraint_count = np.inf*np.ones(n_parts)
    best_self_constraint_count = np.inf*np.ones(n_parts)
    if passo is None:
        passo = np.zeros(n_dims)
    step_size = np.linalg.norm(passo)
    if cat_vars_index is None:
        cat_vars_index = np.zeros(n_dims)
    try:
        prior_best = np.load('LargeEldenBrain.npy')
        baseline = objective_function(prior_best)
        swarm = np.random.normal(prior_best,0.1,(n_parts,n_dims))
        print('valid prior loaded')
    except (FileNotFoundError,ValueError):
        swarm = np.random.normal(0,0.1,(n_parts,n_dims))
        swarm = swarm/np.maximum(np.abs(swarm.max()),np.abs(swarm.min()))
    for i in range(n_parts):
        print('\n')
        print('running element ' +str(i+1)+' at iteration 0')
        swarm[i] = limit_pos(swarm[i], lim, passo, permut)
        best_self_constraint_count[i] = get_constraints(
                swarm[i], constraints=Const_funcs)
        best_pos_history = []
        best_vals_history = []
        values.append(objective_function(swarm[i]))
    values = np.asarray(values)
    global_best_index = np.argmin(values)
    global_best_value = np.min(values)
    global_best_pos = swarm[global_best_index]
    np.save('LargeEldenBrain.npy',global_best_pos)
    best_constraint_break = np.inf
    movement = 0
    r_0 = 0
    r_1 = 0
    r_2 = 0
    speed = np.inf
    tested_points = {}
    self_best = swarm.copy()
    self_best_vals = values.copy()
    j = 0
    global_search_threshold = global_search_threshold**(1/n_dims)
    global_search_threshold += step_size
    global_search_steps = 0
    iter_since_improv = 0
    while speed > 0:
        print('mean value at iter ' + str(j) + ' is: ' + str(values.mean()))
        print('\n')
        time.sleep(5.0)
        j += 1
        print('starting iter: '+str(j))
        progress = j/n_iterations
        speed = 0
        max_speed = (np.min(lim[1]-lim[0]))/(2*(1+progress)) + global_search_threshold
        global_worst_index = np.argmax(values)
        best_subset_index = np.argpartition(values, n_neighbors)[:n_neighbors]
        best_subset = swarm[best_subset_index]
        p_subset = np.random.uniform(0,1,(n_neighbors,1))
        p_subset = p_subset/p_subset.sum()
        new_point = np.asarray([(p_subset*best_subset).sum(axis=0)])
        swarm[global_worst_index] = new_point
        swarm[global_worst_index] = limit_pos(
            swarm[global_worst_index], lim, passo, permut)
        Dists = distance_matrix(swarm, cat_vars_index)
        Dists[Dists == 0] = Dists.max()
        neighbor_indexes = np.argpartition(
            Dists, n_neighbors, axis=0)[:n_neighbors].T
        best_neighbors = []
        best_neighbor_vals = []
        for index in neighbor_indexes:
            neighbors = swarm[index]
            local_vals = values[index]
            chosen_index = local_vals.argmin()
            min_neighbor_val = local_vals[chosen_index]
            best_neighbor = neighbors[chosen_index]
            best_neighbors.append(best_neighbor)
            best_neighbor_vals.append(min_neighbor_val)
        mode_ind = np.random.uniform(0, 1, (n_parts, 3))
        phase_ind = np.random.uniform(-np.pi, np.pi, (n_parts, 3))
        for i in range(n_parts):
            print('\n')
            print('running element ' +str(i+1)+' at iteration ' + str(j))
            if mode_ind[i][0] < 0.5:
                a = (np.sin(phase_ind[i][0])+1)/2.0
                b = (np.cos(phase_ind[i][0])+1)/2.0                
                r_0 = a**b-0.5
            elif mode_ind[i][0] >= 0.5:
                a = (np.sin(phase_ind[i][0])+1)/2.0
                b = (np.cos(phase_ind[i][0])+1)/2.0                
                r_0 = b**a-0.5
            if mode_ind[i][1] < 0.5:
                a = (np.sin(phase_ind[i][1])+1)/2.0
                b = (np.cos(phase_ind[i][1])+1)/2.0                
                r_1 = a**b-0.5
            elif mode_ind[i][1] >= 0.5:
                a = (np.sin(phase_ind[i][1])+1)/2.0
                b = (np.cos(phase_ind[i][1])+1)/2.0                
                r_1 = b**a-0.5
            if mode_ind[i][2] < 0.5:
                a = (np.sin(phase_ind[i][2])+1)/2.0
                b = (np.cos(phase_ind[i][2])+1)/2.0                
                r_2 = a**b-0.5
            elif mode_ind[i][2] >= 0.5:
                a = (np.sin(phase_ind[i][2])+1)/2.0
                b = (np.cos(phase_ind[i][2])+1)/2.0                
                r_2 = b**a-0.5
            movement = (r_0*(best_neighbors[i]-swarm[i]) +
                        r_1*(self_best[i]-swarm[i]))
            ind_speed = np.linalg.norm(movement)
            if ind_speed<global_search_threshold:
                global_search_steps += 1
                movement+= r_2*(global_best_pos - swarm[i])
                ind_speed = np.linalg.norm(movement)
            if ind_speed > max_speed:
                movement *= (max_speed/ind_speed)**n_dims
                ind_speed = np.linalg.norm(movement)
            speed += ind_speed
            swarm[i] = (swarm[i] + movement)
            swarm[i] = limit_pos(swarm[i], lim, passo, permut)
            point = swarm[i]
            p_key = tuple(point)
            if p_key in tested_points:
                values[i] = tested_points[p_key]
            else:
                values[i] = objective_function(point, *args)
                tested_points[p_key] = values[i]
            if values[i]<global_best_value:
                np.save('LargeEldenBrain.npy',swarm[i])
            constraint_count[i] = get_constraints(
                swarm[i], constraints=Const_funcs)
            if constraint_count[i]<best_self_constraint_count[i]:
                self_best[i] = swarm[i]
            elif values[i] <= (self_best_vals[i]>values[i] and
                               constraint_count[i]==best_self_constraint_count[i]):
                self_best[i] = swarm[i]
            if constraint_count[i] < best_constraint_break:
                best_constraint_break = constraint_count[i]
                global_best_index = i
                global_best_value = values[i]
                global_best_pos = swarm[i]
                best_pos_history.append(global_best_pos)
                best_vals_history.append(global_best_value)
                iter_since_improv = 0
            elif constraint_count[i] == best_constraint_break and (values[i] <
                global_best_value):
                global_best_index = i
                global_best_value = values[i]
                global_best_pos = swarm[i]
                best_pos_history.append(global_best_pos)
                best_vals_history.append(global_best_value)
                iter_since_improv = 0
            else:
                iter_since_improv += 1
        pop_std = np.std(swarm, axis=0)
        speed /= n_parts
        if speed <= ((global_search_threshold*0.5)):
            revive = np.random.uniform()
            if revive < (1-progress**2)**2:
                speed = 0.1
                swarm = np.random.normal(
                    global_best_pos, pop_std, (n_parts, n_dims))
        remaining_evals = (n_iterations-j)*n_parts
        rand_1 = np.random.uniform()
        if remaining_evals < rand_1*iter_since_improv:
            break
        if progress >= 1:
            break
    return_value = objective_function(global_best_pos,*args)
    if return_history is True: # serve p avaliar convergencia, n tem utilidade em prod
        return (global_best_pos, return_value, best_constraint_break,
                best_pos_history, best_vals_history)
    if return_history is False:
        return global_best_pos, return_value, best_constraint_break
