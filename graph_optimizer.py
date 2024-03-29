"""pop based optimizer, takes any function, works with discrete,
   continuous, combinatorial and mixed problems"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist


def distance_matrix(points, cat_vars):
    # Separate categorical and spatial points
    cat_points = points * cat_vars
    spacial_points = points * (1 - cat_vars)

    # Calculate Euclidean distances between spatial points
    euclidean_distances = cdist(spacial_points, spacial_points, metric="euclidean")

    # Calculate Hamming distances between categorical points
    cat_distances = cdist(cat_points, cat_points, metric="hamming")

    # Combine distances
    prox_matrix = euclidean_distances + cat_distances

    return prox_matrix


def plot_scatter(points, values, constraint_count):
    "plots all points in the swarm, works for 2D only"
    data_frame = pd.DataFrame(
        {
            "X": points[:, 0],
            "Y": points[:, 1],
            "values": values,
            "constraint": constraint_count.T.flatten().astype(str),
        }
    )
    sns.scatterplot(
        data=data_frame,
        x="X",
        y="Y",
        hue="values",
        markers="constraint",
        palette="magma",
    )
    plt.show()


def get_constraint_breaks(eval_point, constraints=None):
    """
    Applies constraint functions to the problem and returns the number of constraint breaks.

    Args:
        eval_point (float): The evaluation point.
        constraints (dict): A dictionary containing the constraint functions and their parameters.

    Returns:
        int: The number of constraint breaks.
    """
    constraint_breaks = 0
    if constraints:
        for constraint, params in constraints.items():
            args = params.get("args")
            values = constraint(eval_point, args) if args else constraint(eval_point)
            threshold = params["thresh"]
            sign = params["sign"]
            constraint_breaks += sign * threshold > sign * values
    return constraint_breaks


def limit_pos(eval_point, limite, passo, permut):
    """guarantees that tried points comply with any restrictions"""
    num_dim = len(limite[0])
    for i in range(num_dim):
        if eval_point[i] > limite[1][i]:
            dif = (eval_point[i] - limite[1][i]) % (limite[1][i] - limite[0][i])
            eval_point[i] = limite[0][i] + dif
        elif eval_point[i] < limite[0][i]:
            dif = limite[0][i] - eval_point[i] % (limite[1][i] - limite[0][i])
            eval_point[i] = limite[1][i] - dif
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
        s_var = passo[i]
        if s_var > 0.0:
            grid = round(eval_point[i] / s_var)
            eval_point[i] = grid * s_var
    return eval_point


def optimize(
    objective_function,
    lim,
    *args,
    base_solution=None,
    n_parts=20,
    n_iterations=200,
    min_neighbors=3,
    global_search_threshold=1e-1,
    Const_funcs=None,
    passo=None,
    permut=False,
    cat_vars_index=None,
    plot=False,
    return_history=False,
    init_mode="uniform"
):
    """Runs the actual optimization process"""
    lim = np.asarray(lim)
    n_dims = lim.shape[1]
    max_neighbors = n_parts // 3 + 1
    constraint_count = np.inf * np.ones(n_parts)
    best_self_constraint_count = np.inf * np.ones(n_parts)
    values = np.zeros(n_parts)
    if passo is None:
        passo = np.zeros(n_dims)
    step_size = np.linalg.norm(passo)
    if cat_vars_index is None:
        cat_vars_index = np.zeros(n_dims)
    if base_solution is not None:
        std = 0.05 * ((lim[1] - lim[0]).min())
        swarm = base_solution + np.random.normal(0, std, (n_parts, n_dims))
    elif init_mode == "uniform":
        swarm = np.random.uniform(lim[0], lim[1], (n_parts, n_dims))
    elif init_mode == "normal":
        R = lim[1] - lim[0]
        swarm = np.random.normal((lim[0] + lim[1]) / 2.0, 0.05 * R, (n_parts, n_dims))
    for i in range(n_parts):
        swarm[i] = limit_pos(swarm[i], lim, passo, permut)
        best_self_constraint_count[i] = get_constraint_breaks(
            swarm[i], constraints=Const_funcs
        )
    best_pos_history = []
    best_vals_history = []
    for n, part in enumerate(swarm):
        print("\n")
        print("iteration: 0 part: ", n)
        values[n] = objective_function(part, *args)
    global_best_index = np.argmin(values)
    global_best_value = np.min(values)
    global_best_pos = swarm[global_best_index]
    if base_solution is not None:
        ref_value = objective_function(base_solution, *args)
        if ref_value < global_best_value:
            global_best_value = ref_value
            global_best_pos = base_solution
    np.save("MetaBrain.npy", global_best_pos)
    best_constraint_break = np.min(best_self_constraint_count)
    print("best so far: ", global_best_value)
    movement = 0
    r_0 = 0
    r_1 = 0
    r_2 = 0
    speed = np.inf
    tested_points = {}
    self_best = swarm.copy()
    self_best_vals = values.copy()
    if n_dims == 2 and plot:
        c = np.zeros(n_parts)
        plot_scatter(swarm, values, c)
    j = 0
    global_search_threshold = global_search_threshold ** (1 / n_dims)
    global_search_threshold += step_size
    iter_since_improv = 0
    while speed > 0:
        n_neighbors = np.random.randint(min_neighbors, max_neighbors)
        j += 1
        progress = j / n_iterations
        speed = 0
        max_speed = (np.min(lim[1] - lim[0])) / (
            2 * (1 + progress)
        ) + global_search_threshold
        Dists = distance_matrix(swarm, cat_vars_index)
        neighbor_indexes = np.argpartition(Dists, n_neighbors, axis=0)[:n_neighbors].T
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
        mode_ind = np.random.uniform(0, 0.6, (n_parts, 3))
        phase_ind = np.random.uniform(0, np.pi, (n_parts, 3))
        for i in range(n_parts):
            print("\n")
            print("iteration: ", j, " part: ", i)
            if np.linalg.norm(swarm[i] - global_best_pos) > 0.0:
                if mode_ind[i][0] < 0.5:
                    r_0 = np.sin(phase_ind[i][0])
                elif mode_ind[i][0] >= 0.5:
                    r_0 = np.cos(phase_ind[i][0])
                if mode_ind[i][1] < 0.5:
                    r_1 = np.sin(phase_ind[i][1])
                elif mode_ind[i][1] >= 0.5:
                    r_1 = np.cos(phase_ind[i][1])
                if mode_ind[i][2] < 0.5:
                    r_2 = np.sin(phase_ind[i][2])
                elif mode_ind[i][2] >= 0.5:
                    r_2 = np.cos(phase_ind[i][2])
                movement = r_0 * (best_neighbors[i] - swarm[i]) + r_1 * (
                    self_best[i] - swarm[i]
                )
                if (best_neighbors[i] - swarm[i]).sum() == 0:
                    movement += r_2 * (global_best_pos - swarm[i])
            else:
                best_kick = np.random.uniform(
                    -0.01 * (1.0 - progress),
                    0.01 * (1.0 - progress),
                    n_dims,
                )
                swarm[i] += best_kick
                movement = best_kick
            ind_speed = np.linalg.norm(movement)
            if ind_speed > max_speed:
                movement *= (max_speed / ind_speed) ** n_dims
                ind_speed = np.linalg.norm(movement)
            speed += ind_speed
            swarm[i] = swarm[i] + movement
            swarm[i] = limit_pos(swarm[i], lim, passo, permut)
            point = swarm[i]
            p_key = tuple(point)
            if p_key in tested_points:
                values[i] = tested_points[p_key]
            else:
                values[i] = objective_function(point, *args)
                tested_points[p_key] = values[i]
            constraint_count[i] = get_constraint_breaks(
                swarm[i], constraints=Const_funcs
            )
            if constraint_count[i] < best_self_constraint_count[i]:
                self_best[i] = swarm[i]
            elif values[i] <= (
                self_best_vals[i] > values[i]
                and constraint_count[i] == best_self_constraint_count[i]
            ):
                self_best[i] = swarm[i]
            if constraint_count[i] < best_constraint_break:
                best_constraint_break = constraint_count[i]
                global_best_index = i
                global_best_value = values[i]
                global_best_pos = swarm[i]
                best_pos_history.append(global_best_pos)
                best_vals_history.append(global_best_value)
                iter_since_improv = 0
                print("\n")
                print("best so far: ", global_best_value)
                print("\n")
                np.save("MetaBrain.npy", global_best_pos)
            elif constraint_count[i] == best_constraint_break and (
                values[i] < global_best_value
            ):
                global_best_index = i
                global_best_value = values[i]
                print("best so far: ", global_best_value)
                global_best_pos = swarm[i]
                best_pos_history.append(global_best_pos)
                best_vals_history.append(global_best_value)
                iter_since_improv = 0
                np.save("MetaBrain.npy", global_best_pos)
            else:
                iter_since_improv += 1
        speed /= n_parts
        if n_dims == 2 and plot and j % 5 == 1:
            plot_scatter(swarm, values, constraint_count)
        if speed <= ((global_search_threshold * 0.5)):
            revive = np.random.uniform()
            revive_thresh = 1 - progress
            if revive < revive_thresh:
                kick_dev = 0.1 * swarm.std()
                k_1 = np.random.normal(0.0, kick_dev, swarm.shape)
                k_2 = np.random.normal(0.0, kick_dev, swarm.shape)
                kick = k_1 / k_2
                print("\n")
                print("kicked with power: ", kick_dev, " at iter: ", j)
                print("\n")
                speed = np.linalg.norm(kick)
                swarm = swarm + kick
                for part in swarm:
                    part = limit_pos(part, lim, passo, permut)
        remaining_evals = (1 - progress) * n_iterations * n_parts
        rand_1 = np.random.uniform()
        if remaining_evals < rand_1 * iter_since_improv:
            break
        if progress >= 1:
            break
    if return_history is True:  # serve p avaliar convergencia, n tem utilidade em prod
        return (
            global_best_pos,
            global_best_value,
            best_constraint_break,
            best_pos_history,
            best_vals_history,
        )
    if return_history is False:
        return global_best_pos, global_best_value, best_constraint_break
