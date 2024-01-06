import numpy as np
import pandas as pd
import pyautogui
import pydirectinput
import pygetwindow
from scipy.signal import convolve
from collections import deque
import time
from PIL import Image as Image
import cv2

probs_depth = 10
stack_depth = 10
n_filters = 4
n_hidden_1 = 10
n_hidden_2 = 100
n_hidden_3 = 100
kernel_size = 3
n_channels = 3
scale_factor = 10
base_width = 791
base_height = 440
exploration_depth = 200
width = 80  # int(np.ceil(base_width / scale_factor))
height = 44  # int(np.ceil(base_height / scale_factor))
controls_df = pd.read_csv("Custom_Controls.csv")
control_options = list(controls_df["Control"].values)
control_options = np.array(control_options)
for control in control_options:
    control = control.lower()
n_controls = len(control_options)


def reset_trial():
    pydirectinput.keyUp("w")
    pydirectinput.keyUp("a")
    pydirectinput.keyUp("s")
    pydirectinput.keyUp("d")
    pydirectinput.keyUp("space")
    pydirectinput.press("g")
    pydirectinput.press("f")
    pydirectinput.press("f")
    pydirectinput.press("e")
    pydirectinput.press("e")


def walk_to_boss(boss=None):  # margit
    if boss == "sentinel duo":
        time.sleep(0.5)
        pydirectinput.keyDown("s")
        time.sleep(0.2)
        pydirectinput.keyUp("s")
        pydirectinput.keyDown("q")
        time.sleep(0.2)
        pydirectinput.keyUp("q")
        pydirectinput.keyDown("w")
        time.sleep(30.0)
        pydirectinput.keyUp("w")
        time.sleep(0.2)
    if boss == "margit":
        time.sleep(0.5)
        pydirectinput.keyDown("s")
        time.sleep(0.2)
        pydirectinput.keyUp("s")
        pydirectinput.keyDown("w")
        pydirectinput.keyDown("d")
        time.sleep(1.0)
        pydirectinput.keyUp("d")
        pydirectinput.keyDown("a")
        time.sleep(0.1)
        pydirectinput.keyUp("a")
        time.sleep(5.0)
        pydirectinput.keyUp("w")
        pydirectinput.keyDown("w")
        pydirectinput.keyDown("d")
        time.sleep(0.9)
        pydirectinput.keyUp("d")
        pydirectinput.keyUp("w")
        pydirectinput.keyDown("d")
        time.sleep(0.4)
        pydirectinput.keyUp("d")
        pydirectinput.keyDown("w")
        time.sleep(0.2)
        pydirectinput.keyUp("w")
        pydirectinput.press("e")
        time.sleep(0.2)
        pydirectinput.press("e")
    if boss == "bridge troll":
        time.sleep(0.5)
        pydirectinput.keyDown("w")
        time.sleep(7.0)
        pydirectinput.keyUp("w")
    if boss == None:
        time.sleep(0.5)
    print("Engaging Bot")


def get_boss_life(x1, y1, x2, y2):
    stats_bar = (x1 + 180, y1 + 385, x2 - 260, y2 - 470)
    # area = np.abs(stats_bar[2] - stats_bar[0]) * np.abs(stats_bar[3] - stats_bar[1])
    boss_life_bar = np.array(pyautogui.screenshot(region=(stats_bar)))
    hsv = cv2.cvtColor(boss_life_bar, cv2.COLOR_RGB2HSV)
    boss_life_lower_1 = np.array([0, 190, 60])
    boss_life_upper_1 = np.array([30, 255, 80])
    mask_1 = cv2.inRange(hsv, boss_life_lower_1, boss_life_upper_1)
    boss_life_lower_2 = np.array([330, 190, 60])
    boss_life_upper_2 = np.array([359, 255, 80])
    mask_2 = cv2.inRange(hsv, boss_life_lower_2, boss_life_upper_2)
    merge_boss_life_bar = cv2.bitwise_or(mask_1, mask_2)
    life_value = merge_boss_life_bar.sum() / 255.0
    return life_value


def get_values(x1, y1, x2, y2):
    red_bar = (x1 + 65, y1 + 50, x2 - 670, y2 - 480)
    blue_bar = (x1 + 65, y1 + 57, x2 - 670, y2 - 480)
    green_bar = (x1 + 65, y1 + 65, x2 - 670, y2 - 480)
    life_bar = np.array(pyautogui.screenshot(region=red_bar))
    life_lower = np.array([0, 90, 75])
    life_upper = np.array([150, 255, 125])
    life_hsv = cv2.cvtColor(life_bar, cv2.COLOR_RGB2HSV)
    life_bar = cv2.inRange(life_hsv, life_lower, life_upper)
    life_value = life_bar.sum() / 255.0
    # stamina
    stamina_bar = np.array(pyautogui.screenshot(region=green_bar))
    stamina_lower = np.array([6, 52, 24])
    stamina_upper = np.array([74, 255, 77])
    stamina_hsv = cv2.cvtColor(stamina_bar, cv2.COLOR_RGB2HSV)
    stamina_bar = cv2.inRange(stamina_hsv, stamina_lower, stamina_upper)
    stamina_value = stamina_bar.sum() / 255.0
    # mana
    mana_bar = np.array(pyautogui.screenshot(region=blue_bar))
    mana_lower = np.array([120, 52, 24])
    mana_upper = np.array([194, 255, 77])
    mana_hsv = cv2.cvtColor(mana_bar, cv2.COLOR_RGB2HSV)
    mana_bar = cv2.inRange(mana_hsv, mana_lower, mana_upper)
    mana_value = mana_bar.sum() / 255.0
    return life_value, mana_value, stamina_value


def activate_control(command, walking):
    command = command.lower()
    control_action = command.split("+")
    n_actions = len(control_action)
    if n_actions == 2:
        pydirectinput.keyDown(control_action[0])
        time.sleep(0.1)
        pydirectinput.keyDown(control_action[1])
        time.sleep(0.2)
        pydirectinput.keyUp(control_action[1])
        time.sleep(0.1)
        pydirectinput.keyUp(control_action[0])
        if control_action[1] == "x":
            for _ in range(2):
                pydirectinput.keyDown("q")
                time.sleep(0.1)
                pydirectinput.keyUp("q")
                time.sleep(0.1)
    elif n_actions == 1:
        base_action = control_action[0]
        if base_action == "w":
            if not walking:
                pydirectinput.keyDown(base_action)
                walking = True
            elif walking:
                pydirectinput.keyUp(base_action)
                walking = False
        elif base_action == "e":
            pydirectinput.press(base_action)
            time.sleep(0.5)
            pydirectinput.press("q")
        elif base_action == "lmb":
            pydirectinput.mouseDown(button="left")
            time.sleep(0.1)
            pydirectinput.mouseUp(button="left")
        elif base_action == "rmb":
            pydirectinput.mouseDown(button="right")
            time.sleep(0.1)
            pydirectinput.mouseUp(button="right")
        elif base_action in ["do nothing"]:
            time.sleep(0.2)
            # pass
        else:
            pydirectinput.press(base_action)
    return walking


def probs_divergence(target_p, actual_p):
    m = np.multiply(target_p, actual_p)
    bc = np.sum(np.sqrt(m))
    dist = np.sqrt(1 - bc)
    return dist


def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def mish_activation(x):
    rng = np.maximum(np.abs(x.max()), np.abs(x.min()))
    scaled_x = x / rng
    Result = scaled_x * np.tanh(np.log(1 + np.exp(scaled_x)))
    return Result


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_probs(X, stack, prior_probs, score):
    prior_probs = np.array(prior_probs)
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
    X_stack = X[:A]
    # stack_bias =X[A:A]
    X_kernel = X[A : A + B]
    # kernel_bias =X[A+B:A+B]
    X_W_1 = X[A + B : A + B + C]
    X_B_1 = X[A + B + C : A + B + C + D]
    X_W_2 = X[A + B + C + D : A + B + C + D + E]
    X_B_2 = X[A + B + C + D + E : A + B + C + D + E + F]
    X_W_3 = X[A + B + C + D + E + F : A + B + C + D + E + F + G]
    X_B_3 = X[A + B + C + D + E + F + G : A + B + C + D + E + F + G + H]
    X_W_4 = X[A + B + C + D + E + F + G + H : A + B + C + D + E + F + G + H + I]
    X_B_4 = X[A + B + C + D + E + F + G + H + I : A + B + C + D + E + F + G + H + I + J]
    X_W_P = X[
        A
        + B
        + C
        + D
        + E
        + F
        + G
        + H
        + I
        + J : A
        + B
        + C
        + D
        + E
        + F
        + G
        + H
        + I
        + J
        + K
    ]
    X_B_P = X[
        A
        + B
        + C
        + D
        + E
        + F
        + G
        + H
        + I
        + J
        + K : A
        + B
        + C
        + D
        + E
        + F
        + G
        + H
        + I
        + J
        + K
        + L
    ]
    X_kernel_2 = X[-M:]
    stack_kernel = X_stack.reshape((stack_depth, 1, 1, n_channels))
    stack_kernel = stack_kernel * 2.0 - 1.0
    kernel = X_kernel.reshape((n_filters, kernel_size, kernel_size, 1))
    # kernel = kernel * 2.0 - 1.0
    kernel_2 = X_kernel_2.reshape((n_filters, kernel_size, kernel_size, 1))
    # kernel_2 = kernel_2 * 2.0 - 1.0
    W_1 = X_W_1.reshape((n_hidden_1, (height // 4) * (width // 4)))
    B_1 = X_B_1.reshape((n_hidden_1, 1))
    W_2 = X_W_2.reshape((n_hidden_2, n_hidden_1))
    B_2 = X_B_2.reshape((n_hidden_2, 1))
    W_3 = X_W_3.reshape((n_hidden_3, n_hidden_2))
    B_3 = X_B_3.reshape((n_hidden_3, 1))
    W_4 = X_W_4.reshape((n_controls, n_hidden_3))
    B_4 = X_B_4.reshape((n_controls, 1))
    W_P = X_W_P.reshape(n_controls, probs_depth)
    B_P = X_B_P.reshape(n_controls, 1)
    out_stack = convolve(stack_kernel, stack, "valid")
    out_frame = convolve(out_stack, kernel, "same")
    out_frame = out_frame[:, ::2, ::2, :]
    out_frame = convolve(out_frame, kernel_2, "same")
    out_frame = out_frame[:, ::2, ::2, :]
    flat = np.expand_dims(out_frame.flatten(), axis=-1)
    P_1 = np.dot(W_1, flat)
    R_1 = P_1 + B_1
    # R_1 = (R_1 - R_1.mean()) / R_1.std()
    R_1 = mish_activation(R_1)
    P_2 = np.dot(W_2, R_1)
    R_2 = P_2 + B_2
    # R_2 = (R_2 - R_2.mean()) / R_2.std()
    R_2 = mish_activation(R_2)
    P_3 = np.dot(W_3, R_2)
    R_3 = P_3 + B_3
    # R_3 = (R_3 - R_3.mean()) / R_3.std()
    R_3 = mish_activation(R_3)
    P_4 = np.dot(W_4, R_3)
    R_4 = P_4 + B_4
    if score != 0:
        R_4 = R_4 * score
    # R_4 = (R_4 - R_4.mean()) / R_4.std()
    R_4 = mish_activation(R_4)
    weighted_prev = np.dot(W_P, prior_probs)
    # weighted_prev = (weighted_prev - weighted_prev.mean()) / weighted_prev.std()
    weighted_prev = weighted_prev + B_P
    weighted_prev = mish_activation(weighted_prev)
    P = np.dot(weighted_prev, R_4)
    P = np.squeeze(P)
    scaled_P = softmax(P)
    return scaled_P


def run_bot(x, total_steps=100):
    controls_df = pd.read_csv("Custom_Controls.csv")
    weighted_boss_life = 0.0
    stuck = False
    is_alive = False
    is_dead = False
    boss_found = False
    boss_lost = False
    mode = "stochastic"
    exploration_stack = []
    control_options = list(controls_df["Control"].values)
    control_options = np.array(control_options)
    n_controls = len(control_options)
    probs_array = np.zeros((probs_depth, n_controls))
    probs_array[:, -1:] = 1
    probs_stack = deque(probs_array)
    # probs_score = 0.0
    max_steps = total_steps
    total_score = 0.0
    Life, Mana, Stamina = 0, 0, 0
    Last_Life, Last_Mana, Last_Stamina = Life, Mana, Stamina
    boss_life = 0
    my_window_title = "ELDEN RING™"
    my_window = pygetwindow.getWindowsWithTitle(my_window_title)[0]
    my_window.activate()
    my_window.moveTo(0, 0)
    x1, y1, x2, y2 = my_window._getWindowRect()
    r = (x1 + 16, y1 + 40, x2 - 25, y2 - 49)
    while not is_alive:
        Life, Mana, Stamina = get_values(x1, y1, x2, y2)
        if (255.0 * Life) >= 4.0:
            is_alive = True
            is_dead = False
        else:
            pass
    else:
        first_action = True
        if first_action:
            walk_to_boss()
            time.sleep(1.0)
            first_action = False
        frame_stack = deque([])
        counter = 0
        command_stack = deque([])
        command_reps = probs_depth
        max_repetition = 0.3
        Walking = False
        novelty_threshold = 1e-12
        final_exploration_score = 0
        final_step_score = 0
        final_boss_search_score = 0.0
        final_part_score = 0.0
        partial_score = 0.0
        exploration_score = 0.0
        stats_score = 0
        while True:
            Last_Life, Last_Mana, Last_Stamina = Life, Mana, Stamina
            step_score = 0
            boss_search_score = 0.0
            part_score = 0.0
            partial_score = 0.0
            step_score -= np.log(1 + counter)
            boss_life = get_boss_life(x1, y1, x2, y2)
            weighted_boss_life = 0.5 * (weighted_boss_life + boss_life)
            boss_delta = boss_life - weighted_boss_life
            if (
                boss_found == False
                and boss_lost == False
                and boss_delta > 600.0
                and boss_life > 1000
            ):
                boss_found = True
                boss_search_score += 100
            elif boss_found == True and boss_lost == False and boss_delta < -600.0:
                boss_found = False
                boss_lost = True
                boss_search_score -= 10
            elif boss_lost == True and boss_delta > 600.0 and boss_life > 1000:
                boss_found = True
                boss_lost = False
                boss_search_score += 5
            elif (
                boss_found == True
                and boss_lost == False
                and weighted_boss_life >= 1000.0
            ):
                boss_found = True
                boss_lost = False
                boss_search_score += 0.1
            Life, Mana, Stamina = get_values(x1, y1, x2, y2)
            Life_score = Life - Last_Life
            Mana_score = Mana - Last_Mana
            Stamina_score = Stamina - Last_Stamina
            stats_score = Life_score + Mana_score + Stamina_score
            if np.abs(stats_score) < 0.01:
                stats_score = 0.0
            if boss_found:
                damage_score = -1 * boss_delta
                if np.abs(damage_score) < 100.0:
                    damage_score = 0.0
            else:
                damage_score = 0.0
            if damage_score == 0.0:
                damage_score_sign = 1.0
            else:
                damage_score_sign = damage_score / np.abs(damage_score)
            damage_score = damage_score_sign * np.log(1.0 + np.abs(damage_score))
            if (255.0 * Life) < 0.2:
                is_alive = False
                is_dead = True
                time.sleep(2.0)
                if not is_alive:
                    capt_frame = np.array(pyautogui.screenshot(region=r))
                    capt_frame = cv2.cvtColor(capt_frame, cv2.COLOR_RGB2HSV)
                    capt_frame[:, :, :1] = capt_frame[:, :, :1] / 180.0
                    capt_frame[:, :, 1:] = capt_frame[:, :, 1:] / 255.0
                    if capt_frame.mean() < 0.1:
                        pydirectinput.keyUp("w")
                        pydirectinput.press("RightArrow")
                        time.sleep(0.2)
                        pydirectinput.press("e")
                        time.sleep(1.5)
                for Key in ["a", "w", "s", "d", "space"]:
                    pydirectinput.keyUp(Key)
                time.sleep(4.0)
                if total_score == np.inf:
                    total_score = 1.0
                stuck = True
                break
            elif 4.0 > (255.0 * Life) > 0.2:
                is_alive = False
                is_dead = False
            elif (255.0 * Life) >= 4.0:
                is_alive = True
                is_dead = False
            if counter % 16 == 15:
                time.sleep(0.2)
            if is_alive and not stuck:
                counter += 1
                capt_frame = np.array(pyautogui.screenshot(region=r))
                capt_frame = cv2.resize(
                    capt_frame, (width, height), interpolation=cv2.INTER_AREA
                )
                capt_frame = cv2.cvtColor(capt_frame, cv2.COLOR_RGB2HSV)
                if len(exploration_stack) == 0:
                    exploration_stack.append(capt_frame)
                else:
                    ac_diff = np.inf
                    for explored_frame in exploration_stack:
                        frame_diff = np.linalg.norm(capt_frame - explored_frame)
                        frame_diff = frame_diff / (width * height * n_channels)
                        ac_diff = np.minimum(frame_diff, ac_diff)
                    novelty_value = ac_diff
                    if novelty_value > novelty_threshold:
                        exploration_stack.append(capt_frame)
                        exploration_score += novelty_value  # ** 0.5
                        novelty_threshold = 0.05 * np.asarray(exploration_stack).std()
                        exploration_stack = exploration_stack[-exploration_depth:]
                capt_frame[:, :, :1] = capt_frame[:, :, :1] / 180.0
                capt_frame[:, :, 1:] = capt_frame[:, :, 1:] / 255.0
                # if frame_stack == deque([]):
                #     zero_frame = np.zeros(capt_frame.shape)
                #     frame_stack.append(zero_frame)
                frame_stack.append(capt_frame)
                if counter > stack_depth:
                    frame_stack.popleft()
                    frame_array = np.array(frame_stack)
                    # frame_array = np.diff(frame_array, axis=0)
                    probs = get_probs(x, frame_array, probs_stack, total_score)
                    probs = probs + 1e-9
                    probs = probs / probs.sum()
                    sign_stats = 1.0
                    stats_val = np.abs(stats_score)
                    if stats_val > 0.0:
                        sign_stats = stats_score / stats_val
                        stats_score = sign_stats * np.log(1 + stats_val)
                    part_score += stats_score
                    part_score += damage_score
                    if mode == "stochastic":
                        command_choice = np.random.choice(
                            n_controls, p=np.squeeze(probs)
                        )
                    elif mode == "deterministic":
                        command_choice = np.argmax(probs)
                    chosen_command = control_options[command_choice]
                    if chosen_command == "r":
                        part_score -= 1.0
                    elif chosen_command == ["1", "2", "3", "4", "do nothing"]:
                        part_score -= 0.5
                    command_stack.append(chosen_command)
                    ref_P = np.squeeze(probs).copy()
                    adj_P = 0.01 * np.ones(ref_P.shape)
                    adj_P[command_choice] = 1.0
                    adj_P = adj_P / adj_P.sum()
                    probs_stack.popleft()
                    probs_stack.append(adj_P)
                    if len(command_stack) > int(command_reps):
                        command_stack.popleft()
                        command_array = np.array(command_stack)
                        comms, comm_counts = np.unique(
                            command_array, return_counts=True
                        )
                        if np.max(comm_counts) > max_repetition * command_reps:
                            step_score = step_score * 50.0
                            stuck = True
                    Walking = activate_control(chosen_command, Walking)
                if counter > max_steps:
                    print("too many steps")
                    stuck = True
                step_score /= max_steps
                part_score /= max_steps
                partial_score = (
                    part_score + exploration_score + boss_search_score + step_score
                )
                partial_score = partial_score  # * (1 - probs_score)
                final_step_score += step_score
                total_score += partial_score
                final_part_score += part_score
                final_boss_search_score += boss_search_score
                final_exploration_score += exploration_score
            if stuck:
                reset_trial()
                print("survival score: ", final_step_score)
                print("general score: ", final_part_score)
                print("boss detection score: ", final_boss_search_score)
                print("wanderlust score: ", final_exploration_score)
                print("total score: ", total_score)
                print("\n")
                total_score = -1.0 * total_score
                for Key in ["a", "w", "s", "d", "space"]:
                    pydirectinput.keyUp(Key)
                return total_score
