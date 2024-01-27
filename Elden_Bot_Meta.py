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

command_reps = 10
kernel_size = 3
n_channels = 3
base_width = 791
base_height = 440
exploration_depth = 200
width = 160
height = 88
controls_df = pd.read_csv("Custom_Controls.csv")
control_options = controls_df["Control"].values
for control in control_options:
    control = control.lower()
n_controls = len(control_options)


def generate(N1, a, b):
    inps = np.linspace(1e-2, 1.0, N1)
    value = np.sin((a / inps) ** b)
    return value


def build_bot(n):
    stack_depth, n_filters, n_hidden_1, n_hidden_2, n_hidden_3 = (
        n[0],
        n[1],
        n[2],
        n[3],
        n[4],
    )
    A = int(stack_depth)
    B = int(n_filters * kernel_size * kernel_size * n_channels)
    C = int(n_hidden_1 * width // 4 * height // 4 * n_channels)
    D = int(n_hidden_1 * n_hidden_2)
    E = int(n_hidden_2 * n_hidden_3)
    F = int(n_hidden_3 * n_controls)
    G = n_filters * kernel_size * kernel_size * n_channels
    n_nodes = [A, B, C, D, E, F]
    out = []
    split = len(n_nodes)
    _, seeds, shifts = n[:split], n[split:-split], n[-split:]
    for i, N1 in enumerate(n_nodes):
        W_set = generate(N1, seeds[i], shifts[i])
        # W_set = softmax(W_set)
        out.extend(W_set)
    out = np.asarray(out)
    return out, stack_depth, n_filters, n_hidden_1, n_hidden_2, n_hidden_3


def reset_trial():
    keys_to_release = ["w", "a", "s", "d", "space"]
    keys_to_press = ["g", "f", "f", "e", "e"]

    for key in keys_to_release:
        pydirectinput.keyUp(key)

    for key in keys_to_press:
        pydirectinput.press(key)
    time.sleep(5.0)


def walk_to_boss(boss=None):  # margit
    if boss == "smoldering church":
        time.sleep(0.5)
        pydirectinput.keyDown("w")
        time.sleep(12.0)
        pydirectinput.keyUp("w")
        pydirectinput.press("s")
        pydirectinput.press("q")
        time.sleep(5.0)
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
    """
    Calculates the amount of boss life remaining based on the given coordinates of the boss life bar.

    Args:
        x1 (int): The x-coordinate of the top-left corner of the boss life bar.
        y1 (int): The y-coordinate of the top-left corner of the boss life bar.
        x2 (int): The x-coordinate of the bottom-right corner of the boss life bar.
        y2 (int): The y-coordinate of the bottom-right corner of the boss life bar.

    Returns:
        float: The remaining boss life as a decimal value.
    """
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
    """
    Calculates the value of a given bar in an image.

    Args:
        bar (tuple): The region of interest (ROI) coordinates of the bar in the image.
        lower (tuple): The lower threshold values for the HSV color range.
        upper (tuple): The upper threshold values for the HSV color range.

    Returns:
        float: The calculated value of the bar.
    """

    def get_bar_values(bar, lower, upper):
        """
        Calculates the value of a given bar in an image.

        Args:
            bar (tuple): The region of interest (ROI) coordinates of the bar in the image.
            lower (tuple): The lower threshold values for the HSV color range.
            upper (tuple): The upper threshold values for the HSV color range.

        Returns:
            float: The calculated value of the bar.
        """
        bar_region = np.array(pyautogui.screenshot(region=bar))
        bar_hsv = cv2.cvtColor(bar_region, cv2.COLOR_RGB2HSV)
        bar_mask = cv2.inRange(bar_hsv, lower, upper)
        bar_value = bar_mask.sum() / 255.0
        return bar_value

    red_bar = (x1 + 65, y1 + 50, x2 - 670, y2 - 480)
    blue_bar = (x1 + 65, y1 + 57, x2 - 670, y2 - 480)
    green_bar = (x1 + 65, y1 + 65, x2 - 670, y2 - 480)

    life_lower = np.array([0, 90, 75])
    life_upper = np.array([150, 255, 125])
    life_value = get_bar_values(red_bar, life_lower, life_upper)

    stamina_lower = np.array([6, 52, 24])
    stamina_upper = np.array([74, 255, 77])
    stamina_value = get_bar_values(green_bar, stamina_lower, stamina_upper)

    mana_lower = np.array([120, 52, 24])
    mana_upper = np.array([194, 255, 77])
    mana_value = get_bar_values(blue_bar, mana_lower, mana_upper)

    return life_value, mana_value, stamina_value


def activate_control(command, walking):
    """
    Activates the control based on the given command.

    Args:
        command (str): The command to be executed.
        walking (bool): Flag indicating if the character is currently walking.

    Returns:
        bool: The updated value of the walking flag.

    Raises:
        None

    Examples:
        >>> activate_control("w", walking=True)
        False
        >>> activate_control("e", walking=False)
        True
    """
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


def get_runes_score(last_masked_runes, x1, y1, x2, y2):
    r = (x1 + 718, y1 + 457, x2 - 742, y2 - 480)
    runes_box = pyautogui.screenshot(region=r)
    runes_box = runes_box.convert("HSV")
    runes_box = np.array(runes_box)
    runes_lower = np.array([0, 0, 160])
    runes_upper = np.array([255, 255, 255])
    masked_runes = cv2.inRange(runes_box, runes_lower, runes_upper)
    runes_Score = 0
    if last_masked_runes is not None:
        runes_diff = (masked_runes - last_masked_runes).sum()
        if runes_diff > 300.0:
            runes_Score = runes_diff
            runes_Score = np.log(1 + runes_Score)
            print(runes_Score)
    last_masked_runes = masked_runes
    return runes_Score, last_masked_runes


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


def get_probs(X, stack, stack_depth, n_filters, n_hidden_1, n_hidden_2, n_hidden_3):
    """
    Calculate the probabilities for each class label based on the given input.

    Args:
        X (numpy.ndarray): The input data of shape (n_features,).
        stack (numpy.ndarray): The stack data of shape (stack_depth, 1, 1, n_channels).
        prior_probs (list): The prior probabilities for each class label.
        score (float): The score retrived from last action.

    Returns:
        numpy.ndarray: The probabilities for each class label, scaled and normalized.
    """
    stack_depth = int(stack_depth)
    k_size = int(kernel_size)
    n_filters = int(n_filters)
    n_hidden_1 = int(n_hidden_1)
    n_hidden_2 = int(n_hidden_2)
    n_hidden_3 = int(n_hidden_3)
    num_channels = int(n_channels)
    A = stack_depth
    B = n_filters * k_size * k_size * num_channels
    C = n_hidden_1 * width // 4 * height // 4 * num_channels
    D = n_hidden_1 * n_hidden_2
    E = n_hidden_2 * n_hidden_3
    F = n_hidden_3 * n_controls
    G = n_filters * k_size * k_size * num_channels
    X_stack = X[:A]
    # stack_bias =X[A:A]
    X_kernel = X[A : A + B]
    # kernel_bias =X[A+B:A+B]
    X_W_1 = X[A + B : A + B + C]
    X_W_2 = X[A + B + C : A + B + C + D]
    X_W_3 = X[A + B + C + D : A + B + C + D + E]
    X_W_4 = X[A + B + C + D + E : A + B + C + D + E + F]
    X_kernel_2 = X[-G:]
    stack_kernel = X_stack.reshape((stack_depth, 1, 1, 1))
    stack_kernel = stack_kernel  # * 2.0 - 1.0
    stack_sum = stack_kernel.sum()
    if stack_sum == 0:
        stack_kernel[(stack_depth // 2) + 1, :, :, :] = 1
    stack_kernel = stack_kernel / stack_sum
    kernel = X_kernel.reshape((n_filters, k_size, k_size, num_channels))
    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        kernel[:, : (k_size // 2 + 1), : (k_size // 2 + 1), :] = 1
    kernel = kernel / kernel_sum
    kernel_2 = X_kernel_2.reshape((n_filters, k_size, k_size, num_channels))
    kernel_2_sum = kernel_2.sum()
    if kernel_2_sum == 0:
        kernel_2[:, : (k_size // 2 + 1), : (k_size // 2 + 1), :] = 1
    kernel_2 = kernel_2 / kernel_2_sum
    W_1 = X_W_1.reshape(((height // 4) * (width // 4) * num_channels), n_hidden_1)
    W_2 = X_W_2.reshape((n_hidden_1, n_hidden_2))
    W_3 = X_W_3.reshape((n_hidden_2, n_hidden_3))
    W_4 = X_W_4.reshape((n_hidden_3, n_controls))
    out_stack = convolve(stack_kernel, stack, "valid")
    out_stack = (out_stack - out_stack.mean()) / out_stack.std()
    out_frame = convolve(out_stack, kernel, "same")
    out_frame = (out_frame - out_frame.mean()) / out_frame.std()
    out_frame = out_frame[:, ::2, ::2, :]
    out_frame = convolve(out_frame, kernel_2, "same")
    out_frame = (out_frame - out_frame.mean()) / out_frame.std()
    out_frame = out_frame[:, ::2, ::2, :]
    flat = np.expand_dims(out_frame.flatten(), axis=0)
    flat = (flat - flat.mean()) / flat.std()
    P_1 = np.dot(flat, W_1)
    P_1 = mish_activation(P_1)
    P_1 = np.dot(P_1, W_2)
    P_1 = mish_activation(P_1)
    P_1 = np.dot(P_1, W_3)
    P_1 = mish_activation(P_1)
    P_1 = np.dot(P_1, W_4)
    P_1 = mish_activation(P_1)
    P = np.squeeze(P_1)
    scaled_P = softmax(P)
    return scaled_P


def run_bot(Y, max_steps=1000):
    network_size_penalty = np.sqrt(Y[:6].sum())
    x, stack_depth, n_filters, n_hidden_1, n_hidden_2, n_hidden_3 = build_bot(Y)
    weights_penalty = np.linalg.norm(x) / np.sqrt(x.shape[0])
    controls_df = pd.read_csv("Custom_Controls.csv")
    weighted_boss_life = 0.0
    stuck = False
    is_alive = False
    is_dead = False
    boss_found = False
    boss_lost = False
    last_masked_runes = None
    mode = "deterministic"
    exploration_stack = []
    control_options = controls_df["Control"].values
    frame_stack = deque([])
    counter = 0
    command_stack = deque([])
    Walking = False
    novelty_threshold = 1e-12
    final_exploration_score = 0
    final_boss_search_score = 0.0
    final_part_score = 0.0
    final_assertiveness_score = 0.0
    final_rune_score = 0.0
    partial_score = 0.0
    exploration_score = 0.0
    stats_score = 0
    # probs_score = 0.0
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
        if (Life) >= 300.0:
            is_alive = True
            is_dead = False
        else:
            time.sleep(0.5)
    else:
        first_action = True
        if first_action:
            walk_to_boss()
            time.sleep(1.0)
            first_action = False
        while True:
            Last_Life, Last_Mana, Last_Stamina = Life, Mana, Stamina
            boss_search_score = 0.0
            part_score = 0.0
            partial_score = 0.0
            assertiveness_score = 0.0
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
            stats_score = 255.0 * (Life_score + Mana_score + Stamina_score)
            rune_score, last_masked_runes = get_runes_score(
                last_masked_runes, x1, y1, x2, y2
            )
            if stats_score == 0.0:
                stats_sign = 1.0
            else:
                stats_sign = stats_score / np.abs(stats_score)
            stats_score = stats_sign * np.abs(stats_score)
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
            damage_score = damage_score_sign * np.sqrt(np.abs(damage_score))
            damage_score = 100.0 * damage_score
            if (Life) < 2.0:
                is_alive = False
                is_dead = True
                time.sleep(2.0)
                if not is_alive:
                    capt_frame = pyautogui.screenshot(region=r)
                    c_frame = capt_frame.convert("HSV")
                    c_array = np.array(c_frame)
                    c_array = c_array / 255.0
                    if c_array.mean() < 0.1:
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
            if counter % 16 == 15:
                time.sleep(0.2)
            if is_alive and not stuck:
                counter += 1
                capt_frame = pyautogui.screenshot(region=r)
                capt_frame = capt_frame.resize((width, height))
                c_frame = capt_frame.convert("HSV")
                c_array = np.array(c_frame)
                if len(exploration_stack) == 0:
                    exploration_stack.append(c_array)
                else:
                    ac_diff = np.inf
                    for explored_frame in exploration_stack:
                        frame_diff = np.linalg.norm(c_array - explored_frame)
                        frame_diff = frame_diff / (width * height * n_channels)
                        ac_diff = np.minimum(frame_diff, ac_diff)
                    novelty_value = ac_diff
                    if novelty_value > novelty_threshold:
                        exploration_stack.append(c_array)
                        exploration_score += novelty_value  # ** 0.5
                        novelty_threshold = 0.05 * np.asarray(exploration_stack).std()
                        exploration_stack = exploration_stack[-exploration_depth:]
                c_array = c_array / 255.0
                frame_stack.append(c_array)
                if len(frame_stack) > (stack_depth + 1):
                    frame_stack.popleft()
                    frame_array = np.array(frame_stack)
                    frame_array = np.diff(frame_array, axis=0)
                    probs = get_probs(
                        x,
                        frame_array,
                        stack_depth,
                        n_filters,
                        n_hidden_1,
                        n_hidden_2,
                        n_hidden_3,
                    )
                    # print(probs)
                    part_score += stats_score
                    part_score += damage_score
                    det_choice = np.argmax(probs)
                    stoch_choice = np.random.choice(n_controls, p=np.squeeze(probs))
                    if mode == "stochastic":
                        command_choice = stoch_choice
                    elif mode == "deterministic":
                        command_choice = det_choice
                    stoch_index = np.zeros(n_controls)
                    det_index = np.zeros(n_controls)
                    stoch_index[stoch_choice] = 1
                    det_index[det_choice] = 1
                    assertiveness_score = np.sum(stoch_index * det_index)
                    chosen_command = control_options[command_choice]
                    command_stack.append(chosen_command)
                    if len(command_stack) > int(command_reps):
                        command_stack.popleft()
                        command_array = np.array(command_stack)
                        comms, comm_counts = np.unique(
                            command_array, return_counts=True
                        )
                        if np.max(comm_counts) > command_reps:
                            stuck = True
                    Walking = activate_control(chosen_command, Walking)
                if counter > max_steps:
                    print("too many steps")
                    stuck = True
                part_score /= max_steps
                partial_score = (
                    part_score
                    + exploration_score
                    + boss_search_score
                    + assertiveness_score
                    + rune_score
                )
                total_score += partial_score
                final_part_score += part_score
                final_boss_search_score += boss_search_score
                final_exploration_score += exploration_score
                final_assertiveness_score += assertiveness_score
                final_rune_score += rune_score
            if is_dead:
                break
            if stuck:
                break
    for Key in ["a", "w", "s", "d", "space"]:
        pydirectinput.keyUp(Key)
    print("general score: ", final_part_score)
    print("rune score: ", final_rune_score)
    print("boss detection score: ", final_boss_search_score)
    print("wanderlust score: ", final_exploration_score)
    print("confidence score: ", final_assertiveness_score)
    print("weights penalty: ", weights_penalty)
    print("size penalty: ", network_size_penalty)
    total_score = total_score - weights_penalty - network_size_penalty
    print("total score: ", total_score)
    print("\n")
    if stuck:
        reset_trial()
    total_score = -1.0 * total_score
    return total_score
