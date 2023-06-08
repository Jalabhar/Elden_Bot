import numpy as np
import pandas as pd
import pyautogui
import pydirectinput
import pygetwindow
from scipy.signal import convolve
from collections import deque
import time
import skimage.measure

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

def reset_trial():
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('s')
    pydirectinput.keyUp('d')
    pydirectinput.keyUp('space')
    pydirectinput.press('g')
    pydirectinput.press('f')
    pydirectinput.press('f')
    pydirectinput.press('e')
    pydirectinput.press('e')



def walk_to_boss(): # margit
    time.sleep(1.0)
    pydirectinput.keyDown('s')
    time.sleep(0.2)
    pydirectinput.keyUp('s')
    pydirectinput.keyDown('w')
    pydirectinput.keyDown('d')
    time.sleep(1.0)
    pydirectinput.keyUp('d')
    pydirectinput.keyDown('a')
    time.sleep(0.1)
    pydirectinput.keyUp('a')
    time.sleep(5.0)
    pydirectinput.keyUp('w')
    pydirectinput.keyDown('w')
    pydirectinput.keyDown('d')
    time.sleep(0.9)
    pydirectinput.keyUp('d')
    pydirectinput.keyUp('w')
    pydirectinput.keyDown('d')
    time.sleep(0.4)
    pydirectinput.keyUp('d')
    pydirectinput.keyDown('w')
    time.sleep(0.2)
    pydirectinput.keyUp('w')
    pydirectinput.press('e')
    time.sleep(0.2)
    pydirectinput.press('e')

def get_boss_life(x1,y1,x2,y2):
    bar = (x1+190,y1+340,x2-190,y2-80)
    life_bar = np.array(pyautogui.screenshot(region=(bar)))
    low_thresh =90
    high_thresh =120
    life_ref = life_bar.copy()
    life_ref = life_ref[:,:,0]
    life_bar[life_ref<low_thresh] = 0.0
    life_bar[life_ref>high_thresh] = 0.0
    life_bar[life_bar!=0] = 255.0
    life_value = life_bar.mean()
    life_value = life_value/255.0
    return life_value
    
def get_values(x1,y1,x2,y2):
    red_bar = (x1+65,y1+50,x2-670,y2-480)
    blue_bar = (x1+65,y1+57,x2-670,y2-480)
    green_bar = (x1+65,y1+65,x2-670,y2-480)
    life_bar = np.array(pyautogui.screenshot(region=(red_bar)))
    mana_bar = np.array(pyautogui.screenshot(region=(blue_bar)))
    stamina_bar = np.array(pyautogui.screenshot(region=(green_bar)))
    low_thresh =90
    high_thresh =120
    life_ref = life_bar.copy()
    life_ref = life_ref[:,:,0]
    life_bar[life_ref<low_thresh] = 0.0
    life_bar[life_ref>high_thresh] = 0.0
    life_bar[life_bar!=0] = 255.0
    low_thresh =80
    high_thresh =100
    mana_ref = mana_bar.copy()
    mana_ref = mana_ref[:,:,2]
    mana_bar[mana_ref<low_thresh] = 0.0
    mana_bar[mana_ref>high_thresh] = 0.0
    mana_bar[mana_bar!=0] = 255.0
    low_thresh =70
    high_thresh =90
    stamina_ref = stamina_bar.copy()
    stamina_ref = stamina_ref[:,:,1]
    stamina_bar[stamina_ref<low_thresh] = 0.0
    stamina_bar[stamina_ref>high_thresh] = 0.0
    stamina_bar[stamina_bar!=0] = 255.0
    life_value = life_bar.mean()/255.0
    mana_value = mana_bar.mean()/255.0
    stamina_value = stamina_bar.mean()/255.0
    return life_value,mana_value,stamina_value

def activate_control(command,walking,shield):
    command=command.lower()
    control_action = command.split('+')
    n_actions = len(control_action)
    walking_time_limit = 2.0
    if n_actions==2:
        pydirectinput.keyDown(control_action[0])
        time.sleep(0.1)
        pydirectinput.keyDown(control_action[1])
        time.sleep(0.2)
        pydirectinput.keyUp(control_action[1])
        time.sleep(0.1)
        pydirectinput.keyUp(control_action[0])
        if control_action[0] == 's' and control_action[1] == 'x':
            time.sleep(0.1)
            pydirectinput.press('q')
    elif n_actions == 1:
        base_action = control_action[0]
        if base_action =='w':
            if not walking:
                pydirectinput.keyDown(base_action) 
                started_walking = time.time()
                walking = True
            elif walking:
                pydirectinput.keyUp(base_action)
                walking = False
        elif base_action=='lmb':
            pydirectinput.mouseDown(button='left')
            time.sleep(0.1)
            pydirectinput.mouseUp(button='left')
        elif base_action=='rmb':
            pydirectinput.mouseDown(button='right')
            time.sleep(0.1)
            pydirectinput.mouseUp(button='right')
        elif base_action in ['do nothing']:
            pass
        else:
            pydirectinput.press(base_action)
            if base_action in ['a','s','d']:
                    pydirectinput.press('q')
    if walking:
        try:
            time_walking = time.time() - started_walking
        except:
            time_walking = 0.0
        if (time_walking)>walking_time_limit:
            pydirectinput.keyUp('w')
    return walking,shield

def stable_sigmoid(x):

    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def mish_activation(x):
    rng = np.maximum(np.abs(x.max()),np.abs(x.min()))
    scaled_x = x/rng
    Result = scaled_x*np.tanh(np.log(1+np.exp(scaled_x)))
    return Result

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_probs(X,stack,prior_probs,score):
    prior_probs = np.array(prior_probs)
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
    X_stack = X[:A]
    X_kernel = X[A:A+B]
    X_W_1 = X[A+B:A+B+C]
    X_B_1 = X[A+B+C:A+B+C+D]
    X_W_2 = X[A+B+C+D:A+B+C+D+E]
    X_B_2 = X[A+B+C+D+E:A+B+C+D+E+F]
    X_W_3 = X[A+B+C+D+E+F:A+B+C+D+E+F+G]
    X_B_3 = X[A+B+C+D+E+F+G:A+B+C+D+E+F+G+H]
    X_W_4 = X[A+B+C+D+E+F+G+H:A+B+C+D+E+F+G+H+I]
    X_B_4 = X[A+B+C+D+E+F+G+H+I:A+B+C+D+E+F+G+H+I+J]
    X_W_P = X[A+B+C+D+E+F+G+H+I+J:A+B+C+D+E+F+G+H+I+J+K]
    X_B_P = X[A+B+C+D+E+F+G+H+I+J+K:A+B+C+D+E+F+G+H+I+J+K+L]
    X_kernel_2 = X[-M:]
    stack_kernel =X_stack.reshape((stack_depth,1,1,n_channels))
    stack_kernel /= stack_kernel.sum()
    kernel = X_kernel.reshape((n_filters,kernel_size,kernel_size,1))
    kernel /= kernel.sum()
    kernel_2 = X_kernel_2.reshape((n_filters,1,1,1))
    kernel_2 /= kernel_2.sum()
    W_1 = X_W_1.reshape((n_hidden_1,(width+2)*(height+2)))
    B_1 = X_B_1.reshape((n_hidden_1,1)) 
    W_2 = X_W_2.reshape((n_hidden_2,n_hidden_1))
    B_2 = X_B_2.reshape((n_hidden_2,1))
    W_3 = X_W_3.reshape((n_hidden_3,n_hidden_2)) 
    B_3 = X_B_3.reshape((n_hidden_3,1)) 
    W_4 = X_W_4.reshape((n_controls,n_hidden_3)) 
    B_4 = X_B_4.reshape((n_controls,1)) 
    W_P = X_W_P.reshape(n_controls,probs_depth)
    B_P = X_B_P.reshape(n_controls,1)
    compressed_stack = convolve(stack_kernel,stack,'valid')
    out_stack = compressed_stack
    out_stack = mish_activation(out_stack)
    out_stack = (out_stack-out_stack.mean())/out_stack.std()
    out_frame = convolve(out_stack,kernel,'full')
    out_frame = mish_activation(out_frame)
    out_frame = (out_frame-out_frame.mean())/out_frame.std()
    out_frame = convolve(out_frame,kernel_2,'valid')
    out_frame = mish_activation(out_frame)
    out_frame = (out_frame-out_frame.mean())/out_frame.std()
    out_frame = skimage.measure.block_reduce(out_frame,scale_factor,np.max)
    flat = np.expand_dims(out_frame.flatten(),axis=-1)
    noise = np.random.uniform(-0.05,0.05,flat.shape)
    flat = flat+noise
    P_1 = np.dot(W_1,flat)
    R_1 = P_1+B_1
    R_1 = mish_activation(R_1)
    R_1 = (R_1-R_1.mean())/R_1.std()
    P_2 = np.dot(W_2,R_1)
    R_2 = P_2 + B_2
    R_2 = mish_activation(R_2)
    R_2 = (R_2-R_2.mean())/R_2.std()
    P_3 = np.dot(W_3,R_2)
    R_3 = P_3+B_3
    R_3 = mish_activation(R_3)
    R_3 = (R_3-R_3.mean())/R_3.std()
    P_4 = np.dot(W_4,R_3)
    R_4 = P_4+B_4
    if score!=0:
        R_4 = R_4*score
    R_4 = mish_activation(R_4)
    R_4 = (R_4-R_4.mean())/R_4.std()
    weighted_prev = np.dot(W_P,prior_probs)
    weighted_prev = (weighted_prev-weighted_prev.mean())/weighted_prev.std()
    weighted_prev = weighted_prev+B_P
    weighted_prev = mish_activation(weighted_prev)
    P = np.dot(weighted_prev,R_4)
    P = np.squeeze(P)
    P = (P-P.mean())/P.std()
    scaled_P = softmax(P)
    return scaled_P

def run_bot(x):
    controls_df = pd.read_csv('Custom_Controls.csv')
    stuck = False
    control_options = list(controls_df['Control'].values)
    control_options = np.array(control_options)
    n_controls = len(control_options)
    probs_array = np.ones((probs_depth,n_controls))
    probs_array = probs_array/probs_array.sum(axis=0)
    probs_stack = deque(probs_array)
    max_steps = 5000
    is_alive =False
    is_dead = False
    total_score = 1.0
    Life,Mana,Stamina = 0,0,0
    boss_life = 0
    my_window_title = 'ELDEN RING™'
    my_window = pygetwindow.getWindowsWithTitle(my_window_title)[0]
    my_window.activate()
    my_window.moveTo(0,0)
    x1,y1,x2,y2 = my_window._getWindowRect()
    r = (x1+16,y1+40,x2-25,y2-49)
    while is_dead:
        Life,Mana,Stamina = get_values(x1,y1,x2,y2)
        if (255.0*Life)>=4.0:
            is_alive = True
            is_dead = False
    else:
        first_action = True
        frame_stack = deque([])
        counter = 0
        command_stack = deque([])
        command_reps = 20
        max_repetition = 0.4
        mode = 'deterministic'
        Walking = False
        Shield = False
        r2 = (x1+700,y1+430,x2-720,y2-450)
        runes_capture = pyautogui.screenshot(region=(r2))
        runes_capture = np.array(runes_capture)
        rune_count = 0.0
        step_score = 0
        while True:
            step_score += 1/(1+counter)
            part_score = 0.0
            Last_Life,Last_Mana,Last_Stamina = Life,Mana,Stamina
            # last_boss_life = boss_life
            boss_life = get_boss_life(x1,y1,x2,y2)
            Life,Mana,Stamina = get_values(x1,y1,x2,y2)
            Life_score = Life-Last_Life
            Mana_score = Mana-Last_Mana
            Stamina_score = Stamina-Last_Stamina
            stats_score = Life_score+Mana_score+Stamina_score
            damage_score = -1*boss_life
            if np.abs(damage_score)<1e-4:
                damage_score = 0.0
            if (255.0*Life)<0.2:
                is_alive=False
                is_dead = True
                for Key in ['a','w','s','d','space']:
                    pydirectinput.keyUp(Key)
                time.sleep(10.0)
                if total_score == np.inf:
                    total_score = 1.0
                break
            elif 4.0>(255.0*Life)>0.2:
                is_alive = False
                is_dead = False
            elif (255.0*Life)>=4.0:
                is_alive = True
                is_dead = False
            if not is_alive:
                pydirectinput.keyUp('w')
            if counter%16==15:
                time.sleep(0.2)
            if is_alive and not stuck:
                counter += 1
                capt_frame = np.array(pyautogui.screenshot(region=(r)))
                capt_frame = capt_frame/255.0
                blur_radius = 5
                blur_kernel = np.ones((blur_radius,blur_radius,1))
                blur_kernel = blur_kernel/blur_kernel.sum()
                capt_frame = convolve(capt_frame,blur_kernel)
                last_runes_capture = runes_capture
                r2 = (x1+700,y1+430,x2-720,y2-450)
                runes_capture = pyautogui.screenshot(region=(r2))
                runes_capture = np.array(runes_capture)
                runes_capture[runes_capture<160]=0
                runes_capture[runes_capture>=160]=255
                runes_diff = np.abs(runes_capture-last_runes_capture)
                runes_array = np.array(runes_diff)
                runes_array[runes_array<210]=0
                runes_array[runes_array>=210]=255
                last_rune_count = rune_count
                if is_alive:
                    rune_count = runes_array.mean()
                else:
                    rune_count = 0
                rune_score = np.maximum((rune_count-last_rune_count),0)
                if rune_score<0.05:
                    rune_score=0
                rune_score *= 140
                frame_stack.append(capt_frame)
                frame_array = np.array(frame_stack)
                if counter>stack_depth:
                    part_score += rune_score
                    frame_stack.popleft()
                    frame_array = np.array(frame_stack)
                    probs = get_probs(x,frame_array,probs_stack,part_score)
                    adj_P = np.squeeze(probs).copy()
                    probs_stack.popleft()
                    probs_stack.append(adj_P)
                    if first_action:
                        walk_to_boss()
                        time.sleep(1.0)
                        first_action = False
                    part_score += stats_score
                    part_score += damage_score
                    if mode == 'stochastic':
                        command_choice = np.random.choice(n_controls,p=np.squeeze(probs/probs.sum()))
                    elif mode == 'deterministic':
                        command_choice = np.argmax(probs)
                    chosen_command = control_options[command_choice]
                    command_stack.append(chosen_command)
                    if len(command_stack)>int(command_reps):
                        command_stack.popleft()
                        command_array = np.array(command_stack)
                        comms,comm_counts = np.unique(command_array,return_counts=True)
                        if np.max(comm_counts)>max_repetition*command_reps:
                            total_score = total_score/50.0
                            stuck = True
                    Walking,Shield= activate_control(chosen_command,Walking,Shield)
                if counter>max_steps:
                    total_score = total_score/2.0
                    stuck = True
                total_score = step_score  + part_score
                if stuck:
                    reset_trial()
    total_score = -1.0*total_score
    print('you are dead')
    print(total_score)
    pydirectinput.keyDown('e')
    pydirectinput.keyUp('e')
    for Key in ['a','w','s','d','space']:
        pydirectinput.keyUp(Key)
    return total_score
