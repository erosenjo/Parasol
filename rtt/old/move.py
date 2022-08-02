from random import randint
import math
from random import random
from random import choice
from random import randrange
import json

def move_simple(states, measurements, iteration, groundtruth):
    x = abs((measurements[-1] - groundtruth)/groundtruth)
    rows = states[-1][0]
    cols = states[-1][1]
    if iteration > 10 and x<.1:
        return -1
    elif x >.1:
        if rows < 10:
            rows += 1
        if cols < 4096:
            cols = 2**randint(math.log(cols,2)+1,12)
    else:
        if rows > 1:
            rows -= 1
        if cols > 2:
            cols = 2**randint(1,math.log(cols,2)-1)
    hash_w = math.log(cols,2)
    return [rows,int(cols),int(hash_w)]


# TODO:: STARFLOW
# s_slots, l_slots need to be power of 2
# s_slots > l_slots
# max_short_idx needs to be power of 2 - 1 (num_short = max + 1)
# max_short_idx + 1 < num_long
# bounds for random:
#   s_slots 128 - 8192
#   l_slots 64 - 4096
#   max_short_idx 1 - 3
#   num_long 4 - 8
# cost for solution is eviction ratio - if ratio is below some value, then solution is good
# once we're below that value, pick the solution that uses smallest memory?


# randomly choose cms rows/cols, given some bounds
def sim_move():
    tables = randint(5,8)
    table_size = 2**randint(6,16)
    thresh = randrange(10000000, 100000000, 5000000) # 10-100 ms, increments of 5ms
    return tables, table_size, thresh

# single step of simulated_annealing
# we call this after we have TWO COSTS (call simple initially?)
def simulated_annealing_step(curr_state, curr_cost, new_state, new_cost, best_state, best_cost, temp, iteration, step_size, bounds):
    # COST CALCULATION
    if new_cost < best_cost or (new_cost==best_cost and (new_state[0]*new_state[1])<=(best_state[0]*best_state[1])):
    #if new_cost < best_cost or (new_cost==best_cost and new_state[1]<=best_state[1])
        best_state = new_state
        best_cost = new_cost

    diff = new_cost - curr_cost
    t = temp / float(iteration+1)
    if t==0:
        print("ZERO TEMP")
        print(iteration)
        print(temp)
        quit()
    try:
        metropolis = math.exp(-diff/t)
    except OverflowError:
        metropolis = float('inf')
    if diff < 0 or random() < metropolis:
        curr_state, curr_cost = new_state, new_cost

    # gen next step
    # random value w/in bounds
    #new_state = sim_move()
    rows = curr_state[0]+randint(-1*bounds[0],bounds[0])*step_size[0]
    if rows < 1:
        rows = 1
    elif rows > 10:
        rows = 10
    cols = 2**(math.log2(curr_state[1])+randint(-1*bounds[1],bounds[1]))*step_size[1]
    if cols < 2:
        cols = 2
    elif cols > 2048:
        cols = 2048
    new_state = [rows, int(cols)]

    return curr_state, curr_cost, new_state, best_state, best_cost, t


# bayesian optimization



'''
# write new sym values to json file - cols, rows, logcols
logcols = math.log2(cols)
concretes = {}
concretes["sizes"] = {"logcols":logcols, "rows":rows}
concretes["symbolics"] = {"cols":cols}
with open('cms_sym.symb', 'w') as f:
    json.dump(concretes, f, indent=4)

'''




