import math, json, sys, copy, time, itertools
from random import randint, random, getrandbits, choice
from interp_sim import gen_cost, compile_num_stages, layout, dfg
import numpy as np
#from scipy.optimize import basinhopping
import pickle
#from search_ilp import solve
from treelib import Node, Tree
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge

# CONSTANTS
single_stg_mem = 143360 # max number of elements for 32-bit array
single_stg_mem_log2 = 131072 # closest power of 2 for single_stg_mem (most apps require num elements to be power of 2)
single_stg_mem_log2_pairarray = 65536

# helper funcs
'''
# narrow down best sols to 1, using priority
# is there a better/simpler way to do this?????
# NOTE: might want to save original list of solutions, in case one we choose doesn't compile to tofino (more sols to try before we're forced to rerun optimize)
def prioritize(best_sols, opt_info):
    prefs_sym = (opt_info["symbolicvals"]["priority"].keys())
    if len(best_sols) > 1:  # multiple best sols, use priority prefs
        best_sol = [best_sols[0],0]
        for v in prefs_sym: # start w highest priority symbolic, narrow down best sols until we reach 1
            if len(best_sols) == 1: # we've picked a single solution, no need to continue
                break
            to_remove = []
            pref_direction = opt_info["symbolicvals"]["priority"][v]
            for s in range(1,len(best_sols)):   # find solutions that are suboptimal according to priority (higher, lower)
                if pref_direction == "higher":
                    if best_sols[s][v] > best_sol[0][v]:
                        to_remove.append(best_sol[1])
                        best_sol = [best_sols[s],s]
                        continue
                    elif best_sols[s][v] < best_sol[0][v]:
                        to_remove.append(s)
                        continue
                elif pref_direction == "lower":
                    if best_sols[s][v] < best_sol[0][v]:
                        to_remove.append(best_sol[1])
                        best_sol = [best_sols[s],s]
                        continue
                    elif best_sols[s][v] > best_sol[0][v]:
                        to_remove.append(s)
                        continue
                else:
                    print("invalid preference, must be higher or lower")
                    quit()

            to_remove.sort(reverse=True)    # throw out solutions that don't match priority
            to_remove=list(set(to_remove))  # get rid of duplicates
            for i in to_remove:
                best_sols.pop(i)
            best_sol[1] = best_sols.index(best_sol[0])  # in case things get shuffled around after pops
    return best_sol[0]
'''
# helper for sim annealing, rounds to closest power of 2
# copied from: https://stackoverflow.com/questions/28228774/find-the-integer-that-is-closest-to-the-power-of-two
def closest_power(x):
    possible_results = math.floor(math.log2(x)), math.ceil(math.log2(x))
    return 2**min(possible_results, key= lambda z: abs(x-2**z))

# if we have rule-based vars, set them with this function
def set_rule_vars(opt_info, symbolics_opt):
    if "rules" not in opt_info["symbolicvals"]:
        return symbolics_opt
    for rulevar in opt_info["symbolicvals"]["rules"]:
        rule = opt_info["symbolicvals"]["rules"][rulevar].split()
        for v in range(len(rule)):
            # if this is a variable name, replace it with the variable value so we can evaluate the expression
            if rule[v] in opt_info["symbolicvals"]["symbolics"] or rule[v] in opt_info["symbolicvals"]["sizes"]:
                # we have to split into 2 cases here bc log variables won't be in symbolics_opt at this point
                # those get written when we gen the .symb file (when we call get_cost)
                if rule[v] in opt_info["symbolicvals"]["logs"]:
                    rule[v] = str(int(math.log2(symbolics_opt[opt_info["symbolicvals"]["logs"][rule[v]]])))
                else:
                    rule[v] = str(symbolics_opt[rule[v]])
        #print("BEFORE RULEVAR:", rulevar, "VAL:", symbolics_opt[rulevar])
        symbolics_opt[rulevar] = eval(''.join(rule))
        #print("RULEVAR:", rulevar, "VAL:", symbolics_opt[rulevar])
    return symbolics_opt


# RANDOM OPTIMIZATION
# randomly choose, given some var
def gen_next_random_nopreprocessing(symbolics_opt, logs, bounds, structchoice, structinfo, opt_info):
    new_vars = {}
    exclude = False
    if structchoice:
        new_vars[structinfo["var"]] = bool(getrandbits(1))
        if str(new_vars[structinfo["var"]]) in structinfo:
            exclude = True
    for var in symbolics_opt:
        if var not in bounds:   # it's a rule-based var (or we forgot to add bounds)
            new_vars[var] = 0
            continue
        if structchoice and var==structinfo["var"]: # we've handled this above, don't do it again
            continue
        if exclude and var in structinfo[str(new_vars[structinfo["var"]])]: # exlucding this var for struct
            new_vars[var] = symbolics_opt[var]
            continue
        if var in logs.values(): # this var has to be multiple of 2
            new_vars[var] = 2**randint(int(math.log2(bounds[var][0])),int(math.log2(bounds[var][1])))
            continue
        new_vars[var] = randint(bounds[var][0],bounds[var][1])

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        new_vars = set_rule_vars(opt_info, new_vars)

    return new_vars


def gen_next_random_preprocessed(tree, symbolics_opt, solutions, opt_info):
    sol_choice = choice(solutions)
    candidate_index = solutions.index(sol_choice)
    symbolics_opt = set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree, opt_info)

    # set non-resource vars
    for nonresource in opt_info["optparams"]["non_resource"]:
        symbolics_opt[nonresource] = choice(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        symbolics_opt = set_rule_vars(opt_info, symbolics_opt)


    return symbolics_opt, candidate_index


def random_opt(symbolics_opt, opt_info, o, timetest,
               bounds_tree=None, solutions=[]):
    # get total number of solutions, so we know if we've gone through them all
    total_sols = 1
    if not solutions:
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bounds = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bounds[0])), int(math.log2(bounds[1]))+1))
                total_sols *= vals
                continue
            vals = len(range(bounds[0], bounds[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            total_sols *= vals
    else:   # we've preprocessed, used those solutions to calc total
        for nonresource in opt_info["optparams"]["non_resource"]:
            total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
        total_sols *= len(solutions)
    print(total_sols)

    # start w/ randomly chosen values
    if not solutions:
        symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], False, {}, opt_info)
    else:
        symbolics_opt,_ = gen_next_random_preprocessed(bounds_tree, symbolics_opt, solutions, opt_info)

    starting = copy.deepcopy(symbolics_opt)
    num_sols_time = {}
    time_cost = {}

    iterations = 0
    # init best solution as starting, and best cost as inf
    best_sols = [copy.deepcopy(symbolics_opt)]
    best_cost = float("inf")
    # keep track of sols we've already tried so don't repeat
    tested_sols = []
    # we need bounds for random
    bounds = {}
    if "bounds" in opt_info["symbolicvals"]:    # not necessarily required for every opt, but def for random
        bounds = opt_info["symbolicvals"]["bounds"]
    else:
        sys.exit("random opt requires bounds on symbolics")
    # decide if we're stopping by time, iterations, or both (whichever reaches thresh first)
    iters = False
    simtime = False
    iter_time = False
    if "stop_iter" in opt_info["optparams"]:
        iters = True
    if "stop_time" in opt_info["optparams"]:
        simtime = True
    if iters and simtime:
        iter_time = True

    structchoice = False
    structinfo = {}
    if "structchoice" in opt_info:
        structchoice = True
        structinfo = opt_info["structchoice"]

    testing_sols = []
    testing_eval = []
    # start loop
    start_time = time.time()
    while True:
        # check iter/time conditions
        if iters or iter_time:
            if iterations >= opt_info["optparams"]["stop_iter"]:
                break
        if simtime or iter_time:
            if (time.time()-start_time) >= opt_info["optparams"]["stop_time"]:
                break

        curr_time = time.time()
        # TIME TEST (save sols/costs we've evaled up to this point)
        if timetest:
            # 5 min (< 10)
            if 300 <= (curr_time - start_time) < 600:
                with open('5min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('5min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["5min"] = len(testing_sols)
                time_cost["5min"] = copy.deepcopy(testing_eval)
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["10min"] = len(testing_sols)
                time_cost["10min"] = copy.deepcopy(testing_eval)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["30min"] = len(testing_sols)
                time_cost["30min"] = copy.deepcopy(testing_eval)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["45min"] = len(testing_sols)
                time_cost["45min"] = copy.deepcopy(testing_eval)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["60min"] = len(testing_sols)
                time_cost["60min"] = copy.deepcopy(testing_eval)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["90min"] = len(testing_sols)
                time_cost["90min"] = copy.deepcopy(testing_eval)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["120min"] = len(testing_sols)
                time_cost["120min"] = copy.deepcopy(testing_eval)
                break

        # get cost
        if not solutions:
            cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "random")
        else:
            cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "ordered")

        # add sol to tested_sols to count it as already evaluated
        # is it stupid to do deepcopy here? can we be smarter about changing symbolics_opt to avoid this? or is it a wash?
        tested_sols.append(copy.deepcopy(symbolics_opt))

        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(cost)

        # if new cost < best, replace best (if stgs <= tofino)
        if cost < best_cost:
            best_cost = cost
            # not sure if this is slow, but these dicts are likely small (<10 items) so shouldn't be an issue
            best_sols = [copy.deepcopy(symbolics_opt)]
        elif cost == best_cost:
            best_sols.append(copy.deepcopy(symbolics_opt))

        # if we've tested every solution, quit
        if len(tested_sols) == total_sols:
            break

        # get next values, but don't repeat one we've already tried
        while symbolics_opt in tested_sols:
            if not solutions:
                symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], structchoice, structinfo, opt_info)
            else:
                symbolics_opt, _ = gen_next_random_preprocessed(bounds_tree, symbolics_opt, solutions, opt_info)
            

        # incr iterations
        iterations += 1

    # if we have multiple solutions equally as good, use priority from user to narrow it down to 1
    #best_sol = prioritize(best_sols,opt_info)

    with open('final_testing_sols_rand.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('final_testing_eval_rand.pkl','wb') as f:
        pickle.dump(testing_eval,f)
    
    num_sols_time["final"] = len(testing_sols)
    time_cost["final"] = copy.deepcopy(testing_eval)

    # return the first solution in list of acceptable sols
    return best_sols[0], best_cost, time_cost, num_sols_time, starting

# NOTE: this is only for non-preprocessing
def gen_next_simannealing_nopreprocessing(structchoice, symbolics_opt, structinfo, curr, step_size, bounds, logvars, opt_info):
    # if we're choosing between structs, random step for choice is coin toss
    if structchoice:
        symbolics_opt[structinfo["var"]] = bool(getrandbits(1))
        #print("NEWBOOL " + str(symbolics_opt[structinfo["var"]]))
        if str(symbolics_opt[structinfo["var"]]) in structinfo:
            exclude = True
        else:
            exclude = False

    # take a step
    for s in step_size:
        if structchoice and s==structinfo["var"]: # we've handled this above, don't do it again
            print("BOOLEAN"+str(symbolics_opt[s]))
            continue
        # don't take a step if we don't need this var for the struct
        if structchoice and exclude and s in structinfo[str(symbolics_opt[structinfo["var"]])]:
            continue
        # random step with gaussian distr, with mean = curr[s] and stddev = step_size[s]
        # could do single call to randn and gen array of values, or get single value at a time
        symbolics_opt[s] = curr[s] + np.random.randn() * step_size[s]
        if structchoice and s==structinfo["var"]:
            if symbolics_opt[s] >= 1:
                symbolics_opt[s] = True
            else:
                symbolics_opt[s] = False
        # if we happen to hit bottom or top of bounds, set to min/max
        # if we're not in bounds, then this SHOULD be a rule-based var
        if s in bounds:
            if symbolics_opt[s] < bounds[s][0]:
                symbolics_opt[s] = bounds[s][0]
            elif symbolics_opt[s] > bounds[s][1]:
                symbolics_opt[s] = bounds[s][1]
        # we'll get a float, so round to nearest acceptable int
        if s in logvars:    # var has to be power of 2
            symbolics_opt[s] = closest_power(symbolics_opt[s])
        else:   # doesn't have to be power of 2, just round to nearest int
            symbolics_opt[s] = round(symbolics_opt[s])

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        symbolics_opt = set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt

# NOTE: this is for preprocessed solutions
def gen_next_simannealing_preprocessed(solutions, opt_info, symbolics_opt, curr, curr_index, tree):
    # take a step for resource-related vars
    # the step corresponds to what index in solutions we're using
    # presumably, solutions that are close in solutions list are more similar
    # TODO: how to determine step size for solutions list index??
    # ^ by default, step size is len of solutions list
    sol_index = round(curr_index + np.random.randn() * len(solutions))
    if sol_index < 0:
        sol_index = 0
    elif sol_index >= len(solutions):
        sol_index = len(solutions) - 1
    sol_choice = solutions[sol_index]

    symbolics_opt = sol_choice
    #print("OLDINDEX", curr_index)
    #print("SOLINDEX", sol_index)
    #print("SOL CHOICE", sol_choice)

    '''
    # OLD, treating non resource separate
    symbolics_opt = set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree)

    # take a step for non-resource vars
    for nr_var in opt_info["optparams"]["non_resource"]:
        if nr_var not in opt_info["optparams"]["stepsize"]:
            exit(nr_var+" not in step size")
        step_size = opt_info["optparams"]["stepsize"][nr_var]
        symbolics_opt[nr_var] = curr[nr_var] + np.random.randn() * step_size
        # if we happen to hit bottom or top of bounds, set to min/max
        # if we're not in bounds, then this SHOULD be a rule-based var
        if nr_var in opt_info["symbolicvals"]["bounds"]:
            if symbolics_opt[nr_var] < opt_info["symbolicvals"]["bounds"][nr_var][0]:
                symbolics_opt[nr_var] = opt_info["symbolicvals"]["bounds"][nr_var][0]
            elif symbolics_opt[nr_var] > opt_info["symbolicvals"]["bounds"][nr_var][1]:
                symbolics_opt[nr_var] = opt_info["symbolicvals"]["bounds"][nr_var][1]
        if nr_var in opt_info["symbolicvals"]["logs"].values():   # gotta be power of 2
            symbolics_opt[nr_var] = closest_power(symbolics_opt[nr_var])
        else:   # doesn't have to be power of 2, just round to nearest int
            symbolics_opt[nr_var] = round(symbolics_opt[nr_var])
    '''

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        symbolics_opt = set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt, sol_index


# SIMULATED ANNEALING
# copied from: https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
# good for objective functions that have single global optima and mult local optima where search might get stuck
# using numpy.random.randn() - gaussian distr (gives us negative vals, so can move up and down)  
# some notes on step size:
#   if the var has large search space, probably want larger step size
#   step size is std dev --> 99% of all steps w/in 3*stepsize of curr var val
#   ^ not exactly true, bc we have to do some rounding (can't have floats)
def simulated_annealing(symbolics_opt, opt_info, o, timetest,
                        bounds_tree=None, solutions=[]):

    temp = opt_info["optparams"]["temp"]
    bounds = opt_info["symbolicvals"]["bounds"]
    step_size = opt_info["optparams"]["stepsize"]
    logvars = opt_info["symbolicvals"]["logs"].values()

    # decide if we're stopping by time, iterations, or both (whichever reaches thresh first)
    iterations = 1
    iters = False
    simtime = False
    iter_time = False
    if "stop_iter" in opt_info["optparams"]:
        iters = True
    if "stop_time" in opt_info["optparams"]:
        simtime = True
    if iters and simtime:
        iter_time = True

    structchoice = False
    structinfo = {}
    if "structchoice" in opt_info:
        structchoice = True
        structinfo = opt_info["structchoice"]

    '''
    # OLD, treating non-resource vars separate from resource sol index
    # get total number of solutions, so we know if we've gone through them all
    total_sols = 1
    if not solutions:
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                total_sols *= vals
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            total_sols *= vals
    else:   # we've preprocessed, used those solutions to calc total
        for nonresource in opt_info["optparams"]["non_resource"]:
                total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
        total_sols *= len(solutions)

    print("TOTAL_SOLS", total_sols)
    '''

    # NEW, enumerating all solutions, optimizing ONLY index value
    #   (aka treating resource and non resource the same)
    total_sols = 1
    all_solutions_symbolics = []
    if not solutions:
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                total_sols *= vals
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            total_sols *= vals
    else:   # we've preprocessed, used those solutions to calc total
        # TODO: create symbolics_opt from solutions and append to all_solutions_symbolics
        for sol_choice in solutions:
            all_solutions_symbolics.append(copy.deepcopy(set_symbolics_from_tree_solution(sol_choice, symbolics_opt, bounds_tree, opt_info)))
        for nonresource in opt_info["optparams"]["non_resource"]:
                #total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                total_sols *= (len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource])) + 1)
                new_sols = []
                for sol_choice in all_solutions_symbolics:
                    #vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                    vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource]))
                    vals.append(opt_info["symbolicvals"]["bounds"][nonresource][1])
                    for v in vals:
                        # update w/ new value
                        sol_choice[nonresource] = v
                        # append to new solution list
                        new_sols.append(copy.deepcopy(sol_choice))

                all_solutions_symbolics = new_sols

        total_sols *= len(solutions)
        # total_sols = len(all_solutions_symbolics)

    # start time
    start_time = time.time()


    # start at randomly chosen values
    # if solutions is an empty list, then we haven't done preprocessing
    if not solutions:
        symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], False, {}, opt_info)
    else:
        '''
        # OLD, treating non-resource vars separate from resource sol index
        # set resource vars
        symbolics_opt, candidate_index = gen_next_random_preprocessed(bounds_tree, symbolics_opt, solutions, opt_info)
        '''
        # NEW, only optimize index var
        symbolics_opt = choice(all_solutions_symbolics)
        candidate_index = all_solutions_symbolics.index(symbolics_opt)


    starting = copy.deepcopy(symbolics_opt)
    num_sols_time = {}
    time_cost = {}


    # generate and evaluate an initial point
    best_sols = [copy.deepcopy(symbolics_opt)]
    # if solutions is empty, no preprocessing, regular sim annealing (need to compile before interpreter)
    if not solutions:
        best_cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "simannealing")
    else:
        best_cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "ordered")
        curr_index = candidate_index

    # current working solution
    curr, curr_cost = copy.deepcopy(symbolics_opt), best_cost


    # list of output for each iteration
    testing_sols = [copy.deepcopy(symbolics_opt)]
    testing_eval = [best_cost]
    
    # keep track of what we've already done, so we don't repeat it
    tested_sols = [copy.deepcopy(symbolics_opt)]

    # run the algorithm
    #for i in range(opt_info["optparams"]["stop_iter"]-1):   # minus 1 bc counting init cost as iteration
    while True:
        if iters or iter_time:
            if iterations >= opt_info["optparams"]["stop_iter"]:
                break
        if simtime or iter_time:
            if (time.time()-start_time) >= opt_info["optparams"]["stop_time"]:
                break

        curr_time = time.time()
        # TIME TEST (save sols/costs we've evaled up to this point)
        if timetest:
            # 5 min (< 10)
            if 300 <= (curr_time - start_time) < 600:
                with open('5min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('5min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["5min"] = len(testing_sols)
                time_cost["5min"] = copy.deepcopy(testing_eval)
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["10min"] = len(testing_sols)
                time_cost["10min"] = copy.deepcopy(testing_eval)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["30min"] = len(testing_sols)
                time_cost["30min"] = copy.deepcopy(testing_eval)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["45min"] = len(testing_sols)
                time_cost["45min"] = copy.deepcopy(testing_eval)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["60min"] = len(testing_sols)
                time_cost["60min"] = copy.deepcopy(testing_eval)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["90min"] = len(testing_sols)
                time_cost["90min"] = copy.deepcopy(testing_eval)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["120min"] = len(testing_sols)
                time_cost["120min"] = copy.deepcopy(testing_eval)
                break


        # if we've tested every solution, quit
        if len(tested_sols) == total_sols:
            break

        # TODO: should we test repeated sols????
        #while symbolics_opt in tested_sols: # don't bother with repeated sols
        if not solutions:
            symbolics_opt = gen_next_simannealing_nopreprocessing(structchoice, symbolics_opt, structinfo, curr, step_size, bounds, logvars, opt_info)
        else:
            # OLD, treating non-resource and resource differently
            #symbolics_opt, candidate_index = gen_next_simannealing_preprocessed(solutions, opt_info, symbolics_opt, curr, curr_index, bounds_tree)
            symbolics_opt, candidate_index = gen_next_simannealing_preprocessed(all_solutions_symbolics, opt_info, symbolics_opt, curr, curr_index, bounds_tree)
        # evaluate candidate point
        if not solutions:
            candidate_cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "simannealing")
        else:
            candidate_cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")

        tested_sols.append(copy.deepcopy(symbolics_opt))

        # incr iteration
        iterations += 1



        # check for new best solution
        if candidate_cost < best_cost:
            # store new best point
            best_sols, best_cost = [copy.deepcopy(symbolics_opt)], candidate_cost
        elif candidate_cost == best_cost:
            best_sols.append(copy.deepcopy(symbolics_opt))

        # difference between candidate and current point evaluation
        diff = candidate_cost - curr_cost

        # calculate temperature for current epoch
        t = temp / float(iterations + 1)

        # calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t)

        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_cost = copy.deepcopy(symbolics_opt), candidate_cost
            if solutions:
                curr_index = candidate_index

        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(candidate_cost)
        print("BEST", best_cost)
        
    #best_sol = prioritize(best_sols,opt_info)
    with open('final_testing_sols_sa.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('final_testing_eval_sa.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    num_sols_time["final"] = len(testing_sols)
    time_cost["final"] = copy.deepcopy(testing_eval)


    # return first sol in list of sols
    return best_sols[0], best_cost, time_cost, num_sols_time, starting

'''
# BASIN HOPPING (with scipy, like sim annealing)
# scipy doesn't let us restrict to ints (not floats), and doesn't let us restrict to powers of 2
def basin_hopping(symbolics_opt, opt_info, o):
    # put symbolic vars in np array
    x0 = np.empty(shape=(len(symbolics_opt)), dtype=int)
    i = 0
    for v in symbolics_opt:
        x0[i] = symbolics_opt[v]
        i+=1
    # extra args that cost func needs
    args ={'args':(symbolics_opt,opt_info, o, True)}
    # as of python3.7, dicts are insertion order, so should be ok to rely on ordering
    res = basinhopping(gen_cost, x0, minimizer_kwargs=args,niter=100)
'''

# EXHAUSTIVE SEARCH
# start from lower bound and go until upper bound
# keep all variables but 1 static, do for all vars
# note that this is impractical and shouldn't actually be used for optimization
def exhaustive(symbolics_opt, opt_info, o, timetest):
    logvars = opt_info["symbolicvals"]["logs"].values()

    # the starting solution values are what we use when we keep a variable static
    starting = copy.deepcopy(symbolics_opt)

    # init best solution as starting, and best cost as inf
    best_sols = [copy.deepcopy(symbolics_opt)]
    best_cost = float("inf")
    # we need bounds
    bounds = {}
    if "bounds" in opt_info["symbolicvals"]:    # not necessarily required for every opt, but def for exhaustive
        bounds = opt_info["symbolicvals"]["bounds"]
    else:
        sys.exit("exhaustive requires bounds on symbolics")

    # log sols/costs and output
    testing_sols = []
    testing_eval = []

    # start loop
    # don't really care about time for exhaustive, but leaving it anyways
    start_time = time.time()
   
    # go through each symbolic value
    for sv in bounds:
        # start at lower bound, stop once we get to upper bound
        # use stepsize to go through range
        for v in range(bounds[sv][0],bounds[sv][1]+opt_info["optparams"]["stepsize"][sv], opt_info["optparams"]["stepsize"][sv]):
            # do corrections for powers of 2
            if sv in logvars:
                symbolics_opt[sv]=closest_power(v)
            else:
                symbolics_opt[sv] = v

            # get cost
            cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "exhaustive")

            # if new cost < best, replace best (if stgs <= tofino)
            if cost < best_cost:
                best_cost = cost
                # not sure if this is slow, but these dicts are likely small (<10 items) so shouldn't be an issue
                best_sols = [copy.deepcopy(symbolics_opt)]
            elif cost == best_cost:
                best_sols.append(copy.deepcopy(symbolics_opt))

            # save costs to write to file later
            testing_sols.append(copy.deepcopy(symbolics_opt))
            testing_eval.append(cost)

        # reset to starting before going to the next variable
        symbolics_opt = starting


    with open('testing_sols_exhaustive.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('testing_eval_exhaustive.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    return best_sols[0], best_cost


# TODO: get resource usage for leaves of tree (ignore pruning for now)
# ONLY return upper bound for now, not resource usage
def get_max_val_efficient(symbolics_opt, var_to_opt, opt_info, log2, memory, fullcompile, pair, dfg):
    # for memory, start @ ub; lb = 32, ub = whatever we find
    # for non memory, either
    #   start @ lb and increase by > 1 until > 12 stgs, then decrease by 1 (multiplicative incr, additive decr)
    #   start at some ub and decrease by some value
    #   --> start at some middle value and decide to incr/decr accordingly
    # we only need to get resource usage for leaves of tree

    increasing = False
    decreasing = False

    while True:
        print("SYM COMPILING:", symbolics_opt)
        # compile it and get num stages used
        if fullcompile:
            stgs_used = compile_num_stages(symbolics_opt, opt_info)
        else:
            if dfg: # use dataflow analysis
                # if we're greater than user-defined bounds, quit
                if var_to_opt in opt_info["symbolicvals"]["bounds"]:
                    if symbolics_opt[var_to_opt] > opt_info["symbolicvals"]["bounds"][var_to_opt][1]:
                        return opt_info["symbolicvals"]["bounds"][var_to_opt][1]
                resources_used = dfg(symbolics_opt, opt_info)
            else:   # use layout script
                resources_used = layout(symbolics_opt, opt_info)
            stgs_used = resources_used["stages"]
        # incr if we still have more stages to use
        if stgs_used == 0:   # this should only happen if there's no num_stages.txt file, or the program actually doesn't use any stages (which indicates an error somewhere)
            sys.exit("lucid to p4 compiler returned 0 stages used")
        elif stgs_used <= 12: # keep going! we (potentially) have more resources to use
            # if this is a memory variable, we only hit this case if the next highest value returns > 12 stgs
            # so this is the best option without going over
            if memory:
                return symbolics_opt[var_to_opt]
            if not increasing and not decreasing:   # this is the first config we're compiling
                increasing = True
            if increasing:  # we started w/ lb, now we need to keep increasing until hit ub
                # if we need log2 of this number, then just go to next multiple of 2
                if log2:
                    symbolics_opt[var_to_opt] *= 2
                else:
                    symbolics_opt[var_to_opt] += 1
            else:   # we started w/ ub, so we found largest compiling value
                return symbolics_opt[var_to_opt]

        else:   # using more than 12 stgs
            # if memory, we need to decrease and try again (bc working backwards from upper bound)
            # TODO: this assumes memory has to be power of 2 --> what to decrease by if not?
            if memory:
                symbolics_opt[var_to_opt] //=2
                if symbolics_opt[var_to_opt] < 32:
                    sys.exit("can't fit at least 32 regs for var " + var_to_opt)
                continue 
            if not increasing and not decreasing:   # this is the first config we're compiling
                decreasing = True
            # if we need log2 of this number, then just go to next smallest multiple of 2
            if log2:
                symbolics_opt[var_to_opt] //= 2
            else:
                symbolics_opt[var_to_opt] -= 1
            if decreasing: # we started w/ ub, keep decr until hit compiling config
                continue
            else:
                return symbolics_opt[var_to_opt]
                


# start at 1 (or whatever lower bound is), keep increasing until we hit max num stgs used
# compile to p4 each time to see how many stgs
# we start at lower bound bc we know that we can't be < 1; harder to start at upper bound bc we don't really know that would be, unless user tells us (which forces them to reason a bit about resources)
# this returns dictionary of either:
#   {sol_tested: stgs_required} (full compile)
#   {sol_tested: {resource: required} } (layout, dfg)
def get_max_val(symbolics_opt, var_to_opt, opt_info, log2, memory, fullcompile, pair, dfg):
    '''
    # if we have a lower bound for var_to_opt, use it
    # otherwise, it'll already be 1
    if var_to_opt in opt_info["symbolicvals"]["bounds"]:
        symbolics_opt[var_to_opt] = opt_info["symbolicvals"]["bounds"][var_to_opt][0]
    else:
        symbolics_opt[var_to_opt] = 1
    '''

    # while we're using < 12 stages, compile, increase val
    # note that we could save time by doing the reverse for memory vals (bc we have hard upper bound)
    # HOWEVER, we wouldn't know how many stgs each value uses if we do the reverse (start at upper bound, stop once stgs <=12)
    # (this is bc we wouldn't compile for memory vals < upper bound, we'd stop once we hit 12 stgs)
    # NOTE: we keep compiling once we hit stgs = 12 bc we could add more resources without using more stgs
    # (this happens specifically w/ memory vars, but maybe could happen with others?) 
    resources = {}  # resources used by each solution we try (FOR LAYOUT: this is ALL resources; FOR FULL COMPILE: this is JUST stgs)
    while True:
        print("SYM COMPILING:", symbolics_opt)
        # compile it and get num stages used
        if fullcompile:
            stgs_used = compile_num_stages(symbolics_opt, opt_info)
            resources_used = stgs_used
        else:
            if dfg: # use dataflow analysis
                # if we're greater than user-defined bounds, quit
                if var_to_opt in opt_info["symbolicvals"]["bounds"]:
                    if symbolics_opt[var_to_opt] > opt_info["symbolicvals"]["bounds"][var_to_opt][1]:
                        return opt_info["symbolicvals"]["bounds"][var_to_opt][1], resources
                resources_used = dfg(symbolics_opt, opt_info)
            else:   # use layout script
                resources_used = layout(symbolics_opt, opt_info)
            stgs_used = resources_used["stages"]
        # incr if we still have more stages to use
        if stgs_used == 0:   # this should only happen if there's no num_stages.txt file, or the program actually doesn't use any stages (which indicates an error somewhere)
            sys.exit("lucid to p4 compiler returned 0 stages used")
        elif stgs_used <= 12: # keep going! we (potentially) have more resources to use
            '''
            # if this is a memory variable, we only hit this case if the next highest value returns > 12 stgs
            # so this is the best option without going over
            if memory:
                best_stgs[symbolics_opt[var_to_opt]] = stgs_used
                return symbolics_opt[var_to_opt], best_stgs
            # otherwise, we can keep going
            '''
            # if this is a memory var, then check if we've reached resource upper bound
            # if yes, then stop bc we can't fit more onto the pipeline
            # if we need log2 of this number, then just go to next multiple of 2
            if log2:
                #best_stgs[symbolics_opt[var_to_opt]] = stgs_used
                resources[symbolics_opt[var_to_opt]] = resources_used
                # TODO: better sol for pair arrays
                if pair:
                    if memory and symbolics_opt[var_to_opt] == single_stg_mem_log2_pairarray:
                        return symbolics_opt[var_to_opt], resources
                    symbolics_opt[var_to_opt] *= 2
                else:
                    if memory and symbolics_opt[var_to_opt] == single_stg_mem_log2:
                        #return symbolics_opt[var_to_opt], best_stgs
                        return symbolics_opt[var_to_opt], resources
                    symbolics_opt[var_to_opt] *= 2
            else:
                #best_stgs[symbolics_opt[var_to_opt]] = stgs_used
                resources[symbolics_opt[var_to_opt]] = resources_used
                if memory and symbolics_opt[var_to_opt] == single_stg_mem:
                    #return symbolics_opt[var_to_opt], best_stgs
                    return symbolics_opt[var_to_opt], resources
                symbolics_opt[var_to_opt] += 1

        else:   # stages > 12, using too many stgs so go back to the previous value we tried
            '''
            # if memory, we need to decrease and try again (bc working backwards from upper bound)
            # TODO: this assumes memory has to be power of 2 --> what to decrease by if not?
            if memory:
                symbolics_opt[var_to_opt] //= 2
                continue
            '''
            # if this is supposed to be multiple of 2, then divide
            if log2:
                symbolics_opt[var_to_opt] //= 2
            # otherwise, just decrement
            else:
                symbolics_opt[var_to_opt] -= 1
            #return symbolics_opt[var_to_opt], best_stgs
            return symbolics_opt[var_to_opt], resources

        '''
        elif stgs_used == 12:    # we hit the limit, this is the value we're using for upper bound
            best_stgs[symbolics_opt[var_to_opt]] = stgs_used
            # if it's memory, we still want to keep going until we either hit ub or stgs > 12
            if memory and log2 and symbolics_opt[var_to_opt] < single_stg_mem_log2:
                symbolics_opt[var_to_opt] *= 2
                continue
            elif memory and not log2 and symbolics_opt[var_to_opt] < single_stg_mem:
                symbolics_opt[var_to_opt] += 1
                continue
            # if it's not memory, return the upper bound
            return symbolics_opt[var_to_opt], best_stgs
        '''


# TODO: is there a better way to do this??? 
# basically we're building a tree
# each node is a concrete choice for a var, and the children are possible choices for the var that's the next level down
def build_bounds_tree(tree, root, to_find, symbolics_opt, opt_info, fullcompile, pair, efficient, dfg):
    #print("ROOT:",root)
    #print("TOFIND:", to_find)
    #print("SYMBOLICSOPT:", symbolics_opt)
    children = tree.children(root)
    for child in children:
        #print("CHILD:", child.tag)
        #if child.tag[0]=="C":
        #    return
        # set the value for this variable
        symbolics_opt[child.tag[0]] = child.tag[1]
        # move down a level to choose a value for the next variable
        build_bounds_tree(tree, child.identifier, to_find, symbolics_opt, opt_info, fullcompile, pair, efficient, dfg)

    if not efficient:
        if not children:
            # find the bounds for the next variable in the list, given the values for the previous
            # first check if this needs to be a power of 2
            # if yes, then make lower bound = 2 instead of 1
            log2 = False
            lb = 1
            if to_find[0] in opt_info["symbolicvals"]["bounds"]:
                lb = opt_info["symbolicvals"]["bounds"][to_find[0]][0]
            if to_find[0] in opt_info["symbolicvals"]["logs"].values():
                log2 = True
                if lb < 2:  # if user gives us higher bound, don't overwrite it
                    lb = 2
 
            # if it's a memory variable, we're starting from the max memory avail for a single register array and then moving down
            # we have a hard upper bound for memory, so it's faster to start from there (we don't have ub for other vars)
            # NOT starting from upper bound anymore; need to compile each solution to estimate stgs (starting at lb)
            if to_find[0] in opt_info["symbolicvals"]["symbolics"]:
                #if log2: symbolics_opt[to_find[0]] = single_stg_mem_log2
                #else: symbolics_opt[to_find[0]] = single_stg_mem
                symbolics_opt[to_find[0]] = lb
                ub, stgs_used = get_max_val(symbolics_opt,to_find[0], opt_info, log2, True, fullcompile, pair, dfg)
                #tree.create_node((to_find[0],ub), parent=root)
            else:
                # keep compiling until we hit max stgs, get ub 
                #print("FIND BOUNDS")
                symbolics_opt[to_find[0]] = lb
                ub, stgs_used = get_max_val(symbolics_opt, to_find[0], opt_info, log2, False, fullcompile, pair, dfg)
            # once we get the bounds, make a node for each possible value
            for v in range(lb, ub+1):
                # if we need multiple of 2, skip it if it's not
                if log2 and not ((v & (v-1) == 0) and v != 0):
                    continue
                tree.create_node([to_find[0],v,stgs_used[v]], parent=root)
            symbolics_opt[to_find[0]] = lb
            tree.show()
            # keep going if we have more variables (to_find[1:] not empty)
            if not to_find[1:]: # we're done! we've found all vars for this path
                return
            else:   # we still have more variables to find bounds for, so keep going
                build_bounds_tree(tree, root, to_find[1:], symbolics_opt, opt_info, fullcompile, pair, efficient, dfg)


    else:
        if not children:
            # find the bounds for the next variable in the list, given the values for the previous
            # first check if this needs to be a power of 2
            # if yes, then make lower bound = 2 instead of 1
            log2 = False
            startbound = 4
            lb = 1
            if to_find[0] in opt_info["symbolicvals"]["bounds"]:
                startbound = opt_info["symbolicvals"]["bounds"][to_find[0]][0]
                lb = opt_info["symbolicvals"]["bounds"][to_find[0]][0]
            if to_find[0] in opt_info["symbolicvals"]["logs"].values():
                log2 = True
                if lb < 2:
                    lb = 2
                if startbound < 4:  # if user gives us higher bound, don't overwrite it
                    startbound = 4

            # if it's a memory variable, we're starting from the max memory avail for a single register array and then moving down
            # we have a hard upper bound for memory, so it's faster to start from there (we don't have ub for other vars)
            if to_find[0] in opt_info["symbolicvals"]["symbolics"]:
                #if log2: symbolics_opt[to_find[0]] = single_stg_mem_log2
                #else: symbolics_opt[to_find[0]] = single_stg_mem
                if pair:
                    symbolics_opt[to_find[0]] = single_stg_mem_log2_pairarray
                else:
                    symbolics_opt[to_find[0]] = single_stg_mem_log2
                lb = 32
                ub = get_max_val_efficient(symbolics_opt,to_find[0], opt_info, log2, True, fullcompile, pair, dfg)
                #tree.create_node((to_find[0],ub), parent=root)
            else:
                # keep compiling until we hit max stgs, get ub 
                #print("FIND BOUNDS")
                symbolics_opt[to_find[0]] = startbound
                ub = get_max_val_efficient(symbolics_opt, to_find[0], opt_info, log2, False, fullcompile, pair, dfg)
            # once we get the bounds, make a node for each possible value
            for v in range(lb, ub+1):
                # if we need multiple of 2, skip it if it's not
                if log2 and not ((v & (v-1) == 0) and v != 0):
                    continue
                tree.create_node([to_find[0],v], parent=root)
            symbolics_opt[to_find[0]] = lb
            tree.show()
            # keep going if we have more variables (to_find[1:] not empty)
            if not to_find[1:]: # we're done! we've found all vars for this path
                return
            else:   # we still have more variables to find bounds for, so keep going
                build_bounds_tree(tree, root, to_find[1:], symbolics_opt, opt_info, fullcompile, pair, efficient, dfg)


def set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree, opt_info):
    for sol in sol_choice:
        node = tree.get_node(sol)
        if node.tag=="root":
            continue
        symbolics_opt[node.tag[0]] = node.tag[1]
    # set rule vars
    set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt



def prune_fullcompile(solutions, opt_info, bounds_tree):
    sols_by_mem = {}
    sols_by_stgs = {}
    sols_by_hash = {}
    sols_by_regaccess = {}
    mem_formula = opt_info["optparams"]["mem_formula"]
    for sol in solutions:
        mem_formula = opt_info["optparams"]["mem_formula"]
        hash_formula = opt_info["optparams"]["hash_formula"]
        regaccess_formula = opt_info["optparams"]["regaccess_formula"]
        for n_identifier in sol:
            # TODO: better way to do this?
            node = bounds_tree.get_node(n_identifier)
            if node.tag == "root":
                continue
            # replace var name with val of vars in formulas in json file
            var_name = node.tag[0]
            var_value = node.tag[1]
            # replace string with val of var
            mem_formula = mem_formula.replace(var_name, str(var_value))
            hash_formula = hash_formula.replace(var_name, str(var_value))
            regaccess_formula = regaccess_formula.replace(var_name, str(var_value))
        # compute memory using formula string, add to data structure
        mem_usage = eval(mem_formula)
        if mem_usage in sols_by_mem:
            sols_by_mem[mem_usage] += [sol]
        else:
            sols_by_mem[mem_usage] = [sol]
        # compute stg usage (num stgs at leaf)
        stg_usage = bounds_tree.get_node(sol[-1]).tag[2]
        if stg_usage in sols_by_stgs:
            sols_by_stgs[stg_usage] += [sol]
        else:
            sols_by_stgs[stg_usage] = [sol]
        # compute (total) hash units used
        hash_usage = eval(hash_formula)
        if hash_usage in sols_by_hash:
            sols_by_hash[hash_usage] += [sol]
        else:
            sols_by_hash[hash_usage] = [sol]
        # compute (total) reg accesses
        regaccess_usage = eval(regaccess_formula)
        if regaccess_usage in sols_by_regaccess:
            sols_by_regaccess[regaccess_usage] += [sol]
        else:
            sols_by_regaccess[regaccess_usage] = [sol]

    return sols_by_mem, sols_by_stgs, sols_by_hash, sols_by_regaccess

def prune_layout(solutions, bounds_tree):
    sols_by_mem = {}
    sols_by_stgs = {}
    sols_by_hash = {}
    sols_by_regaccess = {}
    for sol in solutions:
        # resource usage for a solution = usage at leaf node
        # mem usage (sram blocks)
        mem_usage = bounds_tree.get_node(sol[-1]).tag[2]["sram"]
        if mem_usage in sols_by_mem:
            sols_by_mem[mem_usage] += [sol]
        else:
            sols_by_mem[mem_usage] = [sol]
        # stg usage (num stgs at leaf)
        stg_usage = bounds_tree.get_node(sol[-1]).tag[2]["stages"]
        if stg_usage in sols_by_stgs:
            sols_by_stgs[stg_usage] += [sol]
        else:
            sols_by_stgs[stg_usage] = [sol]
        # hash usage
        hash_usage = bounds_tree.get_node(sol[-1]).tag[2]["hash"]
        if hash_usage in sols_by_hash:
            sols_by_hash[hash_usage] += [sol]
        else:
            sols_by_hash[hash_usage] = [sol]
        # reg accesses
        regaccess_usage = bounds_tree.get_node(sol[-1]).tag[2]["regaccess"]
        if regaccess_usage in sols_by_regaccess:
            sols_by_regaccess[regaccess_usage] += [sol]
        else:
            sols_by_regaccess[regaccess_usage] = [sol]

    return sols_by_mem, sols_by_stgs, sols_by_hash, sols_by_regaccess

# testing out ordered parameter search
def ordered(symbolics_opt, opt_info, o, timetest, nopruning, fullcompile, exhaustive, pair, preprocessingonly, shortcut, dfg, efficient):
    opt_start_time = time.time()

    # if we're shortcutting, we've already done the preprocessing, so load from preprocessed.pkl 
    if shortcut:
        sols = pickle.load(open('preprocessed.pkl','rb'))
        bounds_tree = sols["tree"]
        solutions = sols["all_sols"]
        if not efficient:
            best_mem_sols = sols["mem_sols"]
            best_stgs_sols = sols["stgs_sols"]
            best_hash_sols = sols["hash_sols"]
            best_regaccess_sols = sols["regaccess_sols"]

    else:
        # STEP 1: reduce parameter space by removing solutions that don't compile
        # get bounds for all variables
        # aka explicitly define (resource) parameter space, excluding solutions that won't compile and that don't use all resources
        bounds_tree = Tree()
        bounds_tree.create_node("root","root")
        # set everything to 1 to start with
        symbolics_opt = dict.fromkeys(symbolics_opt, 1)
        # if we have lower bounds for any of them, replace 1 with those bounds
        for var in symbolics_opt:
            if var in opt_info["symbolicvals"]["bounds"]:
                symbolics_opt[var] = opt_info["symbolicvals"]["bounds"][var][0]
            # if it's a mem var, set to 32
            if var in opt_info["symbolicvals"]["symbolics"]:
                symbolics_opt[var] = 32
        # TODO: deal with this better
        # if caching, just set to true (cms) and in theory do precision in parallel
        if opt_info["lucidfile"] == "caching.dpt":
            print("CACHING")
            symbolics_opt["eviction"] = True
            #symbolics_opt["rows"] = 1
            #symbolics_opt["cols"] = 2
        build_bounds_tree(bounds_tree,"root", opt_info["optparams"]["order_resource"], symbolics_opt, opt_info, fullcompile, pair, efficient, dfg)

        print("UPPER BOUND TIME:", time.time()-opt_start_time)    

        ub_time = time.time()-opt_start_time

        # STEP 2: prune solutions found in step 1 by throwing out solutions that use less resources (memory, stgs) than others
        # iterate through each path in tree
        # need a formula for calculating total memory = x * y + j * k
        # once we calc total resources, remove solutions that are < max
        solutions = bounds_tree.paths_to_leaves()
        if not efficient:
            if fullcompile:
                sols_by_mem, sols_by_stgs, sols_by_hash, sols_by_regaccess = prune_fullcompile(solutions, opt_info, bounds_tree)
            else:
                sols_by_mem, sols_by_stgs, sols_by_hash, sols_by_regaccess = prune_layout(solutions, bounds_tree)

            print("UB+PRUNE TIME:", time.time()-opt_start_time)

            print("TOTAL SOLS:", len(solutions))
            #print(sols_by_mem.keys())
            #print("MAX MEM", max(list(sols_by_mem.keys())))
            best_mem_sols = sols_by_mem[max(list(sols_by_mem.keys()))]
            best_stgs_sols = sols_by_stgs[max(list(sols_by_stgs.keys()))]
            best_hash_sols = sols_by_hash[max(list(sols_by_hash.keys()))]
            best_regaccess_sols = sols_by_regaccess[max(list(sols_by_regaccess.keys()))]
            print("MAX MEM SOLS", len(best_mem_sols))
            print("MAX STGS SOLS", len(best_stgs_sols))
            print("MAX HASH SOLS", len(best_hash_sols))
            print("MAX REGACCESS SOLS", len(best_regaccess_sols))
            #best_mem_stgs = [sol for sol in best_mem_sols if sol in best_stgs_sols]
            #print("OVERLAP SOLS (MEM+STGS)", len(best_mem_stgs))

    if preprocessingonly:
        # dump all solutions to preprocessed.pkl
        sols = {}
        sols["all_sols"] = solutions
        if not efficient:
            sols["mem_sols"] = best_mem_sols
            sols["stgs_sols"] = best_stgs_sols
            sols["hash_sols"] = best_hash_sols
            sols["regaccess_sols"] = best_regaccess_sols
        sols["tree"] = bounds_tree
        sols["time(s)"] = time.time()-opt_start_time
        if dfg:
            with open('preprocessed_dfg.pkl','wb') as f:
                pickle.dump(sols, f)
            exit()
        if efficient:
            with open('preprocessed_efficient.pkl','wb') as f:
                pickle.dump(sols, f)
            exit()
        with open('preprocessed.pkl','wb') as f:
            pickle.dump(sols, f)
        exit()


    if not efficient:
        # if the user wants to prune, then let's discard extra sols
        pruned_sols = []
        if not nopruning:
            for prune_res in opt_info["optparams"]["prune_res"]:
                if prune_res=="memory":
                    pruned_sols.extend(best_mem_sols)
                elif prune_res=="stages":
                    pruned_sols.extend(best_stgs_sols)
                elif prune_res=="hash":
                    pruned_sols.extend(best_hash_sols)
                elif prune_res=="regaccess":
                    pruned_sols.extend(best_regaccess_sols)
                else:
                    exit("invalid pruning resource; must be memory, stages, hash, and/or regaccess")
            # remove any repeated sols in list
            pruned_sols = list(set(pruned_sols))


    #exit()

    # run interpreter on ones w/ most memory first, then maybe next highest???
    # STEP 3: run interpreter to get cost, optimize for non-resource parameters
    # search through that parameter space, using interpreter to get cost
    # pick resource params: pick one child from each level of tree (checking for logs as we go)
    # pick non-resource param: some value w/in user-defined bounds
    # set any rule-based variables
    # run the interpreter!
    #print("SYMBOLICS BEFORE:", symbolics_opt)

    if exhaustive:  # evaluate all compiling solutions w/ interpreter
        iterations = 1
        tested_sols = []
        testing_sols = []
        testing_eval = []
        if "struct" in opt_info["optparams"]:
            if opt_info["optparams"]["struct"] == "cms":
                symbolics_opt["eviction"] = True
            elif opt_info["optparams"]["struct"] == "precision":
                symbolics_opt["eviction"] = False
                symbolics_opt["rows"] = 1
                symbolics_opt["cols"] = 128
                #symbolics_opt["expire_thresh"] = 2
                symbolics_opt["THRESH"] = 2
        # get all possible non_resource vals
        nr_vals = []
        for nonresource in opt_info["optparams"]["non_resource"]:
            nr_range = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource]))
            if opt_info["symbolicvals"]["bounds"][nonresource][1] not in nr_range:
                nr_range.append(opt_info["symbolicvals"]["bounds"][nonresource][1])
            nr_vals.append(nr_range)
        non_resource_sols = list(itertools.product(*nr_vals))

        # start interpreter time
        interpreter_start_time = time.time()
        for sol_choice in solutions:
            curr_iter = "evaling sol " + str(iterations) + " out of " + str(len(solutions)) 
            print(curr_iter)
            with open("progress.txt","w") as f:
                f.write(curr_iter)
            for sol in sol_choice:
                node = bounds_tree.get_node(sol)
                if node.tag=="root":
                    continue
                symbolics_opt[node.tag[0]] = node.tag[1]
            # NOTE: this assumes that order of values generated as non_resource_sols is same as order on non_resource list in json
            for nr_choice in non_resource_sols:
                for nr_var in opt_info["optparams"]["non_resource"]:
                    symbolics_opt[nr_var] = nr_choice[opt_info["optparams"]["non_resource"].index(nr_var)]
                # once we choose all symbolics, set any rule-based symbolics
                if "rules" in opt_info["symbolicvals"]:
                    symbolics_opt = set_rule_vars(opt_info, symbolics_opt)
                print("SYMBOLICS TO EVAL", symbolics_opt)
                cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
                tested_sols.append(copy.deepcopy(symbolics_opt))

                testing_sols.append(copy.deepcopy(symbolics_opt))
                testing_eval.append(cost)
                iterations += 1
        best_sols = [{}]

        with open('final_testing_sols_ordered_exhaustive.pkl','wb') as f:
            pickle.dump(testing_sols,f)
        with open('final_testing_eval_ordered_exhaustive.pkl','wb') as f:
            pickle.dump(testing_eval,f)

    else:   # we're using one of our search strategies
        strat = opt_info["optparams"]["strategy"]

        if strat=="simannealing":
            if nopruning:
                return simulated_annealing(symbolics_opt, opt_info, o, timetest, bounds_tree, solutions)
            else:
                return simulated_annealing(symbolics_opt, opt_info, o, timetest, bounds_tree, pruned_sols)

        elif strat=="random":
            if nopruning:
                return random_opt(symbolics_opt, opt_info, o, timetest, bounds_tree, solutions)
            else:
                return random_opt(symbolics_opt, opt_info, o, timetest, bounds_tree, pruned_sols)

        # TODO: incorporate other params in opt_info (alpha, sigma, etc.)
        elif strat=="neldermead":
            if nopruning:
                return nelder_mead(symbolics_opt, opt_info, o, timetest, solutions=solutions, tree=bounds_tree)
            else:
                return nelder_mead(symbolics_opt, opt_info, o, timetest, solutions=pruned_sols, tree=bounds_tree)

        elif strat=="bayesian":
            if nopruning:
                return bayesian(symbolics_opt, opt_info, o, timetest, solutions, bounds_tree)
            else:
                return bayesian(symbolics_opt, opt_info, o, timetest, pruned_sols, bounds_tree)

    interp_time = time.time()-interpreter_start_time
    print("TOTAL INTERP TIME:", interp_time)
    #print("TOTAL UB TIME:", ub_time)

    return best_sols[0], best_cost


# choosing more opt strategies
# some considerations:
#   we're doing simulations, so we don't know if the function is differentiable or not
#   we could try some strategies for differentiable obj functions, but putting that on the backburner to focus on strategies for simulation-based functions


# nelder-mead uses numpy arrays, so this function converts that to symbolics opt, so we can run interpreter
# NOTE: "candidate_index" is reserved, can't use for variable name
def set_symbolics_from_nparray(nparray, index_dict, symbolics_opt, 
                               opt_info, solutions=[], tree=None):
    '''
    # OLD, treat nonresource differently
    for val_i in range(nparray.size):
        val = int(nparray[val_i])
        var_name = index_dict[val_i]
        if solutions and var_name == "candidate_index":    # this is an index in preprocessed sols list, not var value
            sol_choice = solutions[val]
            # write sol choice symbolics_opt
            symbolics_opt = set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree)
        else:   # this is a variable
            symbolics_opt[var_name] = int(val)
    '''
    sol_index = int(nparray[0])
    symbolics_opt = solutions[sol_index]

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        symbolics_opt = set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt

# nelder-mead simplex
# direct search algo (can get stuck in local optima, so may benefit from trying different starting points)
# start at random val (or user-defined starting point)
# use a shape structure (simplex) composed of n+1 points (n is number of input dimensions to function)

# using scipy:
#   bounds for preprocessed vars is index (min=0, max=len(sols)-1)
#   for non resource vars, will probably have to do some rounding
#   function is interpsim function that returns cost
#   input to function is var values?????

# based on github code found here: https://github.com/fchollet/nelder-mead/blob/master/nelder_mead.py
# NOTE: instead of passing in function as param, we call gen_cost, which runs interpreter and returns objective val
'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''
def nelder_mead(symbolics_opt, opt_info, o, timetest, 
                no_improve_thr=10e-6, no_improv_break=50,
                alpha=10., gamma=20., rho=-0.5, sigma=0.5,
                solutions=[], tree=None): 
    '''
        @param symbolics_opt (dict): dict of symbolics and their values (or index in preprocessed list), used to gen cost
        @param opt_info (dict): from input json file
        @param o (instance of opt): module that contains cost calc for interpreter
        @param timetest (bool): if true, output time test data
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        @param solutions (list): if we preprocessed, this is list of solutions (otherwise empty)
        @param tree (Tree): tree from preprocessing

        return: best symbolics_opt (param config), best cost
    '''

    # TODO: figure out how to set params (step, no_improv_thr, alpha, gamma, rho, sigma)?
    # TODO: don't do repeated solutions, stop if we've tested all sols

    testing_sols = []
    testing_eval = []
    # decide if we're stopping by time, iterations, or both (whichever reaches thresh first)
    iters = False
    simtime = False
    iter_time = False

    if "stop_iter" in opt_info["optparams"]:
        iters = True
    if "stop_time" in opt_info["optparams"]:
        simtime = True
    if iters and simtime:
        iter_time = True

    start_time = time.time()

    # NEW, only optimize for index val, treat resource and nonresource the same
    total_sols = 1
    all_solutions_symbolics = []
    if not solutions:
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                total_sols *= vals
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            total_sols *= vals
    else:   # we've preprocessed, used those solutions to calc total
        # TODO: create symbolics_opt from solutions and append to all_solutions_symbolics
        for sol_choice in solutions:
            all_solutions_symbolics.append(copy.deepcopy(set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree, opt_info)))
        for nonresource in opt_info["optparams"]["non_resource"]:
                #total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                total_sols *= (len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource])) + 1)
                new_sols = []
                for sol_choice in all_solutions_symbolics:
                    #vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                    vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource]))
                    vals.append(opt_info["symbolicvals"]["bounds"][nonresource][1])
                    for v in vals:
                        # update w/ new value
                        sol_choice[nonresource] = v
                        # append to new solution list
                        new_sols.append(copy.deepcopy(sol_choice))

                all_solutions_symbolics = new_sols

        total_sols *= len(solutions)



    # randomly generate starting solution
    # if solutions is an empty list, then we haven't done preprocessing
    if not solutions:
        symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], False, {}, opt_info)
    else:
        # OLD, treat nonresource differently
        # set resource vars
        #symbolics_opt, candidate_index = gen_next_random_preprocessed(tree, symbolics_opt, solutions, opt_info)
        # NEW, only optimize for index val
        symbolics_opt = choice(all_solutions_symbolics)
        candidate_index = all_solutions_symbolics.index(symbolics_opt)
        print("STARTING", candidate_index)
        print(symbolics_opt)

    starting = copy.deepcopy(symbolics_opt)
    num_sols_time = {}
    time_cost = {}

    # create starting numpy array and index_dict (index: var_name)
    # also create step and bounds arrays
    # step: look-around radius in initial step (1 val for each dimension)
    index_dict = {}
    if not solutions:   # TODO: this is a little silly, fix this?
        dimensions = 0
        for var in symbolics_opt:
            if var not in opt_info["symbolicvals"]["logs"] and ("rules" not in opt_info["symbolicvals"] or var not in opt_info["symbolicvals"]["rules"]):
                dimensions += 1 
        x_start = np.zeros(dimensions)
        step = np.zeros(dimensions)
        bounds = np.zeros(shape=(dimensions,2))
        index = 0
        for var in symbolics_opt:
            if var not in opt_info["symbolicvals"]["logs"] and ("rules" not in opt_info["symbolicvals"] or var not in opt_info["symbolicvals"]["rules"]):
                x_start[index] = symbolics_opt[var]
                index_dict[index] = var
                step[index] = opt_info["optparams"]["stepsize"][var]
                bounds[index][0] = opt_info["symbolicvals"]["bounds"][var][0]
                bounds[index][1] = opt_info["symbolicvals"]["bounds"][var][1] 
                index += 1
    else:   # dimensions = number of non resource vars + 1 (for solutions list)
        # OLD, treat nonresource differently
        #dimensions = 1 + len(opt_info["optparams"]["non_resource"])
        # NEW, only optimize for index val
        dimensions = 1
        x_start = np.zeros(dimensions)
        step = np.zeros(dimensions)
        bounds = np.zeros(shape=(dimensions,2))
        index = 0
        # OLD, treat nonresource differently
        '''
        for x_val in opt_info["optparams"]["non_resource"]:
            x_start[index] = symbolics_opt[x_val]
            index_dict[index] = x_val
            step[index] = opt_info["optparams"]["stepsize"][x_val]
            bounds[index][0] = opt_info["symbolicvals"]["bounds"][x_val][0]
            bounds[index][1] = opt_info["symbolicvals"]["bounds"][x_val][1]
            index += 1
        '''
        x_start[index] = candidate_index
        index_dict[index] = "candidate_index"
        #step[index] = 1
        #step[index] = 10
        step[index] = 5
        bounds[index][0] = 0
        # OLD, treat nonresource differently
        #bounds[index][1] = len(solutions) - 1
        # NEW, only optimize for index val
        bounds[index][1] = len(all_solutions_symbolics) - 1
        print("X START", x_start)
        print("INDEX DICT", index_dict)
        print("STEP", step)
        print("BOUNDS", bounds)


    # init
    dim = len(x_start)
    # OLD, treat nonresource differently
    #symbolics_opt = set_symbolics_from_nparray(x_start, index_dict, symbolics_opt, opt_info, solutions, tree)
    # NEW, only optimize for index val
    symbolics_opt = set_symbolics_from_nparray(x_start, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
    print("symbolics opt after set from np array", symbolics_opt)
    #prev_best = f(x_start)
    if not solutions:
        prev_best = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
    else:
        print("EVALED SOL", symbolics_opt)
        prev_best = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
    no_improv = 0
    res = [[x_start, prev_best]]
    testing_sols.append(copy.deepcopy(symbolics_opt))
    testing_eval.append(prev_best)

    for i in range(dim):
        x = copy.copy(x_start)
        # we can't have floating points, so rounding to int
        x[i] = int(x[i] + step[i])
        if x[i] < bounds[i][0]: # we're < lb for variable, set to lb
            x[i]=bounds[i][0]
        elif x[i] > bounds[i][1]:   # we're > ub for variable, set to ub
            x[i]=bounds[i][1]
        # round to closest power of 2 if we need to
        if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
            x[i] = closest_power(x[i])
        # OLD, treat nonresource differently
        #symbolics_opt = set_symbolics_from_nparray(x, index_dict, symbolics_opt, opt_info, solutions, tree)
        # NEW, only optimize for index val
        symbolics_opt = set_symbolics_from_nparray(x, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)

        #score = f(x)
        if not solutions:
            score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
        else:
            print("EVALED SOL", symbolics_opt)
            score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(score)
        res.append([x, score])


    print("RES AFTER FIRST 2", res)

    # simplex iter
    iterations = 0
    while True:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        print("CURRENT TIME", time.time()-start_time)

        print("RES", res)
        # check iter/time conditions
        if iters or iter_time:
            if iterations >= opt_info["optparams"]["stop_iter"]:
                #best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, solutions, tree)
                #return best_sol, res[0][1]
                break
        if simtime or iter_time:
            if (time.time()-start_time) >= opt_info["optparams"]["stop_time"]:
                #best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, solutions, tree)
                #return best_sol, res[0][1]
                break

        curr_time = time.time()
        # TIME TEST (save sols/costs we've evaled up to this point)
        if timetest:
            # 5 min (< 10)
            if 300 <= (curr_time - start_time) < 600:
                with open('5min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('5min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["5min"] = len(testing_sols)
                time_cost["5min"] = copy.deepcopy(testing_eval)
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["10min"] = len(testing_sols)
                time_cost["10min"] = copy.deepcopy(testing_eval)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["30min"] = len(testing_sols)
                time_cost["30min"] = copy.deepcopy(testing_eval)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["45min"] = len(testing_sols)
                time_cost["45min"] = copy.deepcopy(testing_eval)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["60min"] = len(testing_sols)
                time_cost["60min"] = copy.deepcopy(testing_eval)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["90min"] = len(testing_sols)
                time_cost["90min"] = copy.deepcopy(testing_eval)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                num_sols_time["120min"] = len(testing_sols)
                time_cost["120min"] = copy.deepcopy(testing_eval)
                break



        iterations += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            print("IMPROVEMENT!")
            no_improv = 0
            prev_best = best
        else:
            print("NO IMPROVEMENT")
            no_improv += 1

        if no_improv >= no_improv_break:
            #best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, solutions, tree)
            #return best_sol, res[0][1]
            break

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            print("TUP", tup)
            for i, c in enumerate(tup[0]):
                print("C",c)
                x0[i] += c / (len(res)-1)
                x0[i] = int(x0[i])
                if x0[i] < bounds[i][0]: # we're < lb for variable, set to lb
                    x0[i]=bounds[i][0]
                elif x0[i] > bounds[i][1]:   # we're > ub for variable, set to ub
                    x0[i]=bounds[i][1]
                # round to closest power of 2 if we need to
                if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
                    x0[i] = closest_power(x0[i])

        print("x0", x0)

        early_exit = False
        # reflection
        print("XO BEFORE REFLECTION", x0)
        xr = x0 + alpha*(x0 - res[-1][0])
        while int(xr[0]) == int(res[-1][0]) or int(xr[0]) == int(res[0][0]):
            xr[0] = int(xr[0])+1
        print("XR AFTER REFLECTION", xr)
        for i in range(dim):    # check bounds, round
            # can't have floats, round
            xr[i] = int(xr[i])
            if xr[i] < bounds[i][0]: # we're < lb for variable, set to lb
                xr[i]=bounds[i][0]
            elif xr[i] > bounds[i][1]:   # we're > ub for variable, set to ub
                xr[i]=bounds[i][1]
                while int(xr[0]) == int(res[-1][0]) or int(xr[0]) == int(res[0][0]):
                    xr[0] = int(xr[0])-1
                if xr[0] < 0:
                    early_exit = True
                    print("res", res)
                    print("xr", xr)
                    print("early exit, reflection")
            # round to closest power of 2 if we need to
            if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
                xr[i] = closest_power(xr[i])

        if early_exit:
            break
        # OLD, treat nonresource differently
        #symbolics_opt = set_symbolics_from_nparray(xr, index_dict, symbolics_opt, opt_info, solutions, tree)
        # NEW, only optimize for index val
        symbolics_opt = set_symbolics_from_nparray(xr, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
        print(xr[0])
        print(symbolics_opt)
        print(all_solutions_symbolics[int(xr[0])])
        #exit()
        #rscore = f(xr)
        if not solutions:
            rscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
        else:
            print("EVALED SOL AFTER REFLECTION", symbolics_opt)
            rscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(rscore)
        print("RES RELFECTION", res)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            if res[0][0] == res[1][0]:
                print("SAME, reflection")
                break
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            while int(xe[0]) == int(res[-1][0]) or int(xe[0]) == int(res[0][0]):
                xe[0] = int(xe[0])+1
            for i in range(dim):    # check bounds, round
                # can't have floats, round
                xe[i] = int(xe[i])
                if xe[i] < bounds[i][0]: # we're < lb for variable, set to lb
                    xe[i]=bounds[i][0]
                elif xe[i] > bounds[i][1]:   # we're > ub for variable, set to ub
                    xe[i]=bounds[i][1]
                    while int(xe[0]) == int(res[-1][0]) or int(xe[0]) == int(res[0][0]):
                        xe[0] = int(xe[0])-1
                    if xe[0] < 0:
                        early_exit = True
                        print("res", res)
                        print("xe", xe)
                        print("early exit, expansion")
                # round to closest power of 2 if we need to
                if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
                    xe[i] = closest_power(xe[i])

            if early_exit:
                break
            # OLD, treat nonresource differently
            #symbolics_opt = set_symbolics_from_nparray(xe, index_dict, symbolics_opt, opt_info, solutions, tree)
            # NEW, only optimize for index val
            print("XE",xe)
            print(len(all_solutions_symbolics))
            symbolics_opt = set_symbolics_from_nparray(xe, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
            print(xe[0])
            print(symbolics_opt)
            print(all_solutions_symbolics[int(xe[0])])
            #exit()
            #escore = f(xe)
            if not solutions:
                escore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL AFTER EXPANSION", symbolics_opt)
                escore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
            testing_sols.append(copy.deepcopy(symbolics_opt))
            testing_eval.append(escore)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                if res[0][0] == res[1][0]:
                    print("SAME, expansion1")
                    break
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                if res[0][0] == res[1][0]:
                    print("SAME, expansion2")
                    break
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        print("ORIG XC", xc)
        print("ORIG RES", res)
        while int(xc[0]) == int(res[-1][0]) or int(xc[0]) == int(res[0][0]):
            print("XC SAME", xc[0])
            xc[0] = int(xc[0])-1
        for i in range(dim):    # check bounds, round
            # can't have floats, round
            xc[i] = int(xc[i])
            if xc[i] < bounds[i][0]: # we're < lb for variable, set to lb
                xc[i]=bounds[i][0]
                while int(xc[0]) == int(res[-1][0]) or int(xc[0]) == int(res[0][0]):
                    xc[0] = int(xc[0])+1
                    if xc[0] > len(all_solutions_symbolics)-1:
                        early_exit = True
                        print("res", res)
                        print("xc", xr)
                        print("early exit, contraction")
            elif xc[i] > bounds[i][1]:   # we're > ub for variable, set to ub
                xc[i]=bounds[i][1]
            # round to closest power of 2 if we need to
            if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
                xc[i] = closest_power(xc[i])
        if early_exit:
            break
        # OLD, treat nonresource differently
        #symbolics_opt = set_symbolics_from_nparray(xc, index_dict, symbolics_opt, opt_info, solutions, tree)
        # NEW, only optimize for index val
        symbolics_opt = set_symbolics_from_nparray(xc, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
        print("XC", xc[0])
        print("RES", res)
        print(symbolics_opt)
        print(all_solutions_symbolics[int(xc[0])])
        #exit()
        #cscore = f(xc)
        if not solutions:
            cscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
        else:
            print("EVALED SOL AFTER CONTRACTION", symbolics_opt)
            cscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(cscore)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            if res[0][0] == res[1][0]:
                print(res)
                print("SAME, contraction")
                break
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        print("orig res reduction", res)
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            for i in range(dim):    # check bounds, round
                # can't have floats, round
                redx[i] = int(redx[i])
                while int(redx[i]) == int(res[-1][0]) or int(redx[i]) == int(res[0][0]) or (len(nres)>0 and int(redx[i]) == int(nres[0][0])):
                    redx[i] = int(redx[i])-1
                if redx[i] < bounds[i][0]: # we're < lb for variable, set to lb
                    redx[i]=bounds[i][0]
                    while int(redx[i]) == int(res[-1][0]) or int(redx[i]) == int(res[0][0]) or (len(nres)>0 and int(redx[i]) == int(nres[0][0])):
                        redx[i] = int(redx[i])+1
                        if redx[0] > len(all_solutions_symbolics)-1:
                            early_exit = True
                            print("res", res)
                            print("redx", redx)
                            print("early exit, reduction")
                elif redx[i] > bounds[i][1]:   # we're > ub for variable, set to ub
                    redx[i]=bounds[i][1]
                # round to closest power of 2 if we need to
                if index_dict[i] in opt_info["symbolicvals"]["logs"].values():
                    redx[i] = closest_power(redx[i])
            # OLD, treat nonresource differently
            #symbolics_opt = set_symbolics_from_nparray(redx, index_dict, symbolics_opt, opt_info, solutions, tree)
            # NEW, only optimize for index val
            if early_exit:
                break
            symbolics_opt = set_symbolics_from_nparray(redx, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
            print(redx[0])
            print(symbolics_opt)
            print(all_solutions_symbolics[int(redx[0])])
            #exit()
            #score = f(redx)
            if not solutions:
                score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL AFTER REDUCTION", symbolics_opt)
                score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
            testing_sols.append(copy.deepcopy(symbolics_opt))
            testing_eval.append(score)
            nres.append([redx, score])
            print("new redx reduction, redx")
        res = nres
        if res[0][0] == res[1][0]:
            print("SAME, reduction")
            break


    # OLD, treat nonresource differently
    #best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, solutions, tree)
    # NEW, only optimize for index val
    best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
    with open('final_testing_sols_nm.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('final_testing_eval_nm.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    num_sols_time["final"] = len(testing_sols)
    time_cost["final"] = copy.deepcopy(testing_eval)

    return best_sol, res[0][1], time_cost, num_sols_time, starting

    
# bayesian optimization
# step 1: create surrogate function, using regression predictive modeling
#   try gaussian process, but maybe there's something better?
#   use GaussianProcessRegressor scikit, input are sample x vals and their cost/score
#   can try different kernel functions w/ regressor, default is radial basis function
#   can update model by calling fit(x, y)
#   use model by calling predict(X)
#   surrogate used to test range of candidate samples in domain
# step 2: search strategy and acquisition function
#   search strat used to navigate doman in response to surrogate func
#   acquisition used to interpret and score result of surrogate
#   search can be simple random, BFGS is popular though
#   choose candidate sol, eval with acquisition, then max acquisition
#   acq func decides whether sol is worth evaling w real obj func
#   many types of acq funcs (Probability of Improvement is simplest)
def bayesian(symbolics_opt, opt_info, o, timetest, solutions, bounds_tree):
    iters = False
    simtime = False
    iter_time = False
    if "stop_iter" in opt_info["optparams"]:
        iters = True
    if "stop_time" in opt_info["optparams"]:
        simtime = True
    if iters and simtime:
        iter_time = True

    # step 0: sample domain and get cost (to build surrogate model)
    # compute the total number of solutions we have
    # get total number of solutions, so we know if we've gone through them all
    # TODO: get this to work with non preprocessed
    # TODO: simplify this, only need to count length of all_solutions_symbolics to get total_sols
    total_sols = 1
    all_solutions_symbolics = []
    if not solutions:
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                total_sols *= vals
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            total_sols *= vals
    else:   # we've preprocessed, used those solutions to calc total
        # TODO: create symbolics_opt from solutions and append to all_solutions_symbolics
        for sol_choice in solutions:
            all_solutions_symbolics.append(copy.deepcopy(set_symbolics_from_tree_solution(sol_choice, symbolics_opt, bounds_tree, opt_info)))
        for nonresource in opt_info["optparams"]["non_resource"]:
                #total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                total_sols *= (len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource])) + 1)
                new_sols = []
                for sol_choice in all_solutions_symbolics:
                    #vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
                    vals = list(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1], opt_info["optparams"]["stepsize"][nonresource]))
                    vals.append(opt_info["symbolicvals"]["bounds"][nonresource][1])
                    for v in vals:
                        # update w/ new value
                        sol_choice[nonresource] = v
                        # append to new solution list
                        new_sols.append(copy.deepcopy(sol_choice))
                all_solutions_symbolics = new_sols


        if "rules" in opt_info["symbolicvals"]:
            for sol in all_solutions_symbolics:
                sol = set_rule_vars(opt_info, sol)
        #print(all_solutions_symbolics)
        #exit()
        total_sols *= len(solutions)

    #print(all_solutions_symbolics)
    #exit()

    print("TOTAL_SOLS", total_sols)

    # sample x% of solutions
    # TODO: do this for non preprocessed
    if "samplesize" in opt_info["optparams"]:
        sample_size=int(opt_info["optparams"]["samplesize"]*total_sols)
    else:
        #sample_size = int(0.05*total_sols)
        sample_size = int(0.01*total_sols)
        #sample_size = int(0.1*total_sols)
    print("SAMPLE SIZE", sample_size)
    sampled_sols = []
    sample_xvals = []


    '''
    # randomly sample % of solutions
    test_index = None
    test_cost = None
    while len(sampled_sols) < sample_size:
        if not solutions:
            symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], False, {}, opt_info)
            xvals = []
        else:
            symbolics_opt,sample_index = gen_next_random_preprocessed(bounds_tree, symbolics_opt, solutions, opt_info)
            if not test_index:
                test_index = [copy.deepcopy(symbolics_opt),sample_index]
            xvals = [sample_index]
            #for resource in opt_info["optparams"]["order_resource"]:
            #    xvals.append(symbolics_opt[resource])
            for nonresource in opt_info["optparams"]["non_resource"]:
                xvals.append(symbolics_opt[nonresource])       
 
        # don't allow repeat samples
        if symbolics_opt in sampled_sols:
            #print("REPEAT")
            continue
        sampled_sols.append(copy.deepcopy(symbolics_opt))
        #print("ADDED", symbolics_opt)
        sample_xvals.append(xvals)
    '''
    # grid(ish) sampling
    # instead of random, sample every total_samples/sample_size samples
    if solutions:
        sample_xvals = [[x] for x in range(0, len(all_solutions_symbolics), int(total_sols/sample_size))]
        for xval in sample_xvals:
            sampled_sols.append(all_solutions_symbolics[xval[0]])


    # save initial sampled sols
    with open('init_samplesols.pkl','wb') as f:
        pickle.dump(sampled_sols,f)

    # convert to numpy array, to use w/ scikit
    np_xvals = np.array(sample_xvals)
    #np.append(np_xvals, np.array([39])) 
    #print(np_xvals)
    #print(np_xvals.shape)
    #exit()


    # we're including sample time in overall time
    start_time = time.time()

    # step 1: create surrogate function
    # step 1.1: get cost for each sampled value
    sample_costs = []
    for sample in sampled_sols:
        if not solutions:
            score = gen_cost(sample, sample, opt_info, o, False, "bayesian")
        else:
            score = gen_cost(sample, sample, opt_info, o, False, "ordered")
        sample_costs.append(score)
        '''
        if not test_cost:
            test_cost = score
        '''

    np_yvals = np.array(sample_costs)

    # step 1.2: define the model (gaussian process regression)
    # TODO: try something other than gp regressor model????
    #kernel = kernels.ConstantKernel(2.0, (1e-1, 1e3)) * kernels.RBF(2.0, (1e-3, 1e3))
    kernel_rbf = 1.0 * kernels.RBF(length_scale=10.0, length_scale_bounds=(5, 1e2))
    kernel_exp = kernels.ExpSineSquared(length_scale=0.5, periodicity=10)
    kernel = kernel_rbf+kernel_exp
    model = GaussianProcessRegressor(kernel=kernel)
    #model = KernelRidge(kernel='poly')
    # step 1.3: fit the model (given sampled values)  
    model.fit(np_xvals, np_yvals)
 

    # step 2: optimize acquisition function
    # step 2.1: choose points in domain (via some search strategy)
    # TODO: do for non preprocessed
    # use index in sol list, not var value
    # pick min value, eval with gen_cost, refit model w/ actual, then repeat
    # keep track of current min
    '''
    acq_xsamples = []
    acq_sample_size = int(0.5*total_sols) 
    while len(acq_xsamples) < acq_sample_size:
        if not solutions:
            symbolics_opt = gen_next_random_nopreprocessing(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], False, {}, opt_info)
            xvals = []
        else:
            symbolics_opt,sample_index = gen_next_random_preprocessed(bounds_tree, symbolics_opt, solutions, opt_info)
            xvals = [sample_index]
            for nonresource in opt_info["optparams"]["non_resource"]:
                xvals.append(symbolics_opt[nonresource])

        # don't allow repeat samples
        if xvals in acq_xsamples:
            #print("REPEAT")
            continue
        #print("ADDED", symbolics_opt)
        acq_xsamples.append(xvals)

    np_acq_xvals = np.array(acq_xsamples)
    '''

    # use var values directly, not index in sol list
    acq_xsamples = []
    acq_samples_symbolics = []
    
    '''
    if solutions:
        for sol in solutions:
            symbolics_opt = set_symbolics_from_tree_solution(sol, symbolics_opt, bounds_tree) 
            xvals = [solutions.index(sol)]
            #for resource in opt_info["optparams"]["order_resource"]:
            #    xvals.append(symbolics_opt[resource])
            #for nonresource in opt_info["optparams"]["non_resource"]:
            #    xvals.append(symbolics_opt[nonresource])

            acq_xsamples.append(xvals)
            #acq_samples_symbolics.append(copy.deepcopy(symbolics_opt))
    '''
    #print("ACQ SAMPLES", acq_xsamples)

    acq_xsamples = [[x] for x in range(len(all_solutions_symbolics))]
    acq_samples_symbolics = all_solutions_symbolics

    np_acq_xvals = np.array(acq_xsamples)
    #mu, std = model.predict(np_acq_xvals, return_std=True)
    #mu = model.predict(np_acq_xvals)
    #with open('mutest.pkl','wb') as f:
    #    pickle.dump(mu, f)


    #exit()


     
    mu_vals = []
    best_sols = []
    best_mu = []
    actual_eval = []

    num_sols_time = {}
    time_cost = {}

    iterations = 0
    while True:
        # check iter/time conditions
        if iters or iter_time:
            if iterations >= opt_info["optparams"]["stop_iter"]:
                break
        if simtime or iter_time:
            if (time.time()-start_time) >= opt_info["optparams"]["stop_time"]:
                break

        curr_time = time.time()
        # TIME TEST (save sols/costs we've evaled up to this point)
        if timetest:
            # 5 min (< 10)
            if 300 <= (curr_time - start_time) < 600:
                with open('5min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('5min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["5min"] = len(best_sols)
                time_cost["5min"] = copy.deepcopy(actual_eval)
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('10min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["10min"] = len(best_sols)
                time_cost["10min"] = copy.deepcopy(actual_eval)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('30min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["30min"] = len(best_sols)
                time_cost["30min"] = copy.deepcopy(actual_eval)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('45min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["45min"] = len(best_sols)
                time_cost["45min"] = copy.deepcopy(actual_eval)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('60min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["60min"] = len(best_sols)
                time_cost["60min"] = copy.deepcopy(actual_eval)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('90min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["90min"] = len(best_sols)
                time_cost["90min"] = copy.deepcopy(actual_eval)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('120min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
                num_sols_time["120min"] = len(best_sols)
                time_cost["120min"] = copy.deepcopy(actual_eval)
                break


        # step 2.2: eval chosen points w/ surrogate
        #yhat, _ = surrogate(model, np_xvals)
        yhat = None
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            yhat, _ = model.predict(np_xvals, return_std=True)
        best = min(yhat)
        print("BEST", best)
    
        # calculate mean and stdev via surrogate function
        #mu, std = surrogate(model, np_acq_xvals)
        mu = None
        std = None
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            mu, std = model.predict(np_acq_xvals, return_std=True)
        #mu = mu[:, 0]
        # calculate the probability of improvement
        # TODO: try different probablistic acquisition functions? (or just surrogate directly)
        probs = norm.cdf((mu - best) / (std+1E-9))
        ix_prob = np.argmin(probs)
        ix = np.argmin(mu)
        mu_sorted = np.sort(mu)

        print("MU MIN VALUE")
        print("MU VALUE", mu[ix])
        best_mu.append(mu[ix])

        # save mu and best values so we can analyze later
        mu_vals.append(list(mu))
    

        # step 2.3: return best point
        print(np_acq_xvals[ix])
        #sol_choice = solutions[np_acq_xvals[ix, 0]]
        sol_choice = all_solutions_symbolics[np_acq_xvals[ix,0]]

        early_exit = False
        # DON'T eval a sol we've already tried:
        min_index = 1
        while sol_choice in sampled_sols:
            if min_index >= len(mu):
                # we've somehow gone through all the solutions
                early_exit = True
                break
            #print("MU",mu)
            #print("MU_SORTED",mu_sorted)
            #print("min_index", min_index)
            #print("mu sorted at index", mu_sorted[min_index])
            ix_next = np.where(mu==mu_sorted[min_index])[0][0]
            sol_choice = all_solutions_symbolics[np_acq_xvals[ix_next,0]]
            min_index += 1
        # ARGMAX best values: 87,37
        # ARGMIN best value: 151 (9 rows, 65536 cols) 
        '''
        for sol in sol_choice:
            node = bounds_tree.get_node(sol)
            if node.tag=="root":
                continue
            print("var:", node.tag[0], "value:", node.tag[1])
        '''
        print("BEST", sol_choice)

        if early_exit:
            break

        # eval best choice, fit model w/ new value
        #symbolics_opt = set_symbolics_from_tree_solution(sol_choice, symbolics_opt, bounds_tree)
        best_sols.append(copy.deepcopy(sol_choice))
        if not solutions:
            score = gen_cost(sol_choice, sol_choice, opt_info, o, False, "bayesian")
        else:
            score = gen_cost(sol_choice, sol_choice, opt_info, o, False, "ordered")
        print("ACTUAL VALUE:", score)
        actual_eval.append(score)

        # udpate w/ new value (np_acq_xvals[ix], score)
        # check if we've already evaluated this one (if not, add)
        print(sol_choice)
        if sol_choice not in sampled_sols:
            # add the new solution
            sampled_sols.append(copy.deepcopy(sol_choice))
            sample_xvals.append([np_acq_xvals[ix,0]])
            np_xvals = np.array(sample_xvals)
            # add the new score
            sample_costs.append(score)
            np_yvals = np.array(sample_costs)
            # refit the model
            model.fit(np_xvals, np_yvals) 

        iterations += 1

    '''
    print("PROB MIN VALUE")
    print("PROB MU VALUE", mu[ix_prob])
    print(np_acq_xvals[ix_prob])
    sol_choice = solutions[np_acq_xvals[ix_prob, 0]]
    # ARGMAX best values: 87,37
    # ARGMIN best value: 151 (9 rows, 65536 cols) 
    for sol in sol_choice:
        node = bounds_tree.get_node(sol)
        if node.tag=="root":
            continue
        print("var:", node.tag[0], "value:", node.tag[1])
    '''

    '''
    print("ARGMIN")
    print(np_acq_xvals[ix])

    print("MU")
    ix = np.argmin(mu)
    print(np_acq_xvals[ix])
    print("MU VALUE", mu[ix])
    print(mu)

    print("ACQ VALS")
    print(np_acq_xvals)
    '''

    '''
    print("PREDICT SOL", np_acq_xvals[test_index[1]])
    print("PREDICT COST", mu[test_index[1]])
    print("TEST SOL", test_index)
    print("TEST COST", test_cost)
    '''

    with open('mu.pkl','wb') as f:
        pickle.dump(mu_vals, f)

    with open('final_best_eval.pkl','wb') as f:
        pickle.dump(best_mu, f)

    with open('final_actual_eval.pkl','wb') as f:
        pickle.dump(actual_eval,f)

    with open('final_best_sols.pkl','wb') as f:
        pickle.dump(best_sols,f)

    with open('final_testing_sols.pkl','wb') as f:
        pickle.dump(sampled_sols, f)

    with open('final_testing_eval.pkl','wb') as f:
        pickle.dump(sample_costs, f)

    best_eval = min(sample_costs)
    best_index = sample_costs.index(best_eval)

    print("TIME", time.time()-start_time)
    print("BEST SOL:", sampled_sols[best_index])
    print("BEST EVAL:", best_eval)



