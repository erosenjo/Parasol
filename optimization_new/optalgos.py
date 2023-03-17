import math, copy, time, itertools, pickle
import numpy as np
from random import choice
from interp_sim import gen_cost, gen_cost_multitrace
from treelib import Node, Tree
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge

'''
HELPER FUNCTIONS
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


def set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree, opt_info):
    for sol in sol_choice:
        node = tree.get_node(sol)
        if node.tag=="root":
            continue
        symbolics_opt[node.tag[0]] = node.tag[1]
    # set rule vars
    set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt


# nelder-mead uses numpy arrays, so this function converts that to symbolics opt, so we can run interpreter
# NOTE: "candidate_index" is reserved, can't use for variable name
def set_symbolics_from_nparray(nparray, index_dict, symbolics_opt,
                               opt_info, solutions=[], tree=None):
    sol_index = int(nparray[0])
    symbolics_opt = solutions[sol_index]

    # set any rule-based vars
    if "rules" in opt_info["symbolicvals"]:
        symbolics_opt = set_rule_vars(opt_info, symbolics_opt)

    return symbolics_opt


# NOTE: this is for BOTH preprocessed and non solutions
def gen_next_simannealing(solutions, opt_info, symbolics_opt, curr, curr_index):
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
                        solutions=[], bounds_tree=None):

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

    # enumerating all solutions, optimizing ONLY index value
    #   (aka treating resource and non resource the same)
    all_solutions_symbolics = []
    if not solutions:   # we didn't preprocess
        non_preprocess_ranges = {}
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                non_preprocess_ranges[bounds_var] = [2**var_val for var_val in list(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))]
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            non_preprocess_ranges[bounds_var] = list(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))

        # get all possible solutions given bounds
        # dict.keys() and dict.values() SHOULD be in same order according to documentation
        # (as long as no changes are made to dict in between calls)
        # itertools.product should return in the same order as lists that are passed to it
        possible_sols = list(itertools.product(*list(non_preprocess_ranges.values())))
        for sol in possible_sols:
            symbolics = {}
            sol_index = 0
            for var in non_preprocess_ranges:
                symbolics[var] = sol[sol_index]
                sol_index += 1
            if opt_info["lucidfile"]=="caching.dpt" and "entries" in symbolics and "tables" in symbolics:
                if symbolics["entries"] * symbolics["tables"] > 10000:
                    continue
                if "struct" in opt_info["optparams"] and opt_info["optparams"]["struct"]=="hash":
                    if symbolics["tables"] > 1:
                        continue
                if "rows" in symbolics and "cols" in symbolics:
                    if symbolics["rows"] * symbolics["cols"] > 10000:
                        continue
            all_solutions_symbolics.append(symbolics)


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

    total_sols = len(all_solutions_symbolics)

    # start time
    start_time = time.time()


    # start at randomly chosen values
    # only optimize index var
    symbolics_opt = choice(all_solutions_symbolics)
    candidate_index = all_solutions_symbolics.index(symbolics_opt)


    # generate and evaluate an initial point
    best_sols = [copy.deepcopy(symbolics_opt)]
    if "interp_traces" not in opt_info: # single trace for opt, named the same as dpt file
        # if solutions is empty, no preprocessing, regular sim annealing (need to compile before interpreter)
        if not solutions:
            best_cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "simannealing")
        else:
            best_cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "preprocessed")
    else: # multiple training traces, arbitrary names
        if not solutions:
            best_cost = gen_cost_multitrace(symbolics_opt,symbolics_opt,opt_info, o,False, "simannealing")
        else:
            best_cost = gen_cost_multitrace(symbolics_opt,symbolics_opt,opt_info, o,False, "preprocessed")

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
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_sa.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_sa.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                break


        # if we've tested every solution, quit
        if len(tested_sols) == total_sols:
            break

        # TODO: should we test repeated sols????
        #while symbolics_opt in tested_sols: # don't bother with repeated sols
        symbolics_opt, candidate_index = gen_next_simannealing(all_solutions_symbolics, opt_info, symbolics_opt, curr, curr_index)

        # evaluate candidate point
        if "interp_traces" not in opt_info:
            if not solutions:
                candidate_cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "simannealing")
            else:
                candidate_cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
        else:
            if not solutions:
                candidate_cost = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "simannealing")
            else:
                candidate_cost = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

        if symbolics_opt not in tested_sols:
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
            curr_index = candidate_index

        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(candidate_cost)
        print("BEST", best_cost)
        
    #best_sol = prioritize(best_sols,opt_info)
    with open('final_testing_sols_sa.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('final_testing_eval_sa.pkl','wb') as f:
        pickle.dump(testing_eval,f)


    # return first sol in list of sols
    return best_sols[0], best_cost

# EXHAUSTIVE SEARCH
def exhaustive(symbolics_opt, opt_info, o, timetest, solutions, bounds_tree):
    print("EXHAUSTIVE")

    # get list of all possible sols, then iterate through them all
    all_solutions_symbolics = []
    if not solutions:   # no preprocessing
        non_preprocess_ranges = {}
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                #vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                non_preprocess_ranges[bounds_var] = [2**var_val for var_val in list(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))]
                continue
            #vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            non_preprocess_ranges[bounds_var] = list(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))

        # get all possible solutions given bounds
        # dict.keys() and dict.values() SHOULD be in same order according to documentation
        # (as long as no changes are made to dict in between calls)
        # itertools.product should return in the same order as lists that are passed to it
        possible_sols = list(itertools.product(*list(non_preprocess_ranges.values())))
        for sol in possible_sols:
            symbolics = {}
            sol_index = 0
            for var in non_preprocess_ranges:
                symbolics[var] = sol[sol_index]
                sol_index += 1
            if opt_info["lucidfile"]=="caching.dpt" and "entries" in symbolics and "tables" in symbolics:
                if symbolics["entries"]*symbolics["tables"] > 10000:
                    continue
            if "rows" in symbolics and "cols" in symbolics:
                if symbolics["rows"]*symbolics["cols"] > 10000:
                    continue
            all_solutions_symbolics.append(symbolics)

    else:
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


    start_time = time.time()

    testing_sols = []
    testing_eval = []
    # init best solution, best cost as inf
    best_sols = []
    best_cost = float("inf")
    for sol in all_solutions_symbolics:
        if "interp_traces" not in opt_info:
            cost = gen_cost(sol, sol, opt_info, o, False, "exhaustive")
        else:
            cost = gen_cost_multitrace(sol, sol, opt_info, o, False, "exhaustive")
        testing_sols.append(sol)
        testing_eval.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_sols=[sol]
        elif cost == best_cost:
            best_sols.append(sol)

        current_progress = {"evaling": sol, "best eval": min(testing_eval), "best sol": testing_sols[testing_eval.index(min(testing_eval))], "iteration": all_solutions_symbolics.index(sol), "total sols": len(all_solutions_symbolics)}
        with open("progress.pkl", 'wb') as f:
            pickle.dump(current_progress, f)


    with open('testing_sols_exhaustive.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('testing_eval_exhaustive.pkl','wb') as f:
        pickle.dump(testing_eval,f)


    print("EXHAUSTIVE SEARCH TIME:", time.time()-start_time)
    print("NUM SOLS:", len(all_solutions_symbolics))

    return best_sols[0], best_cost



# choosing more opt strategies
# some considerations:
#   we're doing simulations, so we don't know if the function is differentiable or not
#   we could try some strategies for differentiable obj functions, but putting that on the backburner to focus on strategies for simulation-based functions


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
    all_solutions_symbolics = []
    if not solutions:   # we didn't preprocess
        non_preprocess_ranges = {}
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                non_preprocess_ranges[bounds_var] = [2**var_val for var_val in list(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))]
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            non_preprocess_ranges[bounds_var] = list(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))

        # get all possible solutions given bounds
        # dict.keys() and dict.values() SHOULD be in same order according to documentation
        # (as long as no changes are made to dict in between calls)
        # itertools.product should return in the same order as lists that are passed to it
        possible_sols = list(itertools.product(*list(non_preprocess_ranges.values())))
        for sol in possible_sols:
            symbolics = {}
            sol_index = 0
            for var in non_preprocess_ranges:
                symbolics[var] = sol[sol_index]
                sol_index += 1
            if opt_info["lucidfile"]=="caching.dpt" and "entries" in symbolics and "tables" in symbolics:
                if symbolics["entries"] * symbolics["tables"] > 10000:
                    continue
                if "struct" in opt_info["optparams"] and opt_info["optparams"]["struct"]=="hash":
                    if symbolics["tables"] > 1:
                        continue
                if "rows" in symbolics and "cols" in symbolics:
                    if symbolics["rows"] * symbolics["cols"] > 10000:
                        continue
            all_solutions_symbolics.append(symbolics)


    else:   # we've preprocessed, used those solutions to calc total
        # TODO: create symbolics_opt from solutions and append to all_solutions_symbolics
        for sol_choice in solutions:
            all_solutions_symbolics.append(copy.deepcopy(set_symbolics_from_tree_solution(sol_choice, symbolics_opt, tree, opt_info)))
        for nonresource in opt_info["optparams"]["non_resource"]:
                #total_sols *= len(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]+opt_info["optparams"]["stepsize"][nonresource], opt_info["optparams"]["stepsize"][nonresource]))
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

    total_sols = len(all_solutions_symbolics)


    # randomly generate starting solution
    # NEW, only optimize for index val
    symbolics_opt = choice(all_solutions_symbolics)
    candidate_index = all_solutions_symbolics.index(symbolics_opt)
    print("STARTING", candidate_index)
    print(symbolics_opt)

    # create starting numpy array and index_dict (index: var_name)
    # also create step and bounds arrays
    # step: look-around radius in initial step (1 val for each dimension)
    index_dict = {}
    # dimensions = number of non resource vars + 1
    # NEW, only optimize for index val
    dimensions = 1
    x_start = np.zeros(dimensions)
    step = np.zeros(dimensions)
    bounds = np.zeros(shape=(dimensions,2))
    index = 0
    x_start[index] = candidate_index
    index_dict[index] = "candidate_index"
    #step[index] = 1
    #step[index] = 10
    step[index] = 5
    bounds[index][0] = 0
    # NEW, only optimize for index val
    bounds[index][1] = len(all_solutions_symbolics) - 1
    print("X START", x_start)
    print("INDEX DICT", index_dict)
    print("STEP", step)
    print("BOUNDS", bounds)


    # init
    dim = len(x_start)
    # NEW, only optimize for index val
    symbolics_opt = set_symbolics_from_nparray(x_start, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
    print("symbolics opt after set from np array", symbolics_opt)
    #prev_best = f(x_start)
    if "interp_traces" not in opt_info: # single trace, same name as dpt file
        if not solutions:
            prev_best = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
        else:
            print("EVALED SOL", symbolics_opt)
            prev_best = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

    else: # multiple traces, arbitrary name
        if not solutions:
            prev_best = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
        else:
            print("EVALED SOL", symbolics_opt)
            prev_best = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
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
        # NEW, only optimize for index val
        symbolics_opt = set_symbolics_from_nparray(x, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)

        #score = f(x)
        if "interp_traces" not in opt_info: # single trace, same name as dpt file
            if not solutions:
                score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL", symbolics_opt)
                score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

        else: # multiple traces, arbitrary name
            if not solutions:
                score = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL", symbolics_opt)
                score = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

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
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_nm.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_nm.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
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
        if "interp_traces" not in opt_info: # single trace, same name as dpt file
            if not solutions:
                rscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL", symbolics_opt)
                rscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

        else: # multiple traces, arbitrary name
            if not solutions:
                rscore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL AFTER REFLECTION", symbolics_opt)
                rscore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(rscore)
        print("RES RELFECTION", res)
        if res[0][1] <= rscore < res[-1][1]:
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
            # NEW, only optimize for index val
            print("XE",xe)
            print(len(all_solutions_symbolics))
            symbolics_opt = set_symbolics_from_nparray(xe, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
            print(xe[0])
            print(symbolics_opt)
            print(all_solutions_symbolics[int(xe[0])])
            #exit()
            #escore = f(xe)
        if "interp_traces" not in opt_info: # single trace, same name as dpt file
            if not solutions:
                escore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL", symbolics_opt)
                escore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

        else: # multiple traces, arbitrary name
            if not solutions:
                escore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL AFTER EXPANSION", symbolics_opt)
                escore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
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
        # NEW, only optimize for index val
        symbolics_opt = set_symbolics_from_nparray(xc, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
        print("XC", xc[0])
        print("RES", res)
        print(symbolics_opt)
        print(all_solutions_symbolics[int(xc[0])])
        #exit()
        #cscore = f(xc)
        if "interp_traces" not in opt_info: # single trace, same name as dpt file
            if not solutions:
                cscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL", symbolics_opt)
                cscore = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

        else: # multiple traces, arbitrary name
            if not solutions:
                cscore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
            else:
                print("EVALED SOL AFTER CONTRACTION", symbolics_opt)
                cscore = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
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
            # NEW, only optimize for index val
            if early_exit:
                break
            symbolics_opt = set_symbolics_from_nparray(redx, index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
            print(redx[0])
            print(symbolics_opt)
            print(all_solutions_symbolics[int(redx[0])])
            #exit()
            #score = f(redx)
            if "interp_traces" not in opt_info: # single trace, same name as dpt file
                if not solutions:
                    score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
                else:
                    print("EVALED SOL", symbolics_opt)
                    score = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")

            else: # multiple traces, arbitrary name
                if not solutions:
                    score = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "neldermead")
                else:
                    print("EVALED SOL AFTER REDUCTION", symbolics_opt)
                    score = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "preprocessed")
            testing_sols.append(copy.deepcopy(symbolics_opt))
            testing_eval.append(score)
            nres.append([redx, score])
            print("new redx reduction, redx")
        res = nres
        if res[0][0] == res[1][0]:
            print("SAME, reduction")
            break


    print("TESTED SOLS", testing_sols)
    print("TESTED EVALS", testing_eval)
    print("BEST EVAL", min(testing_eval))
    print("BEST SOL", testing_sols[testing_eval.index(min(testing_eval))])
    print("RES", res)

    # NEW, only optimize for index val
    best_sol = set_symbolics_from_nparray(res[0][0], index_dict, symbolics_opt, opt_info, all_solutions_symbolics, tree)
    with open('final_testing_sols_nm.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('final_testing_eval_nm.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    return best_sol, res[0][1]

    
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
    all_solutions_symbolics = []
    if not solutions:   # we didn't preprocess
        non_preprocess_ranges = {}
        for bounds_var in opt_info["symbolicvals"]["bounds"]:
            bound = opt_info["symbolicvals"]["bounds"][bounds_var]
            if bounds_var in opt_info["symbolicvals"]["logs"].values():
                vals = len(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))
                non_preprocess_ranges[bounds_var] = [2**var_val for var_val in list(range(int(math.log2(bound[0])), int(math.log2(bound[1]))+1))]
                continue
            vals = len(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))
            non_preprocess_ranges[bounds_var] = list(range(bound[0], bound[1]+opt_info["optparams"]["stepsize"][bounds_var], opt_info["optparams"]["stepsize"][bounds_var]))

        # get all possible solutions given bounds
        # dict.keys() and dict.values() SHOULD be in same order according to documentation
        # (as long as no changes are made to dict in between calls)
        # itertools.product should return in the same order as lists that are passed to it
        possible_sols = list(itertools.product(*list(non_preprocess_ranges.values())))
        for sol in possible_sols:
            symbolics = {}
            sol_index = 0
            for var in non_preprocess_ranges:
                symbolics[var] = sol[sol_index]
                sol_index += 1
            if opt_info["lucidfile"]=="caching.dpt" and "entries" in symbolics and "tables" in symbolics:
                if symbolics["entries"] * symbolics["tables"] > 10000:
                    continue
                if "struct" in opt_info["optparams"] and opt_info["optparams"]["struct"]=="hash":
                    if symbolics["tables"] > 1:
                        continue
                if "rows" in symbolics and "cols" in symbolics:
                    if symbolics["rows"] * symbolics["cols"] > 10000:
                        continue
            all_solutions_symbolics.append(symbolics)

    else:   # we've preprocessed, used those solutions to calc total
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

    total_sols = len(all_solutions_symbolics)
    print("TOTAL_SOLS", total_sols)

    # sample x% of solutions
    if "samplesize" in opt_info["optparams"]:
        sample_size=int(opt_info["optparams"]["samplesize"]*total_sols)
    else:
        #sample_size = int(0.05*total_sols)
        sample_size = int(0.01*total_sols)
        #sample_size = int(0.1*total_sols)
    print("SAMPLE SIZE", sample_size)
    sampled_sols = []
    sample_xvals = []


    # grid(ish) sampling
    # instead of random, sample every total_samples/sample_size samples
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
        # single trace file
        if "interp_traces" not in opt_info:
            if not solutions:
                score = gen_cost(sample, sample, opt_info, o, False, "bayesian")
            else:
                score = gen_cost(sample, sample, opt_info, o, False, "preprocessed")
        # else, multi trace
        else:
            if not solutions:
                score = gen_cost_multitrace(sample, sample, opt_info, o, False, "bayesian")
            else:
                score = gen_cost_multitrace(sample, sample, opt_info, o, False, "preprocessed")
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
    # use index in sol list, not var value
    # pick min value, eval with gen_cost, refit model w/ actual, then repeat
    # keep track of current min

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
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('10min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('30min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('45min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('60min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('90min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_best_sols_bayesian.pkl','wb') as f:
                    pickle.dump(best_sols,f)
                with open('120min_actual_eval_bayesian.pkl','wb') as f:
                    pickle.dump(actual_eval,f)
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
        if "interp_traces" not in opt_info: # single trace, same name as dpt file
            if not solutions:
                score = gen_cost(sol_choice, sol_choice, opt_info, o, False, "bayesian")
            else:
                score = gen_cost(sol_choice, sol_choice, opt_info, o, False, "preprocessed")
        else:   # multiple traces, arbitrary names
            if not solutions:
                score = gen_cost_multitrace(sol_choice, sol_choice, opt_info, o, False, "bayesian")
            else:
                score = gen_cost_multitrace(sol_choice, sol_choice, opt_info, o, False, "preprocessed")
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

    return sampled_sols[best_index], best_eval

