import math, json, sys, copy, time
from random import randint, random, getrandbits, choice
from interp_sim import gen_cost, compile_num_stages
import numpy as np
from scipy.optimize import basinhopping
import pickle
#from search_ilp import solve
from treelib import Node, Tree

# CONSTANTS
single_stg_mem = 143360 # max number of elements for 32-bit array
single_stg_mem_log2 = 131072 # closest power of 2 for single_stg_mem (most apps require num elements to be power of 2)

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



# RANDOM OPTIMIZATION
# randomly choose, given some var
def get_next_random(symbolics_opt, logs, bounds, structchoice, structinfo):
    new_vars = {}
    exclude = False
    if structchoice:
        new_vars[structinfo["var"]] = bool(getrandbits(1))
        if str(new_vars[structinfo["var"]]) in structinfo:
            exclude = True
    for var in symbolics_opt:
        if structchoice and var==structinfo["var"]: # we've handled this above, don't do it again
            continue
        if exclude and var in structinfo[str(new_vars[structinfo["var"]])]: # exlucding this var for struct
            new_vars[var] = symbolics_opt[var]
            continue
        if var in logs.values(): # this var has to be multiple of 2
            new_vars[var] = 2**randint(int(math.log2(bounds[var][0])),int(math.log2(bounds[var][1])))
            continue
        new_vars[var] = randint(bounds[var][0],bounds[var][1])
    return new_vars

def random_opt(symbolics_opt, opt_info, o, timetest):
    iterations = 0


    # TODO: this is silly, but not sure a better way to do this rn
    # using 0 bc these values don't matter, will get overwritten at some point
    for var_group in opt_info["symbolicvals"]["ilp_vars"]:
        opt_info["symbolicvals"]["symbolics"][var_group["int"]["name"]] = 0
        opt_info["symbolicvals"]["sizes"][var_group["size"]["name"]] = 0

    # solve ilp to get vals for other reg arrays
    const_vars = opt_info["symbolicvals"]["const_vars"] 
    ilp_sol = solve(opt_info["num_stgs"], opt_info["total_mem"], opt_info["hashes"], opt_info["symbolicvals"]["ilp_vars"], const_vars)
    # add ilp result to symbolics
    full_symbolics_opt = {**symbolics_opt, **ilp_sol} 
    # init best solution as starting, and best cost as inf
    best_sols = [copy.deepcopy(full_symbolics_opt)]
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
            # 10 min (< 30)
            if 600 <= (curr_time - start_time) < 1800:
                with open('10min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 30 min
            if 1800 <= (curr_time - start_time) < 2700:
                with open('30min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 45 min
            if 2700 <= (curr_time - start_time) < 3600:
                with open('45min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 60 min
            if 3600 <= (curr_time - start_time) < 5400:
                with open('60min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 90 min
            if 5400 <= (curr_time - start_time) < 7200:
                with open('90min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 120  min (end)
            if 7200 <= (curr_time - start_time):
                with open('120min_testing_sols_rand.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_rand.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                break



        # get cost
        cost = gen_cost(full_symbolics_opt,full_symbolics_opt,opt_info, o,False)

        testing_sols.append(copy.deepcopy(full_symbolics_opt))
        testing_eval.append(cost)


        if cost == -1:  # doesn't fit w/in stgs, not counting this as an iteration
            print("TOO MANY STGS")
            for var_group in opt_info["symbolicvals"]["ilp_vars"]:
                var_group["size"]["ub"] = full_symbolics_opt[var_group["size"]["name"]] - 1
                print(var_group["size"]["ub"])
           

        else: 
            # add sol to tested_sols to count it as already evaluated
            # is it stupid to do deepcopy here? can we be smarter about changing symbolics_opt to avoid this? or is it a wash?
            # NOTE: ONLY adding randomly chosen symbolics to this, not ones from ilp
            tested_sols.append(copy.deepcopy(symbolics_opt))


            # if new cost < best, replace best (if stgs <= tofino)
            if cost < best_cost:
                best_cost = cost
                # not sure if this is slow, but these dicts are likely small (<10 items) so shouldn't be an issue
                best_sols = [copy.deepcopy(full_symbolics_opt)]
            elif cost == best_cost:
                best_sols.append(copy.deepcopy(full_symbolics_opt))

            # get next values (only if this actually fit w/in stgs, otherwise try again w/ same vals)
            symbolics_opt = get_next_random(symbolics_opt, opt_info["symbolicvals"]["logs"], opt_info["symbolicvals"]["bounds"], structchoice, structinfo)

            # update const_vars
            for var_group in const_vars:
                var_group["int"]["value"] = symbolics_opt[var_group["int"]["name"]]
                var_group["size"]["value"] = symbolics_opt[var_group["size"]["name"]]

        # solve ilp
        ilp_sol = solve(opt_info["num_stgs"], opt_info["total_mem"], opt_info["hashes"], opt_info["symbolicvals"]["ilp_vars"], const_vars)
        # update symbolics opt w ilp sol
        full_symbolics_opt = {**symbolics_opt, **ilp_sol}
        # incr iterations (only if this actually fit w/in stgs, otherwise doesn't count)
        if cost != -1:
            iterations += 1
            # remove upper bound for ilp vars??? consequence of this is we'll have to run ilp A LOT more, but it's fast
            # (also means we have to compile to p4 more often, which can be slow)
            for var_group in opt_info["symbolicvals"]["ilp_vars"]:
                if "ub" in var_group["size"]:
                    del var_group["size"]["ub"]

    # if we have multiple solutions equally as good, use priority from user to narrow it down to 1
    #best_sol = prioritize(best_sols,opt_info)

    with open('testing_sols_rand.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('testing_eval_rand.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    # return the first solution in list of acceptable sols
    return best_sols[0], best_cost


# SIMULATED ANNEALING
# copied from: https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
# good for objective functions that have single global optima and mult local optima where search might get stuck
# using numpy.random.randn() - gaussian distr (gives us negative vals, so can move up and down)  
# some notes on step size:
#   if the var has large search space, probably want larger step size
#   step size is std dev --> 99% of all steps w/in 3*stepsize of curr var val
#   ^ not exactly true, bc we have to do some rounding (can't have floats)
def simulated_annealing(symbolics_opt, opt_info, o, timetest):
    temp = opt_info["optparams"]["temp"]
    bounds = opt_info["symbolicvals"]["bounds"]
    step_size = opt_info["optparams"]["stepsize"]
    logvars = opt_info["symbolicvals"]["logs"].values()

    structchoice = False
    structinfo = {}
    if "structchoice" in opt_info:
        structchoice = True
        structinfo = opt_info["structchoice"]

    # generate and evaluate an initial point
    best_sols = [copy.deepcopy(symbolics_opt)]
    best_cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False)

    # current working solution
    curr, curr_cost = copy.deepcopy(symbolics_opt), best_cost


    # list of output for each iteration
    testing_sols = [copy.deepcopy(symbolics_opt)]
    testing_eval = [best_cost]

    # run the algorithm
    for i in range(opt_info["optparams"]["stop_iter"]-1):   # minus 1 bc counting init cost as iteration
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
            if symbolics_opt[s] < bounds[s][0]:
                symbolics_opt[s] = bounds[s][0]
            elif symbolics_opt[s] > bounds[s][1]:
                symbolics_opt[s] = bounds[s][1]
            # we'll get a float, so round to nearest acceptable int
            if s in logvars:    # var has to be power of 2
                symbolics_opt[s] = closest_power(symbolics_opt[s])
            else:   # doesn't have to be power of 2, just round to nearest int
                symbolics_opt[s] = round(symbolics_opt[s])


        # evaluate candidate point
        candidate_cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False)

        # check for new best solution
        if candidate_cost < best_cost:
            # store new best point
            best_sols, best_cost = [copy.deepcopy(symbolics_opt)], candidate_cost
        elif candidate_cost == best_cost:
            best_sols.append(copy.deepcopy(symbolics_opt))

        # difference between candidate and current point evaluation
        diff = candidate_cost - curr_cost

        # calculate temperature for current epoch
        t = temp / float(i + 1)

        # calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t)

        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_cost = copy.deepcopy(symbolics_opt), candidate_cost

        testing_sols.append(copy.deepcopy(symbolics_opt))
        testing_eval.append(candidate_cost)
        
    #best_sol = prioritize(best_sols,opt_info)
    with open('testing_sols_sa.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('testing_eval_sa.pkl','wb') as f:
        pickle.dump(testing_eval,f)

    # return first sol in list of sols
    return best_sols[0], best_cost

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
            cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False)

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

# start at 1 (or whatever lower bound is), keep increasing until we hit max num stgs used
# compile to p4 each time to see how many stgs
# we start at lower bound bc we know that we can't be < 1; harder to start at upper bound bc we don't really know that would be, unless user tells us (which forces them to reason a bit about resources)
def get_max_val(symbolics_opt, var_to_opt, opt_info, log2, memory):
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
    best_stgs = {}   # stgs used by each solution we try
    while True:
        print("SYM COMPILING:", symbolics_opt)
        # compile it and get num stages used
        stgs_used = compile_num_stages(symbolics_opt, opt_info)
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
                best_stgs[symbolics_opt[var_to_opt]] = stgs_used
                if memory and symbolics_opt[var_to_opt] == single_stg_mem_log2:
                    return symbolics_opt[var_to_opt], best_stgs
                symbolics_opt[var_to_opt] *= 2
            else:
                best_stgs[symbolics_opt[var_to_opt]] = stgs_used
                if memory and symbolics_opt[var_to_opt] == single_stg_mem:
                    return symbolics_opt[var_to_opt], best_stgs
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
            return symbolics_opt[var_to_opt], best_stgs

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


# TODO: is this stupid? is there a better way to do this??? idk but i'm getting soooooo confused
# basically we're building a tree
# each node is a concrete choice for a var, and the children are possible choices for the var that's the next level down
def build_bounds_tree(tree, root, to_find, symbolics_opt, opt_info):
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
        build_bounds_tree(tree, child.identifier, to_find, symbolics_opt, opt_info)

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
            ub, stgs_used = get_max_val(symbolics_opt,to_find[0], opt_info, log2, True)
            #tree.create_node((to_find[0],ub), parent=root)
        else:
            # keep compiling until we hit max stgs, get ub 
            #print("FIND BOUNDS")
            symbolics_opt[to_find[0]] = lb
            ub, stgs_used = get_max_val(symbolics_opt, to_find[0], opt_info, log2, False)
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
            build_bounds_tree(tree, root, to_find[1:], symbolics_opt, opt_info)

def choose_random_resource(tree, symbolics_opt, solutions):
    sol_choice = choice(solutions)
    for sol in sol_choice:
        node = tree.get_node(sol)
        if node.tag=="root":
            continue
        symbolics_opt[node.tag[0]] = node.tag[1]
    return

    '''
    # start at root
    # at each level, choose random child as the val for that variable
    if not tree.children(root): # if we're at the bottom, we're done, so return
        return
    var_choice = choice(tree.children(root))
    symbolics_opt[var_choice.tag[0]] = var_choice.tag[1]
    choose_random_resource(tree,symbolics_opt,var_choice.identifier)
    return
    '''

# testing out ordered parameter search
def ordered(symbolics_opt, opt_info, o, timetest, nopruning):
    opt_start_time = time.time()

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
    build_bounds_tree(bounds_tree,"root", opt_info["optparams"]["order_resource"], symbolics_opt, opt_info)

    print("UPPER BOUND TIME:", time.time()-opt_start_time)    

    # STEP 2: prune solutions found in step 1 by throwing out solutions that use less resources (memory, stgs) than others
    # iterate through each path in tree
    # need a formula for calculating total memory = x * y + j * k
    # once we calc total resources, remove solutions that are < max
    # (NOTE: is there any case where < max is preffered?) (maybe keep a few that are the next closest?)
    # TODO: sort by things other than memory --> hash units, reg accesses?
    solutions = bounds_tree.paths_to_leaves()
    sols_by_mem = {}
    sols_by_stgs = {}
    mem_formula = opt_info["optparams"]["mem_formula"]
    for sol in solutions:
        mem_formula = opt_info["optparams"]["mem_formula"]
        for n_identifier in sol:
            # TODO: better way to do this?
            node = bounds_tree.get_node(n_identifier)
            if node.tag == "root":
                continue
            # replace var name with val of vars in memory formula in json file
            var_name = node.tag[0]
            var_value = node.tag[1]
            # replace string with val of var
            mem_formula = mem_formula.replace(var_name, str(var_value))
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

    print("UB+PRUNE TIME:", time.time()-opt_start_time)

    print("TOTAL SOLS:", len(solutions))
    #print(sols_by_mem.keys())
    #print("MAX MEM", max(list(sols_by_mem.keys())))
    best_mem_sols = sols_by_mem[max(list(sols_by_mem.keys()))]
    best_stgs_sols = sols_by_stgs[max(list(sols_by_stgs.keys()))]
    print("MAX MEM SOLS", len(best_mem_sols))
    print("MAX STGS SOLS", len(best_stgs_sols))
    best_mem_stgs = [sol for sol in best_mem_sols if sol in best_stgs_sols]
    print("OVERLAP SOLS (MEM+STGS)", len(best_mem_stgs))

    # run interpreter on ones w/ most memory first, then maybe next highest???
    # STEP 3: run interpreter to get cost, optimize for non-resource parameters
    # search through that parameter space, using interpreter to get cost
    # pick resource params: pick one child from each level of tree (checking for logs as we go)
    # pick non-resource param: some value w/in user-defined bounds
    # set any rule-based variables
    # run the interpreter!
    #print("SYMBOLICS BEFORE:", symbolics_opt)

    iterations = 0
    best_sols = []
    best_cost = float("inf")
    tested_sols = []
    iters = False
    simtime = False
    iter_time = False

    if "stop_iter" in opt_info["optparams"]:
        iters = True
    if "stop_time" in opt_info["optparams"]:
        simtime = True
    if iters and simtime:
        iter_time = True

    # TODO: add struct choice stuff, for caching
    # TODO: change time test to do overall time, not just interpreter??
    testing_sols = []
    testing_eval = []
    # start the interpreter loop
    interpreter_start_time = time.time()
    # print out time it took to get search space
    print("SEARCH TIME(s):", interpreter_start_time-opt_start_time)
    while True:
        print("ITERATION:", iterations)
        # check iter/time conditions
        if iters or iter_time:
            if iterations >= opt_info["optparams"]["stop_iter"]:
                break
        if simtime or iter_time:
            if (time.time()-interpreter_start_time) >= opt_info["optparams"]["stop_time"]:
                break
        curr_time = time.time()
        # TIME TEST (save sols/costs we've evaled up to this point)
        if timetest:
            # 5 min (< 10)
            if 300 <= (curr_time - interpreter_start_time) < 600:
                with open('5min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('5min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 10 min (< 30)
            if 600 <= (curr_time - interpreter_start_time) < 1800:
                with open('10min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('10min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 30 min
            if 1800 <= (curr_time - interpreter_start_time) < 2700:
                with open('30min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('30min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 45 min
            if 2700 <= (curr_time - interpreter_start_time) < 3600:
                with open('45min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('45min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 60 min
            if 3600 <= (curr_time - interpreter_start_time) < 5400:
                with open('60min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('60min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 90 min
            if 5400 <= (curr_time - interpreter_start_time) < 7200:
                with open('90min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('90min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
            # 120  min (end)
            if 7200 <= (curr_time - interpreter_start_time):
                with open('120min_testing_sols_ordered.pkl','wb') as f:
                    pickle.dump(testing_sols,f)
                with open('120min_testing_eval_ordered.pkl','wb') as f:
                    pickle.dump(testing_eval,f)
                break

        # choose values to evaluate (randomly for now)
        # if we're not doing the pruning stage, pick from all possible paths in the tree
        if nopruning:
            choose_random_resource(bounds_tree, symbolics_opt, solutions)
        # if we are pruning, pick from list of sols w/ highest memory usage (TODO - other resources?)
        else:
            choose_random_resource(bounds_tree, symbolics_opt, sols_by_mem[max(list(sols_by_mem.keys()))])

        # TODO: for now, randomly picking values for non resource params
        for nonresource in opt_info["optparams"]["non_resource"]:
            symbolics_opt[nonresource] = choice(range(opt_info["symbolicvals"]["bounds"][nonresource][0], opt_info["symbolicvals"]["bounds"][nonresource][1]))

        # NOTE: right now this rule stuff is very specific to conquest, and i don't think it appears anywhere else?
        # this is just for cases where the symbolic value = some mathematic expression of other variables
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
                        rule[v] = symbolics_opt[rule[v]]
            symbolics_opt[rulevar] = eval(''.join(rule))


        print("SYMBOLICS TO EVAL", symbolics_opt)
        cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False)

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

        # incr iterations
        iterations += 1


    with open('testing_sols_ordered.pkl','wb') as f:
        pickle.dump(testing_sols,f)
    with open('testing_eval_ordered.pkl','wb') as f:
        pickle.dump(testing_eval,f)



    return best_sols[0], best_cost

    '''
    bounds = {}
    # for non-reg,resources params (sizes, NOT symbolics), do some preprocessing to get upper bounds (cut down search space)
    for var in opt_info["optparams"]["order"]:
        # if variable is num regs in an array,  we can just set ub to 65536, lb = 2 (unless user provides some other bounds)
        # TODO: this will probs change when lucid compiler gets memory update
        if var in opt_info["symbolicvals"]["symbolics"]:
            if var in opt_info["symbolicvals"]["bounds"]
                bounds[var] = [opt_info["symbolicvals"]["bounds"][0],opt_info["symbolicvals"]["bounds"][1]]
            else:
                bounds[var] = [2, 65536]
            continue


        # set everything to 1 to start with
        symbolics_opt = dict.fromkeys(symbolics_opt, 1)
        lb = 1
        log2 = False
        # if we have a lower bound for var, use it
        if var in opt_info["symbolicvals"]["bounds"]:
            lb = opt_info["symbolicvals"]["bounds"][var][0]
        # if we need log2 of this var, then lb should be 2 (bc if it's 1, then log2 = 0, and we never want vars < 1)
        if var in opt_info["symbolicvals"]["logs"].values():
            print("LOG")
            log2 = True
            # don't overwrite the lb to 2 if the user gives us bound > 2
            if lb < 2:
                lb = 2 
        symbolics_opt[var] = lb
        # keep compiling until we hit max stgs, get ub
        ub = get_max_val(symbolics_opt, var, opt_info, log2)
        # bounds are INCLUSIVE
        bounds[var] = [lb,ub]
        print(bounds)
        exit()
     

    return [], []    
    '''

    '''
    if opt_info["optparams"]["search"] == "simple": # this is for examples like cms, hash table (single data structure, no non-resource params)
        # there should be an order to how we pick parameters
        # for now assume that we're always gonna max it out, bc they're all resource params
        # if symbolic, assume it's regs, and we know max will be 65536
        # if size, assume arrays
        # TODO: either specify this in json or get it from the program
        for var in order:
            if var in opt_info["symbolicvals"]["symbolics"]:    # it's regs, max 65536
                symbolics_opt[var] = 65536
            elif var in opt_info["symbolicvals"]["sizes"]:  # it's reg array, need to compile to get max val
                # get_max_val()
            else:
                sys.exit("Variable in [optparams][order] not in [symbolicvals]")
                 

    #elif ??? == "": # this is for structs like fridge, flowlet switching (single struct, one non-resource param)

    # this is for starflow
    '''


# bayesian optimization

# genetic algo





