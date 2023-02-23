import time, sys, copy, itertools, pickle
from treelib import Node, Tree
from interp_sim import gen_cost, compile_num_stages, layout, dfg, gen_cost_multitrace

'''
CONSTANTS
'''
single_stg_mem = 143360 # max number of elements for 32-bit array
single_stg_mem_log2 = 131072 # closest power of 2 for single_stg_mem (most apps require num elements to be power of 2)
single_stg_mem_log2_pairarray = 65536


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
        print("CHILD:", child.tag)
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
            # set lb for all left to find
            for v in to_find:
                symbolics_opt[v] = 1
                if v in opt_info["symbolicvals"]["symbolics"]:  # memory
                    symbolics_opt[v] = 32
                if v in opt_info["symbolicvals"]["bounds"]:
                    symbolics_opt[v] = opt_info["symbolicvals"]["bounds"][v][0]

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
            print("VAR", to_find[0])
            print("LB", lb, "\n")
            # keep going if we have more variables (to_find[1:] not empty)
            if not to_find[1:]: # we're done! we've found all vars for this path
                return
            else:   # we still have more variables to find bounds for, so keep going
                build_bounds_tree(tree, root, to_find[1:], symbolics_opt, opt_info, fullcompile, pair, efficient, dfg)


def preprocess(symbolics_opt, opt_info, o, timetest, fullcompile, exhaustive, pair, preprocessingonly, shortcut, dfg, efficient):
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

        solutions = bounds_tree.paths_to_leaves()


    if preprocessingonly:
        # dump all solutions to preprocessed.pkl
        sols = {}
        sols["all_sols"] = solutions
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
                if testing_eval:
                    current_progress = {"evaling": symbolics_opt, "best eval": min(testing_eval), "best sol": testing_sols[testing_eval.index(min(testing_eval))]}
                    with open("current.pkl", 'wb') as f:
                        pickle.dump(current_progress, f)


                if "struct" in opt_info["optparams"]:
                    if opt_info["optparams"]["struct"] != "hash" and opt_info["optparams"]["struct"] != "precision":
                        if "tables" in symbolics_opt and "entries" in symbolics_opt and "rows" in symbolics_opt and "cols" in symbolics_opt and "THRESH" in symbolics_opt:
                            if symbolics_opt["tables"]*symbolics_opt["entries"] < 8192:
                                continue
                            if symbolics_opt["rows"] < 2 or symbolics_opt["cols"] > 8192:
                                continue
                            if "skew" in opt_info["optparams"]:
                                if opt_info["optparams"]["skew"] == "less":
                                    if symbolics_opt["THRESH"] < 3000 or symbolics_opt["expire_thresh"] < 95000000:
                                        continue
                                if opt_info["optparams"]["skew"] == "uniform":
                                    if symbolics_opt["THRESH"] < 4000 or symbolics_opt["expire_thresh"] < 90000000:
                                        continue

                            # skewed
                            elif symbolics_opt["THRESH"] < 2000 or symbolics_opt["expire_thresh"] < 90000000:
                                continue
                if opt_info["lucidfile"] == "starflow.dpt":
                    if symbolics_opt["num_long"] + symbolics_opt["num_short"] < 15 or symbolics_opt["S_SLOTS"] < 16384 or symbolics_opt["L_SLOTS"] < 16384:
                        iterations += 1
                        continue
                if opt_info["lucidfile"] == "stateful_firewall.dpt":
                    if symbolics_opt["stages"] < 2 or symbolics_opt["entries"] < 65536 or symbolics_opt["timeout"] < 700000000 or symbolics_opt["interscan_delay"] < 800000000:
                        iterations += 1
                        continue

                if "interp_traces" not in opt_info:
                    cost = gen_cost(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
                else:
                    cost = gen_cost_multitrace(symbolics_opt, symbolics_opt, opt_info, o, False, "ordered")
                tested_sols.append(copy.deepcopy(symbolics_opt))

                testing_sols.append(copy.deepcopy(symbolics_opt))
                testing_eval.append(cost)
                iterations += 1
        best_sols = [{}]
        print("EXHAUSTIVE TIME", time.time() - interpreter_start_time)
        print("NUM SOLS", len(testing_eval))
        with open('final_testing_sols_ordered_exhaustive.pkl','wb') as f:
            pickle.dump(testing_sols,f)
        with open('final_testing_eval_ordered_exhaustive.pkl','wb') as f:
            pickle.dump(testing_eval,f)

    else:   # we're using one of our search strategies
        strat = opt_info["optparams"]["strategy"]

        if strat=="simannealing":
            return simulated_annealing(symbolics_opt, opt_info, o, timetest, bounds_tree, solutions)

        # TODO: incorporate other params in opt_info (alpha, sigma, etc.)
        elif strat=="neldermead":
            return nelder_mead(symbolics_opt, opt_info, o, timetest, solutions=solutions, tree=bounds_tree)

        elif strat=="bayesian":
            return bayesian(symbolics_opt, opt_info, o, timetest, solutions, bounds_tree)

    interp_time = time.time()-interpreter_start_time
    print("TOTAL INTERP TIME:", interp_time)
    #print("TOTAL UB TIME:", ub_time)

    return best_sols[0], best_cost





