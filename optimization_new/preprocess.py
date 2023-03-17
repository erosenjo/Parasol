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
    # for non memory, start @ default and increase/decrease by 1 until = 12 stgs

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


def preprocess(symbolics_opt, opt_info, o, timetest, fullcompile, pair, shortcut, dfg, efficient):
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


    sols = {}
    sols["all_sols"] = solutions
    sols["tree"] = bounds_tree
    sols["time(s)"] = time.time()-opt_start_time

    return sols



