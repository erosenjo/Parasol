from gurobipy import *
import json

# adapt for parasol
# used for (hopefully) cutting down search space
# pick some (resource-related) variables, use ilp to pick other (resource-related) vars
# we always want to use as much mem as possible, so obj is max mem (or other limiting resource)
# TODO:
#   deps????
#   assume 1 reg array per stg?
#   constraints assume 32 bit regs
#   CLOSEST POWER OF 2: we can't enforce sol is power of 2 for ints, so we'll round down to closest power of 2 (if we round up, we could use more mem than avail)

# changes from original:
#   remove phv/metadata constraint
#   remove tcam constraints
#   ignoring stateful alu constraint (hashes more limited?)

# starflow:
#   vars: s_slots (cols), l_slots (cols), num_long (rows), max_short_idx (rows - 1)
#   pick s_slots and max_short_idx, then use ilp to get num_long and l_slots?
#   have lower bound for num_long/l_slots, if sol < lb, throw it out
#   obj: max num_long, l_slots


# helper funcs
def highestPowerof2(n): # from: https://www.geeksforgeeks.org/highest-power-2-less-equal-given-number/
    res = 0;
    for i in range(n, 0, -1):     
        # If i is a power of 2
        if ((i & (i - 1)) == 0):
            res = i;
            break;
    return res;

def solve():
    # Model
    solver = Model("solver")

    # read in constants from json file
    # these include: resource constraints of switch, value of var chosen by opt, what variable we need to solve for (and any bounds), what resource to opt for/maximize?
    opt_info = {}
    with open("ilp_info.json") as f:
        opt_info = json.load(f)
    # resources
    num_stgs = opt_info["num_stgs"]
    total_mem = opt_info["total_mem"]
    hashes = opt_info["hashes"]
    # variable values chosen by search
    # int vars represent length of reg array (e.g., cols)
    # size vars represent number of reg arrays (e.g., rows)
    const_vars = opt_info["const_vars"]
    # variables we need to solve for
    ilp_vars = opt_info["ilp_vars"]
    # VARIABLES
    # we create vars for both the consts and ones we solve for
    # consts have a constraint that they equal their given value
    # variables for each reg_array, stg pair
    #   value should be num of reg arrays in each stg
    # variables for each reg_array_size, stg pair
    #   value should be mem used by single reg array in that stg (???)
    # NOTE: using intvar bc numvar is continuous var (we can't have floats)
    const_vars_sizes = {}
    const_vars_ints = {}
    vars_int_size_groups = {} # keep track of which sizes/ints go together
    stages_vars_sizes = [[]]*num_stgs
    stages_vars_ints = [[]]*num_stgs
    # const_vars is list of dicts, each dict contains group of vars (int and size)
    # NOTE: there HAS to be both int and size for each group
    # size is num reg arrays
    # int is reg array width
    for var_group in const_vars:
        size_name = var_group["size"]["name"]
        int_name = var_group["int"]["name"]
        size_val = var_group["size"]["value"]
        int_val = var_group["int"]["value"]
        vars_int_size_groups[int_name] = size_name
        
        const_vars_sizes[size_name] = []
        const_vars_ints[int_name] = []

        for n in range(num_stgs):
            # size vars
            # as in p4all ilp, this is an indicator var
            # we'll have separate vars for cols in reg array
            # (this makes writing constraints easier, at least for me)
            # name is var,stg_number
            const_vars_sizes[size_name].append(solver.addVar(vtype=GRB.BINARY, name=size_name+","+str(n)))
            stages_vars_sizes[n] = stages_vars_sizes[n] + [const_vars_sizes[size_name][-1]]

            # int vars
            # name is var, stg_number
            const_vars_ints[int_name].append(solver.addVar(lb=0,ub=int_val, vtype=GRB.INTEGER, name=int_name+","+str(n)))
            stages_vars_ints[n] = stages_vars_ints[n] + [const_vars_ints[int_name][-1]]
            # int constr
            # add constraint that reg array width = indicator var * value
            solver.addConstr(const_vars_ints[int_name][-1] == const_vars_sizes[size_name][-1]*int_val)

        # size constr
        # add constraint that total num of arrays = what we're given
        # aka sum of these vars == value
        solver.addConstr(quicksum(const_vars_sizes[size_name]) == size_val)
        #solver.Add(sum(const_vars_sizes[size_name]) == size_val)        

    # VARIABLES (not constant)
    ilp_vars_sizes = {}
    ilp_vars_ints = {}
    for var_group in ilp_vars:
        size_name = var_group["size"]["name"]
        int_name = var_group["int"]["name"]
        size_lb = var_group["size"]["lb"]
        int_lb = var_group["int"]["lb"]
        vars_int_size_groups[int_name] = size_name

        ilp_vars_sizes[size_name] = []
        ilp_vars_ints[int_name] = []

        for n in range(num_stgs):
            # size vars
            ilp_vars_sizes[size_name].append(solver.addVar(vtype=GRB.BINARY, name=size_name+","+str(n)))
            stages_vars_sizes[n] = stages_vars_sizes[n] + [ilp_vars_sizes[size_name][-1]]

            # int vars
            ilp_vars_ints[int_name].append(solver.addVar(lb=0, ub=total_mem, vtype=GRB.INTEGER, name=int_name+","+str(n)))
            stages_vars_ints[n] = stages_vars_ints[n] + [ilp_vars_ints[int_name][-1]]
            # add constr that arrays not placed have width=0 
            solver.addConstr(ilp_vars_ints[int_name][-1] <= ilp_vars_sizes[size_name][-1]*total_mem)
            # add constr that placed arrays have width > 0
            solver.addConstr(ilp_vars_ints[int_name][-1] >= ilp_vars_sizes[size_name][-1])

        # add constr that we at least reach lower bound for size var
        solver.addConstr(quicksum(ilp_vars_sizes[size_name]) >= size_lb)

        # add constr so that regs have the same size
        # here's how we do this:
        #   we assume that we have to place AT LEAST one reg array
        #   width of an array * indicator of first array == width of first array * indicator of array
        # if we don't place array, it's 0; otherwise it equals width of first array
        for var in range(len(ilp_vars_ints[int_name])):
            #ind = ilp_vars_ints[int_name].index(var)
            if var == 0:
                continue
            solver.addConstr(ilp_vars_ints[int_name][var]*ilp_vars_sizes[size_name][0] == ilp_vars_ints[int_name][0]*ilp_vars_sizes[size_name][var])

    # CONSTRAINTS
    # num hashes per stage < hashes
    # NOTE: assume that each reg array uses exactly 1 hash unit
    for stg in stages_vars_sizes:
        solver.addConstr(quicksum(stg) <= hashes)

    # amount of mem per stage < total_mem
    for stg in stages_vars_ints:
        solver.addConstr(quicksum(stg) <= total_mem)

    # simple test objective function
    # TODO: specific to starflow
    solver.setObjective(quicksum(ilp_vars_ints["L_SLOTS"]), GRB.MAXIMIZE)
    solver.optimize()


    # NOTE: for each int var, we're rounding down to nearest power of 2
    print('Solution:')
    print('Objective value =', solver.ObjVal)   # this is obj BEFORE rounding down to power of 2
    for s in const_vars_sizes:
        for s_stg in const_vars_sizes[s]:
            print(s_stg.varName, '=', s_stg.X)

    for i in const_vars_ints:
        for i_stg in const_vars_ints[i]:
            print(i_stg.varName, '=', highestPowerof2(int(i_stg.X)))

    for s in ilp_vars_sizes:
        for s_stg in ilp_vars_sizes[s]:
            print(s_stg.varName, '=', s_stg.X)

    for i in ilp_vars_ints:
        for i_stg in ilp_vars_ints[i]:
            print(i_stg.varName, '=', highestPowerof2(int(i_stg.X)))


def main():
    solve()


if __name__ == '__main__':
    main()





