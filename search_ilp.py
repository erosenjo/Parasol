from ortools.linear_solver import pywraplp
import json

# adapt for parasol
# used for (hopefully) cutting down search space
# pick some (resource-related) variables, use ilp to pick other (resource-related) vars
# we always want to use as much mem as possible, so obj is max mem (or other limiting resource)
# TODO:
#   deps?
#   how to know if uses hash func? --> assume every reg array uses hash func
#   assume 1 reg array per stg?
# changes from original:
#   remove phv/metadata constraint
#   remove tcam constraints
#   ignoring stateful alu constraint (hashes more limited?)

# starflow:
#   vars: s_slots (cols), l_slots (cols), num_long (rows), max_short_idx (rows - 1)
#   pick s_slots and max_short_idx, then use ilp to get num_long and l_slots?
#   have lower bound for num_long/l_slots, if sol < lb, throw it out
#   obj: max num_long, l_slots


# using glop until i can get gurobi license installed
solver = pywraplp.Solver.CreateSolver('GLOP')

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
        const_vars_sizes[size_name].append(solver.IntVar(0,1, size_name+","+str(n)))

        # int vars
        # name is var, stg_number
        const_vars_ints[int_name].append(solver.IntVar(0,int_val, int_name+","+str(n)))
        # int constr
        # add constraint that reg array width = indicator var * value
        solver.Add(const_vars_ints[int_name][-1] == const_vars_sizes[size_name][-1]*int_val)

    # size constr
    # add constraint that total num of arrays = what we're given
    # aka sum of these vars == value
    solver.Add(sum(const_vars_sizes[size_name]) == size_val)        

# VARIABLES (not constant)




# CONSTRAINTS
# TODO: deps?????????
# num hashes per stage < hashes

# amount of mem per stage < total_mem


# simple test objective function
# TODO: specific to starflow
solver.Maximize(sum(const_vars_ints["S_SLOTS"]))
solver.Solve()

print('Solution:')
print('Objective value =', solver.Objective().Value())
for s in const_vars_sizes:
    for s_stg in const_vars_sizes[s]:
        print(s_stg, '=', s_stg.solution_value())

for i in const_vars_ints:
    for i_stg in const_vars_ints[i]:
        print(i_stg, '=', i_stg.solution_value())






