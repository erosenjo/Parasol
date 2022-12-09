import pickle, json, subprocess, sys
import os
from treelib import Node, Tree
from interp_sim import compile_num_stages
from importlib import reload

sols = pickle.load(open('preprocessed.pkl','rb'))
opt_info = json.load(open(sys.argv[1],'r'))

bounds_tree = sols["tree"]
solutions = sols["all_sols"]

lucid_sols = []
tofino_sols = []


# set non resource vars
non_resource = {}
if opt_info["optparams"]["non_resource"]:
    for nr in opt_info["optparams"]["non_resource"]:
        non_resource[nr] = 128

for sol_choice in solutions:
    resource_syms = {}
    for sol in sol_choice:
        node = bounds_tree.get_node(sol)
        if node.tag=="root":
            continue
        resource_syms[node.tag[0]] = node.tag[1]

    # combine symbolics opt and non resource
    symbolics_opt = {**resource_syms, **non_resource}
    # compile lucid to p4
    stgs_used = compile_num_stages(symbolics_opt, opt_info)
    if stgs_used <= 12:
        lucid_sols.append(symbolics_opt)
    # compile to tofino
    cmd = ["make", "assemble"]
    ret = subprocess.run(cmd)
    if ret.returncode != 0: # stop if there's an error running compiler
        exit("tofino compile error")
    # check manifest.json
    tof_file = os.getcwd()+"/build/lucid/manifest.json"
    manifest = json.load(open(tof_file,'r'))
    if manifest["compilation_succeeded"]:
        tofino_sols.append(symbolics_opt)


print("LUCID SOLS:", len(lucid_sols))
print("TOFINO SOLS:", len(tofino_sols))



