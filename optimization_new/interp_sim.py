import subprocess, json, math, pickle, time, os, re

# this function runs interpreter with whatever symb file is in directory and returns measurement of interest
def interp_sim(lucidfile,outfiles,tracefile=None):
    # run the interpreter
    if not tracefile:   # single trace file, that should have the same name as dpt file
        cmd = ["make", "interp"]
    else:               # mult trace files, name does NOT have to be same as dpt file
        cmd = ["/media/data/mh43/lucid/dpt", "--suppress-final-state", lucidfile, "--spec", tracefile]

    #with open('output.txt','w') as outfile:
    #    ret = subprocess.run(cmd, stdout=outfile, shell=True)
    ret = subprocess.run(cmd)
    if ret.returncode != 0: # stop if there's an error running interp
        exit("error running interpreter")


    # get output from interpreter
    # we might have multiple files, so we loop through and store measurements from all of them
    measurement = []
    for out in outfiles:
        measurement.append(pickle.load(open(out,"rb")))

    return measurement


def write_symb(sizes, symbolics, logs, symfile, opt_info):
    # we often have symbolics that should = log2(some other symbolic)
    # in that case, we compute it here
    for var in logs:
        if logs[var] in sizes:
            log = int(math.log2(sizes[logs[var]]))
        else:
            log = int(math.log2(symbolics[logs[var]]))
        if var in sizes:
            sizes[var] = log
        else:
            symbolics[var] = log
    if "rules" in opt_info["symbolicvals"]:
        for rulevar in opt_info["symbolicvals"]["rules"]:
            rule = opt_info["symbolicvals"]["rules"][rulevar].split()
            for v in range(len(rule)):
                if rule[v] in opt_info["symbolicvals"]["symbolics"]:
                    rule[v] = str(symbolics[rule[v]])
                elif rule[v] in opt_info["symbolicvals"]["sizes"]:
                    rule[v] = str(sizes[rule[v]])
            if rulevar in sizes:
                sizes[rulevar] = eval(''.join(rule))
            else:
                symbolics[rulevar] = eval(''.join(rule))

 
    concretes = {}
    concretes["sizes"] = sizes
    concretes["symbolics"] = symbolics
    with open(symfile, 'w') as f:
        json.dump(concretes, f, indent=4)
    print("SYMB FILE SIZES", sizes)
    print("SYMB FILE SYMBOLICS", symbolics)

def update_sym_sizes(symbolics_opt, sizes, symbolics):
    for var in symbolics_opt:
        if var in sizes:
            sizes[var] = symbolics_opt[var]
            continue
        if var in symbolics:
            symbolics[var] = symbolics_opt[var]

    return sizes, symbolics


# full lucid -> p4 compiler
def compile_num_stages(symbolics_opt, opt_info):
    # gen symbolic file so we can compile with new values
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
    write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"], opt_info)
    # NEW LUCID COMPILATION
    #cmd = ["../../lucid/dptc", opt_info["lucidfile"], "build", "--symb", opt_info["symfile", "--silent"]
    cmd = ["make", "compile"]
    ret = subprocess.run(cmd)
    if ret.returncode != 0: # stop if there's an error running compiler
        exit("compiler error")
    num_stg = 0
    # NOTE(!!!!!!): makefile compile command MUST call folder build, otherwise this will fail
    stg_file = os.getcwd()+"/build/num_stages.txt"
    with open(stg_file) as f:
        num_stg = int(f.readline())
    return num_stg


# partial compiler, layout script
def layout(symbolics_opt, opt_info):
    # gen symbolic file so we can compile with new values
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
    write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"], opt_info)
    # NEW DATA FLOW COMPILE AND LAYOUT
    cmd = ["make", "layout"]
    ret = subprocess.run(cmd, capture_output=True)
    #ret = subprocess.check_output(cmd)
    res = re.split('LAYOUTSTAGES', str(ret.stdout))
    resources = json.loads(res[1])
    if ret.returncode != 0: # stop if there's an error  
        exit("layout error")
    # NOTE: assuming that we're calling layout and opt from same working directory 
    #resfile = os.getcwd()+"/resources.json"
    #resources = json.load(open(resfile,'r'))
    print(resources)
    return resources

# partial compiler, data flow graph
def dfg(symbolics_opt, opt_info):
    # gen symbolic file so we can compile with new values
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
    write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"], opt_info)
    # NEW DATA FLOW COMPILE AND LAYOUT
    cmd = ["make", "dfg"]
    ret = subprocess.run(cmd)
    if ret.returncode != 0: # stop if there's an error  
        exit("dfg error")
    # NOTE: assuming that we're calling layout and opt from same working directory 
    resfile = os.getcwd()+"/resources_dfg.json"
    resources = json.load(open(resfile,'r'))
    return resources


def gen_cost(symbolics_opt_vars,syms_opt, opt_info, o, scipyalgo, searchtype):
    # if scipyalgo is true, then symolics_opt is np array, not dict
    print("VARS:", symbolics_opt_vars)
    if scipyalgo:
        symbolics_opt = {}
        sym_keys = list(syms_opt.keys())
        for v in range(len(symbolics_opt_vars)):
            symbolics_opt[sym_keys[v]] = int(symbolics_opt_vars[v])
    else:
        symbolics_opt = symbolics_opt_vars


    # if this is trace version, we only need to write symbolic once at the beginning
    if searchtype != "trace":
        # generate symbolic file
        update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
        write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"], opt_info)

    '''
    # moving generation of symbolic file to compile_num_stages function
    # gen symbolic file
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
    write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"])
    '''

    # compile to p4 and check if stgs <= tofino
    # if we didn't preprocess, compile to check stgs first
    # if this is the trace version, assume that we've already verified it compiles
    if searchtype != "preprocessed" and searchtype != "trace":
        res = layout(symbolics_opt, opt_info)
        if res["stages"] > 12:  # we won't fit on the switch
            return opt_info["optparams"]["maxcost"]
    
    # call init_iteration for opt class
    o.init_iteration(symbolics_opt)

    # run interp!
    m = interp_sim(opt_info["lucidfile"],opt_info["outputfiles"])

    # pass measurement(s) to cost func, get cost of sol
    cost = o.calc_cost(m)

    return cost


def gen_cost_multitrace(symbolics_opt_vars,syms_opt, opt_info, o, scipyalgo, searchtype):
    symbolics_opt = symbolics_opt_vars
    # generate symbolic file
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"]) # python passes dicts as reference, so this is fine
    write_symb(opt_info["symbolicvals"]["sizes"],opt_info["symbolicvals"]["symbolics"],opt_info["symbolicvals"]["logs"],opt_info["symfile"], opt_info)

    # compile to p4 and check if stgs <= tofino
    # if we didn't preprocess, compile to check stgs first
    if searchtype != "preprocessed":
        res = layout(symbolics_opt, opt_info)
        if res["stages"] > 12:  # we won't fit on the switch
            return opt_info["optparams"]["maxcost"]


    # we have multiple json traces to simulate
    # each trace should dump output to a different file
    # run interp on each trace, THEN calc the cost

    # TODO: NEED TO ADD ARG TO INTERP COMMAND?? how to pass in trace name to make command?
    m = []
    for trace in opt_info["interp_traces"]:
        # call init_iteration for opt class
        o.init_iteration(symbolics_opt)       

        m += interp_sim(opt_info["lucidfile"],opt_info["outputfiles"],trace)

    # pass measurement(s) to cost func, get cost of sol
    cost = o.calc_cost(m)
 
    return cost


