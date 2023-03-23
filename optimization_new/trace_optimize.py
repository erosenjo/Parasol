import time, importlib, argparse, os, sys, json
from preprocess import *
from optalgos import *
from interp_sim import update_sym_sizes, write_symb

def init_opt_trace(optfile, cwd):
    sys.path.append(cwd)
    # NOTE: assuming optfile is in current working directory
    # import json file
    opt_info = json.load(open(optfile))

    # import opt class that has funcs we need to get traffic, cost
    # NOTE: module has to be in current working directory
    optmod = importlib.import_module(opt_info["optmodule"])
    o = optmod.Opt()

    symbolics_opt = {}
    # is there a better way to merge? quick solution for now
    for var in opt_info["symbolicvals"]["sizes"]:
        symbolics_opt[var] = opt_info["symbolicvals"]["sizes"][var]
    for var in opt_info["symbolicvals"]["symbolics"]:
        symbolics_opt[var] = opt_info["symbolicvals"]["symbolics"][var]

    trace_params = opt_info["traceparams"]
    trace_bounds = opt_info["tracebounds"]

    return opt_info,symbolics_opt, o, trace_params, trace_bounds

# usage: python3 optimize.py <json opt info file>
def main():
    parser = argparse.ArgumentParser(description="optimization of lucid symbolics in python, default uses layout script instead of compiler")
    parser.add_argument("optfile", metavar="optfile", help="name of json file with optimization info")
    args = parser.parse_args()

    #opt_info = json.load(open(sys.argv[1]))
    #print(opt_info)

    # get current working directory
    cwd = os.getcwd()

    '''
    OPTIMIZE TRACE: keep symbolic values/struct configuration the same; change attributes of the input trace each iteration
    '''
    opt_info, symbolics_opt, o, trace_params, trace_bounds = init_opt_trace(args.optfile, cwd)
    # write symbolic file w/ vals given in json
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
    write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], [], opt_info["symfile"], opt_info)

    # TODO: put this in a loop/function outside of optimize.py (e.g., optalgos.py)
    if opt_info["optparams"]["strategy"] == "random":
        # generate trace
        o.gen_traffic(trace_params)
        # get cost
        cost = gen_cost(symbolics_opt,symbolics_opt,opt_info, o,False, "trace")
        print(cost)
    else:
        exit("input strategy is not implemented for trace version of parasol")



if __name__ == "__main__":
    main()



'''
json fields:
    symbolicvals: (anything info related to symbolics)
        sizes: symbolic sizes and starting vals
        symbolics: symbolic vals (ints, bools) and starting vals
        logs: which (if any) symbolics are log2(another symbolic)
        bounds: [lower,upper] bounds for symbolics, inclusive (don't need to include logs, bc they're calculated from other syms)
    optparams: (any info related to optimization algo)
        optalgo: if using one of our provided functions, tell us the name (simannealing, bayesian, neldermead)
        optalgofile: if using your own, tell us where to find it (python file)
        stop_iter: num iterations to stop at
        stop_time: time to stop at (in seconds)
        temp: initial temp for simannealing (init temps are almost arbitrary??)
        stepsize: stddev for simannealing (per symbolic? or single?
        maxcost: cost to return if solution uses too many stages
    symfile: file to write symbolics to
    lucidfile: dpt file
    outputfiles: list of files that output is stored in (written to by externs)
    optmodule: name of module that has class w/ necessary funcs
    trafficpcap: name of pcap file to use

sys reqs:
    python3
    lucid
    numpy

'''
