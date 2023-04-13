import time, importlib, argparse, os, sys, json
from preprocess import *
from optalgos import *
from interp_sim import update_sym_sizes, write_symb

# grab optimization parameters, generate traffic trace if necessary, create dict of symbolic values to optimize
def init_opt(optfile, notraffic, cwd):
    sys.path.append(cwd)
    # NOTE: assuming optfile is in current working directory
    # import json file
    opt_info = json.load(open(optfile))

    # import opt class that has funcs we need to get traffic, cost
    # NOTE: module has to be in current working directory
    optmod = importlib.import_module(opt_info["optmodule"])
    o = optmod.Opt(opt_info["trafficpcap"])

    # gen traffic
    if not notraffic:
        o.gen_traffic()

    # it's easier for gen next values if we exclude logs and don't separate symbolics/sizes, so let's do that here
    symbolics_opt = {}

    # is there a better way to merge? quick solution for now
    for var in opt_info["symbolicvals"]["sizes"]:
        if var not in opt_info["symbolicvals"]["logs"]:
            symbolics_opt[var] = opt_info["symbolicvals"]["sizes"][var]
    for var in opt_info["symbolicvals"]["symbolics"]:
        if var not in opt_info["symbolicvals"]["logs"]:
            symbolics_opt[var] = opt_info["symbolicvals"]["symbolics"][var]

    return opt_info,symbolics_opt, o

# usage: python3 optimize.py <json opt info file>
def main():
    parser = argparse.ArgumentParser(description="optimization of lucid symbolics in python, default uses layout script instead of compiler")
    parser.add_argument("optfile", metavar="optfile", help="name of json file with optimization info")
    parser.add_argument("--timetest", help="time test, output results at benchmark times", action="store_true")
    parser.add_argument("--notrafficgen", help="don't call gen_traffic, this is just for testing", action="store_true")
    parser.add_argument("--fullcompile", help="use lucid-p4 compiler instead of layout script", action="store_true")
    parser.add_argument("--pair", help="hacky solution to identify when we have pair arrays", action="store_true")
    parser.add_argument("--preprocessingonly", help="only do preprocessing, store sols in preprocessed.pkl", action="store_true")
    parser.add_argument("--shortcut", help="don't do preprocessing, load already preprocessed sols from preprocessed.pkl", action="store_true")
    parser.add_argument("--dfg", help="use dataflow analysis instead of layout in preprocessing", action="store_true")
    args = parser.parse_args()

    # get current working directory
    cwd = os.getcwd()

    # initialize everything we need to run opt algo
    opt_info,symbolics_opt, o = init_opt(args.optfile, args.notrafficgen, cwd)

    bounds_tree = None
    solutions = None

    # check if we're doing preprocessing
    # if we're not, then we rely on user-provided bounds to define our search space
    if "optalgo" in opt_info["optparams"] and opt_info["optparams"]["optalgo"] == "preprocess":
        sols = preprocess(symbolics_opt, opt_info, o, args.timetest, args.fullcompile, args.pair, args.shortcut, args.dfg)
        bounds_tree = sols["tree"]
        solutions = sols["all_sols"]

        # TODO: save preprocessed sols every time???
        if args.preprocessingonly:  # only preprocessing, no optimization (save bounds tree and quit)
            if args.dfg: # we used dataflow graph heuristic instead of layout
                with open('preprocessed_dfg.pkl','wb') as f:
                    pickle.dump(sols, f)
                return
            with open('preprocessed.pkl','wb') as f:
                pickle.dump(sols, f)
            return


    # optimize!
    start_time = time.time()
    # TODO: allow user to pass in func
    # built-in strategies are: simannealing, bayesian, neldermead simplex, exhaustive
    if "optalgofile" in opt_info["optparams"]:   # only include field in json if using own algo
        # import module, require function to have standard name and arguments
        user = True

    elif opt_info["optparams"]["strategy"] == "simannealing":
        best_sols, best_cost, all_evaled, all_evaled_sols_sorted = simulated_annealing(symbolics_opt, opt_info, o, args.timetest, solutions, bounds_tree)

    elif opt_info["optparams"]["strategy"] == "exhaustive":
        best_sols, best_cost, all_evaled, all_evaled_sols_sorted = exhaustive(symbolics_opt, opt_info, o, args.timetest, solutions, bounds_tree)

    elif opt_info["optparams"]["strategy"] == "bayesian":
        best_sols, best_cost, all_evaled, all_evaled_sols_sorted = bayesian(symbolics_opt, opt_info, o, args.timetest, solutions, bounds_tree)

    elif opt_info["optparams"]["strategy"] == "neldermead":
        best_sols, best_cost, all_evaled, all_evaled_sols_sorted = nelder_mead(symbolics_opt, opt_info, o, args.timetest, solutions=solutions, tree=bounds_tree)


    end_time = time.time()
    # write symb with final sol
    update_sym_sizes(best_sols[0], opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
    write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], opt_info["symbolicvals"]["logs"], opt_info["symfile"], opt_info)



    # try compiling best sol to tofino
    # 1 - check if we even have bf sde installed (check env vars)
    if not os.environ.get('SDE') or not os.environ.get('SDE_INSTALL'):
        print("cannot compile to tofino, environment variables (SDE, SDE_INSTALL) are not set")
        print("returning best solution as determined by lucid compiler and optimization")
        print("BEST:")
        print(best_sols[0])
        print("BEST COST:")
        print(best_cost)
        print("TIME(s):")
        print(end_time-start_time)
        return

    best_compiling_sol = None
    best_compiling_eval = None
    for sol in all_evaled_sols_sorted:
        # gen symbolic file
        update_sym_sizes(sol, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
        write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], opt_info["symbolicvals"]["logs"], opt_info["symfile"], opt_info)
        # 2 - compile best sol w/ lucid compiler
        print("COMPILING LUCID TO P4")
        cmd = ["make", "compile"]
        ret = subprocess.run(cmd)
        if ret.returncode != 0: # stop if there's an error running compiler
            exit("lucid compiler error")
        # 3 - compile to tofino
        print("COMPILING TO TOFINO")
        cmd = ["make", "assemble"]
        ret = subprocess.run(cmd)
        if ret.returncode != 0: # stop if there's an error running compiler
            exit("tofino compiler error")
        # 4 - check for manifest.json file (compiled field)
        tof_file = os.getcwd()+"/build/lucid/manifest.json"
        try:
            manifest = json.load(open(tof_file,'r'))
        except:
            exit("tofino compilation did not generate manifest.json. in build/makefile, make sure build command uses 'build', NOT 'build_quiet'. otherwise, this is probably a compiler error")
        # 5 - if compiles, return
        if manifest["compilation_succeeded"]:
            print("tofino compilation successful!")
            best_compiling_sol = sol
            best_compiling_eval = all_evaled[sol]
        #   if not, then try next best sol? (check return of optalgos) 
        else:
            print("failed to compile to tofino, trying next solution")
            continue


    if not best_compiling_sol:
        print("no evaluated configurations compiled to tofino. this is likely the result of a PHV allocation issue that must be fixed manually.")
    else:
        print("BEST:")
        print(best_compiling_sol)
        print("BEST COST:")
        print(best_compiling_eval)


    print("TIME(s):")
    print(end_time-start_time)



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
        order_resource: list of symbolics that affect resources
        non_resource: list of symbolics that don't affect resources
        optalgo: "preprocess" if doing preprocessing, exclude if not doing preprocessing
        strategy: if using one of our provided functions, tell us the name (simannealing, bayesian, neldermead, exhaustive)
        optalgofile: if using your own, tell us where to find it (python file)
        stop_iter: num iterations to stop at
        stop_time: time to stop at (in seconds)
        temp: initial temp for simannealing (init temps are almost arbitrary??)
        stepsize: stddev for simannealing (per symbolic), only include if NOT preprocessing
        maxcost: cost to return if solution uses too many stages, only include if NOT preprocessing
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
