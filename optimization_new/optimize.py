import time, importlib, argparse, os, sys
from optalgos import *
from interp_sim import update_sym_sizes, write_symb

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
    parser.add_argument("--nopruning", help="don't do pruning phase of ordered search", action="store_true")
    parser.add_argument("--fullcompile", help="use lucid-p4 compiler instead of layout script", action="store_true")
    parser.add_argument("--exhaustive", help="test every solution that compiles w interpreter", action="store_true")
    parser.add_argument("--pair", help="hacky solution to identify when we have pair arrays", action="store_true")
    parser.add_argument("--preprocessingonly", help="only do preprocessing, store sols in preprocessed.pkl", action="store_true")
    parser.add_argument("--shortcut", help="don't do preprocessing, load already preprocessed sols from preprocessed.pkl", action="store_true")
    args = parser.parse_args()

    '''
    if len(sys.argv) < 2:
        print("usage: python3 optimize.py <json opt info file>")
        quit()
    '''
    #opt_info = json.load(open(sys.argv[1]))
    #print(opt_info)

    # get current working directory
    cwd = os.getcwd()

    # initialize everything we need to run opt algo
    opt_info,symbolics_opt, o = init_opt(args.optfile, args.notrafficgen, cwd)

    # optimize!
    start_time = time.time()
    #basin_hopping(symbolics_opt, opt_info,o)
    #quit()
    # TODO: allow user to pass in func
    if "optalgofile" in opt_info["optparams"]:   # only include field in json if using own algo
        # import module, require function to have standard name and arguments
        user = True


    elif opt_info["optparams"]["optalgo"] == "random":    # if not using own, should for sure have optalgo field
        best_sol, best_cost = random_opt(symbolics_opt, opt_info, o, args.timetest)

    elif opt_info["optparams"]["optalgo"] == "simannealing":
        best_sol, best_cost = simulated_annealing(symbolics_opt, opt_info, o, args.timetest)

    elif opt_info["optparams"]["optalgo"] == "exhaustive":
        best_sol, best_cost = exhaustive(symbolics_opt, opt_info, o, args.timetest)

    # testing out ordered search
    elif opt_info["optparams"]["optalgo"] == "ordered":
        best_costs = []
        best_sols = []
        time_costs = []
        num_sols_time = []
        starting_sols = []
        #best_sol, best_cost = ordered(symbolics_opt, opt_info, o, args.timetest, args.nopruning, args.fullcompile, args.exhaustive, args.pair, args.preprocessingonly, args.shortcut)
        for i in range(1): # repeat once times
            best_sol, best_cost, time_cost, num_sol, starting = ordered(symbolics_opt, opt_info, o, args.timetest, args.nopruning, args.fullcompile, args.exhaustive, args.pair, args.preprocessingonly, args.shortcut)
            best_costs.append(best_cost)
            best_sols.append(best_sol)
            time_costs.append(time_cost)
            num_sols_time.append(num_sol)
            starting_sols.append(starting)

        results = {}
        results["best_costs"] = best_costs
        results["best_sols"] = best_sols
        results["time_costs"] = time_costs
        results["num_sols_time"] = num_sols_time
        results["starting_sols"] = starting_sols
        with open("1iter_results.pkl",'wb') as f:
            pickle.dump(results, f)

    elif opt_info["optparams"]["optalgo"] == "neldermead":
        best_sol, best_cost = nelder_mead(symbolics_opt, opt_info, o, args.timetest)


    end_time = time.time()
    # write symb with final sol
    update_sym_sizes(best_sol, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
    write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], opt_info["symbolicvals"]["logs"], opt_info["symfile"])


    '''
    # try compiling to tofino?
    # we could test the top x solutions to see if they compile --> if they do, we're done!
    # else, we can repeat optimization, excluding solutions we now know don't compile
    # (we have to have a harness p4 file for this step, but not for interpreter)
    # NOTE: use vagrant vm to compile
    for sol in top_sols:
        write_symb(sol[0],sol[1])
        # compile lucid to p4
        cmd_lp4 = ["../../dptc cms_sym.dpt ip_harness.p4 linker_config.json cms_sym_build --symb cms_sym.symb"]
        ret_lp4 = subprocess.run(cmd_lp4, shell=True)
        # we shouldn't have an issue compiling to p4, but check anyways
        if ret_lp4.returncode != 0:
            print("error compiling lucid code to p4")
            break
        # compile p4 to tofino
        cmd_tof = ["cd cms_sym_build; make build"]
        ret_tof = subprocess.run(cmd_tof, shell=True)
        # return value of make build will always be 0, even if it fails to compile
        # how can we check if it compiles????

        # if compiles, break bc we've found a soluion
    '''


    print("BEST:")
    print(best_sol)
    print("BEST COST:")
    print(best_cost)
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
    structchoice: (tells us if we're choosing between structs)
        var: which of the symbolic vars corresponds to struct choice (boolean)
        True: if var==True, list any symbolic vars the corresponding struct doesn't use
        False: if var==False, ^
    optparams: (any info related to optimization algo)
        optalgo: if using one of our provided functions, tell us the name (random, simannealing)
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
