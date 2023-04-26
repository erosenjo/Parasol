'''
script to run lucid interpreter in interactive mode and periodically feed it traces

to use with RL agent: (see main function as example)
    - call init_simulation(optfile) to start up simulation, where optfile is json with sketch params
    - call send_pkts(numpkts, opt_info, o, sim_process) for each iterval to get bits set and error after the interval
    - call end_simulation(sim_process) to kill interpreter
'''

import time, importlib, argparse, os, sys, json
import subprocess
import pickle

def update_sym_sizes(symbolics_opt, sizes, symbolics):
    for var in symbolics_opt:
        if var in sizes:
            sizes[var] = symbolics_opt[var]
            continue
        if var in symbolics:
            symbolics[var] = symbolics_opt[var]

    return sizes, symbolics

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


def init_opt_trace(optfile, cwd):
    sys.path.append(cwd)
    # NOTE: assuming optfile is in current working directory
    # import json file
    opt_info = json.load(open(optfile))

    # get config of symbolics
    symbolics_opt = {}
    # is there a better way to merge? quick solution for now
    for var in opt_info["symbolicvals"]["sizes"]:
        symbolics_opt[var] = opt_info["symbolicvals"]["sizes"][var]
    for var in opt_info["symbolicvals"]["symbolics"]:
        symbolics_opt[var] = opt_info["symbolicvals"]["symbolics"][var]

    trace_params = opt_info["traceparams"]
    trace_bounds = opt_info["tracebounds"]


    # import opt class that has funcs we need to get traffic, cost
    # NOTE: module has to be in current working directory
    optmod = importlib.import_module(opt_info["optmodule"])
    o = optmod.Opt(symbolics_opt)

    o.init_simulation(opt_info["optparams"]["maxpkts"])

    trace_params = opt_info["traceparams"]
    trace_bounds = opt_info["tracebounds"]

    return opt_info,symbolics_opt, o, trace_params, trace_bounds


# this function starts a simulation in interactive mode
# it processes the events in the trace json file (same name as dpt file)
# and returns the process (so we can pass in more events later)
# beginning trace file can just be empty or it can contain packets
def start_interactive_simulation():
    # start with single trace file, that should have the same name as dpt file
    cmd = ["make", "interactive"]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    return process

# sends the next set of packets
# returns the corresponding measurement taken
def send_next_events(process, events, outfiles):
    '''
    for e in events:
        process.stdin.write(e)
        process.stdin.flush()
        # we need some pause in between writing events (until lucid allows us to pass in list of events)
        time.sleep(0.0001)
    '''
    process.stdin.write(events)
    process.stdin.flush()
    # wait until file is in directory
    # delete the file after we grab the measurement
    while not os.path.isfile(outfiles[0]):
        time.sleep(1)
    measurement = []
    for out in outfiles:
        measurement.append(pickle.load(open(out,"rb")))
        os.remove(out)

    return measurement


# interactive interpreter will continuously wait for more input,
# so we need to kill it once we're done
def end_simulation(process):
    stdout, stderr = process.communicate()
    return


# start simulation of rotating cms, given params specified in optfile
#   optfile is the json file with params for layout of sketch
# this returns 
#   opt_info (optfile contents)
#   o (object that we use to generate traces and calc cost of an iteration)
#   sim_process (object for interpreter process)
def init_simulation(optfile):
    # get current working directory
    cwd = os.getcwd()

    opt_info, symbolics_opt, o, trace_params, trace_bounds = init_opt_trace(optfile, cwd)
    # write symbolic file w/ vals given in json
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
    write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], [], opt_info["symfile"], opt_info)

    sim_process = start_interactive_simulation()
    # add a bit of a buffer period before we starting sending pkts
    # TODO: remove this?
    time.sleep(1)

    return opt_info, o, sim_process

# this represents one interval of a rotating cms application
# takes as input
#   number of pkts to send
#   opt_info (json file content)
#   o (same object returned from init_simulation)
#   sim_process (same object returned from init_simulation
def send_pkts(numpkts, opt_info, o, sim_process):
    # generate lucid events
    opt_info["traceparams"]["numpkts"] = numpkts 
    events = o.gen_traffic(opt_info["traceparams"])

    # interpret trace, get measurements
    measurement = send_next_events(sim_process, events, opt_info["outputfiles"])

    # calc cost
    cost = o.calc_cost(measurement)

    # return bits set for each sketch and avg error for all flows in the iteration
    # NOTE: this returns avg error, but we can also return gt and estimated counts for each flow instead
    return measurement[1], cost

# usage: python3 optimize.py <json opt info file>
def main():
    parser = argparse.ArgumentParser(description="optimization of lucid symbolics in python, default uses layout script instead of compiler")
    parser.add_argument("optfile", metavar="optfile", help="name of json file with optimization info")
    args = parser.parse_args()

    #opt_info = json.load(open(sys.argv[1]))
    #print(opt_info)


    # start the interpreter
    opt_info, o, sim_process = init_simulation(args.optfile)
    
    counter = 0
    while True:
        # generate a trace w/ 260 pkts
        bits_set, avg_err = send_pkts(260, opt_info, o, sim_process)
        print(bits_set, avg_err)
        counter += 1
        if counter >= 5:
            break

    end_simulation(sim_process)

    '''
    # get current working directory
    cwd = os.getcwd()

    opt_info, symbolics_opt, o, trace_params, trace_bounds = init_opt_trace(args.optfile, cwd)
    # write symbolic file w/ vals given in json
    update_sym_sizes(symbolics_opt, opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"])
    write_symb(opt_info["symbolicvals"]["sizes"], opt_info["symbolicvals"]["symbolics"], [], opt_info["symfile"], opt_info)

    cost = 0
    sim_process = start_interactive_simulation()
    time.sleep(5)
    counter = 0
    while True:
        # generate trace
        events = o.gen_traffic(trace_params)
        measurement = send_next_events(sim_process, events, opt_info["outputfiles"])
        cost = o.calc_cost(measurement)
        print(cost)
        # SEND COST/REWARD BACK TO RL AGENT, GEN NEXT SET OF PACKETS
        # rl agent should adjust something in trace_params to adjust trace

        counter += 1
        if counter >= 5:
            break
    end_simulation(sim_process)
    '''


if __name__ == "__main__":
    main()



'''
TODO: update this

json fields:
    symbolicvals: (anything info related to symbolics)
        sizes: symbolic sizes and starting vals
        symbolics: symbolic vals (ints, bools) and starting vals
        logs: which (if any) symbolics are log2(another symbolic)
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
