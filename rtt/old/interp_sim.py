import subprocess, os, ast, json, math, pickle, time
from move import *
from traffic import *


# this function runs interpreter with whatever symb file is in directory and returns measurement of interest
def interp_sim(lucidfile,outfile):
    # run the interpreter
    cmd = ["../../dpt", "--suppress-final-state", lucidfile]

    #with open('output.txt','w') as outfile:
    #    ret = subprocess.run(cmd, stdout=outfile, shell=True)
    ret = subprocess.run(cmd)
    if ret.returncode != 0: # stop if there's an error running interp
        print("err")
        quit()


    # get output from interpreter
    # we only have one line (final measurement), but can put this in a loop and read mult lines if we want
    #measurement = pickle.load(open(outfile,"rb"))
    measurement = 0
    with open(outfile,'r') as f:
        measurement = int(f.readline())

    '''
    # OLD
    # get the measurement output by interpreter (reg array)
    measurement = []
    lastline = ""
    pipeline = False
    with open('output.txt','r') as datafile:
        # skip until we get to the Pipeline line
        # for cms example, we always want the last reg in the pipeline
        lines = datafile.readlines()
        for line in lines:
            if "Pipeline" in line:
                pipeline = True
                lastline = line
                continue
            if pipeline:
                if line.strip()=="]":   # end of reg output, so the last line has the measurement reg
                    measurement=ast.literal_eval(lastline.split(':')[1].replace(';',',').replace("u32","").strip())
                    break
                lastline=line

    '''
    return measurement


# compute eviction ratio (ground_truth = num pkts)
def calc_cost(evicts, ground_truth):
    return evicts/ground_truth


def write_symb(tables, table_size, thresh):
    log_tables = int(math.log2(table_size))

    concretes = {}
    concretes["sizes"] = {"tables":tables, "log_tables":log_tables}
    concretes["symbolics"] = {"table_size":table_size, "TS_EXPIRE_THRESHOLD":thresh}
    with open('rtt.symb', 'w') as f:
        json.dump(concretes, f, indent=4)

# create init symb file
tables = 1
#table_size=128
table_size = 65536
thresh = 50000000
#thresh = 50000
write_symb(tables, table_size, thresh)

#write_symb(sizes,symbolics)

# compute ground truth
ground_truth = gen_traffic("univ1_pt1.pcap")
#ground_truth = gen_traffic("equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")

# we keep track of best solution(s) and cost(s)
best_sol = [tables, table_size, thresh]
best_cost = 0
iterations = 0
# we need both of these for simulated annealing
# (not necessary to use sim annealing, can use any optimization)
# new represents the concretes/measurements from most recent test
# curr is used to choose next solution, we set curr = new w/ some probability based on the costs
# (^ specific to simulated annealing)
curr_cost = 0
new_cost = 0
curr_sol = best_sol
new_sol = best_sol
# sim annealing params
temp=100


# TODO::
# bounds and step_size determine how far we move each iteration
# (for sim annealing, not random)
bounds=[1,5]
step_size=[1,1]




# keep track of top x (2) solutions to see if they compile
#top_sols = [best_sol, best_sol]

tested_sols = [[tables, table_size, thresh]]

start_time = time.time()
while True:
    print(new_cost)
    # stop after x iterations
    if iterations >= 1:
        break
    # run interp w/ current symb file
    m = interp_sim("rtt.dpt","total.txt")
    # use measurement to choose next vals --> compare w ground truth, and then move accordingly
    new_cost = calc_cost(m,ground_truth)
    '''
    # sim annealing
    # we need at least 2 data points for sim annealing, so if it's first iteration just randomly choose next one
    if iterations < 1:
        best_cost = new_cost
        curr_cost = new_cost
        tables, table_size, thresh = sim_move()
        new_sol = [tables, table_size, thresh]
        write_symb(tables, table_size, thresh)
        iterations += 1
        continue
    # we have at least 2 data points, so can do sim annealing
    curr_sol, curr_cost, new_sol, best_sol, best_cost, temp = simulated_annealing_step(curr_sol,curr_cost,new_sol,new_cost,best_sol,best_cost,temp,iterations,step_size,bounds)
    write_symb(new_sol[0],new_sol[1],new_sol[2],new_sol[3])
    iterations += 1
    '''

    #'''
    # random
    if iterations < 1:
        best_cost = new_cost
        while [tables, table_size, thresh] in tested_sols:   # don't test the same sols
            tables, table_size, thresh = sim_move()
        tested_sols.append([tables, table_size, thresh])
        best_sol = [tables, table_size, thresh]
        write_symb(tables, table_size, thresh)
        iterations += 1
        continue

    '''
    # RTT COST
    if 
    #if (new_cost<0.4 and (num_long*l_slots+(max_short_idx+1)*s_slots) < (best_sol[3]*best_sol[1]+(best_sol[2]+1)*best_sol[0])):
        best_cost = new_cost
        best_sol = [tables, table_size, thresh]

    '''
    while [tables, table_size, thresh] in tested_sols:   # don't test the same sols
        tables, table_size, thresh = sim_move()
    tested_sols.append([tables, table_size, thresh])
    write_symb(tables, table_size, thresh)
    iterations += 1
    #'''

end_time = time.time()
print("BEST:")
print(best_sol)
print("TIME(s):")
print(end_time-start_time)

# we test the top x solutions to see if they compile --> if they do, we're done!
# else, we can repeat above loop, excluding solutions we now know don't compile
# (we have to have a harness p4 file for this step, but not for interpreter)
# for now, we have a second lucid program that doesn't have interpreter sim measurements, this is the version we want to compile to tofino
# NOTE: use vagrant vm to compile
'''
for sol in top_sols:
    write_symb(sol[0],sol[1])
    # compile lucid to p4
    cmd_lp4 = ["../../dptc cms_sym_nomeasure.dpt ip_harness.p4 linker_config.json cms_sym_build --symb cms_sym.symb"]
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

