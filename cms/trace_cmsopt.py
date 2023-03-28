# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json
from scapy.all import *
import pickle
import time
import random

# helpers
def i2Hex (n):
    hstr = str(hex(int(n)))
    if "0x" in hstr:
        if len(hstr[2:]) == 1: return "0" + hstr[2:].upper()
        else: return hstr[2:].upper()
    else:
        return hstr.upper()

def hexadecimal (ip_addr):
    # 0xC00002EB ( Hexadecimal )
    return "0x" + "".join( map( i2Hex,  ip_addr.split(".") ) )

class Opt:
    def __init__(self):
        # ground truth for entire trace
        self.ground_truth = {}
        # ground truth for most recent set of packets
        self.iteration_ground_truth = {}

    # this gets called before each iteration
    # we can adjust the attributes of a trace by changing arguments
    def gen_traffic(self, traceparams, prev_timestamp):
        # reset ground truth for this iteration
        self.iteration_ground_truth = {}
        numpkts = traceparams["numpkts"]
        numflows = traceparams["numflows"]
        distribution = traceparams["distribution"]
        rate = traceparams["rate"]
        events = []

        # generate packets
        for p in range(numpkts):
            # for each packet, generate its header fields according to input arguments
            # e.g., if distribution = random, then we randomly select from numflows to set IPs
            #       if rate = 1000, then we send pkts every 1000ns (set timestamp = p*rate)
            if distribution == "random":
                ip_int = random.randint(1,numflows)
                timestamp = prev_timestamp+((p+1)*rate)
                args = [128, ip_int, ip_int, 0]
                p = {"name":"ip_in", "args":args, "timestamp":timestamp}
                events.append(p)
                if str(ip_int)+str(ip_int) not in self.ground_truth:
                    self.ground_truth[str(ip_int)+str(ip_int)] = 1
                else:
                    self.ground_truth[str(ip_int)+str(ip_int)] += 1
                if str(ip_int)+str(ip_int) not in self.iteration_ground_truth:
                    self.iteration_ground_truth[str(ip_int)+str(ip_int)] = 1
                else:
                    self.iteration_ground_truth[str(ip_int)+str(ip_int)] += 1


        # update last dummy byte, this helps us identify the last pkt
        # (we call different extern on last pkt to write counts to file)
        events[-1]["args"][-1] = 1



        #events_bytes = [(json.dumps(p)+'\n').encode('utf-8') for p in events]
        events_bytes = (json.dumps(events)+'\n').encode('utf-8')
        #print(self.iteration_ground_truth)

        # add a control event that cleans entire sketch
        # TODO: CLEAN SKETCH
        #   for a single sketch, this could look something like:
        #       {"type": "command", "name":"Array.setrange", "args":{"array":"myarr", "start":0, "end":8,"value":[0]}}
        # or we can have rotating sketches that we switch off writing/cleaning from
        # but how do we know the name of sketches if they're defined w/ symbolics???? (check interp output)

        return events_bytes


    # called after every interp run
    # measurement is list of measurements (one measurement for each output file)
    # order in list is same ordered specified in opt json
    def calc_cost(self,measure):  # compute avg error for our cms (mean abs error)
        m = measure[0]  # cms only has 1 output file, so 1 set of measurements
        s = []
        for k in self.iteration_ground_truth:
            #if abs(m[k]-self.ground_truth[k])/self.ground_truth[k] > 1:
                #print("ERR!! ERROR > 1")
                #print("est: "+str(m[k]))
                #print("gt: "+str(self.ground_truth[k]))
                #print("key: "+str(k))
                #quit()
            s.append(abs(m[k]-self.iteration_ground_truth[k])/self.iteration_ground_truth[k])
        #if sum(s)/len(s) > 1:
            #print("ERR!!!")
            #quit()
        print(sum(s)/len(s))
        return sum(s)/len(s)

    # called before every interp run
    # create a blank json file with no events
    # (pass in events later in interactive mode)
    def init_simulation(self, maxpkts):
        info = {}
        info["switches"] = 1
        info["max time"] = maxpkts*1000
        info["default input gap"] = 100
        info["random seed"] = 0
        info["python file"] = "cms_sym.py"
        info["events"] = []
        with open('cms_sym.json', 'w') as f:
            json.dump(info, f, indent=4)
        pass

'''
o = Opt("pcap")
o.gen_traffic()
#exit()
starttime = time.time()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

print("TIME", time.time()-starttime)
measurement = []
outfiles = ["test.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
'''


