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
        self.ground_truth = {}


    # this gets called before each iteration
    # we can adjust the attributes of a trace by changing arguments
    def gen_traffic(self, traceparams):
        numpkts = traceparams["numpkts"]
        numflows = traceparams["numflows"]
        distribution = traceparams["distribution"]
        rate = traceparams["rate"]
        infopkts = []
        info = {}
        info["switches"] = 1
        # multiply by 100 just as a buffer (will need to do this for sure if there's recirculation)
        info["max time"] = numpkts*rate*100
        info["default input gap"] = rate
        info["random seed"] = 0
        info["python file"] = "cms_sym.py"
        events = []

        # generate packets
        for p in range(numpkts):
            # for each packet, generate its header fields according to input arguments
            # e.g., if distribution = random, then we randomly select from numflows to set IPs
            #       if rate = 1000, then we send pkts every 1000ns (set timestamp = p*rate)
            if distribution == "random":
                ip_int = random.randint(1,numflows)
                timestamp = p*rate
                args = [128, ip_int, ip_int, 0]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                if str(ip_int)+str(ip_int) not in self.ground_truth:
                    self.ground_truth[str(ip_int)+str(ip_int)] = 1
                else:
                    self.ground_truth[str(ip_int)+str(ip_int)] += 1

        # update last dummy byte, this helps us identify the last pkt
        # (we call different extern on last pkt to write counts to file)
        events[-1]["args"][-1] = 1

        info["events"] = events
        with open('cms_sym.json', 'w') as f:
            json.dump(info, f, indent=4)

    # called after every interp run
    # measurement is list of measurements (one measurement for each output file)
    # order in list is same ordered specified in opt json
    def calc_cost(self,measure):  # compute avg error for our cms (mean abs error)
        m = measure[0]  # cms only has 1 output file, so 1 set of measurements
        s = []
        for k in self.ground_truth:
            #if abs(m[k]-self.ground_truth[k])/self.ground_truth[k] > 1:
                #print("ERR!! ERROR > 1")
                #print("est: "+str(m[k]))
                #print("gt: "+str(self.ground_truth[k]))
                #print("key: "+str(k))
                #quit()
            s.append(abs(m[k]-self.ground_truth[k])/self.ground_truth[k])
        #if sum(s)/len(s) > 1:
            #print("ERR!!!")
            #quit()
        print(sum(s)/len(s))
        return sum(s)/len(s)

    # called before every interp run
    def init_iteration(self, symbs):
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


