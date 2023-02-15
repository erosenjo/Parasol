# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json
from scapy.all import *
import pickle
import time

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
    def __init__(self, pktpcap):
        self.ground_truth = {}
        self.pkts = pktpcap


    # this should convert pcap into json format for interp (can write it here, or write in init_iteration, see starflow)
    # call this once before we start optimization
    def gen_traffic(self):

        testinfo = {}
        testinfo["name"]= "cms_sym_test"
        testinfo["input_port"]= "128"
        infopkts = []
        info = {}
        info["switches"] = 1
        info["max time"] = 99999999999
        info["default input gap"] = 100
        info["random seed"] = 0
        info["python file"] = "cms_sym.py"
        events = []
        pktcounter = 0
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                if not (pkt.haslayer(IP)):
                    continue
                pktcounter += 1
                #if pktcounter < 1000000:    # get a different part of the trace to test our solution
                #    continue
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                # 0 as dummy argument, for byte-alignment
                args = [128, src_int, dst_int, 0]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                if str(src_int)+str(dst_int) not in self.ground_truth:
                    self.ground_truth[str(src_int)+str(dst_int)] = 1
                else:
                    self.ground_truth[str(src_int)+str(dst_int)] += 1


                pi = {"ip.src" : pkt[IP].src, "ip.dst": pkt[IP].dst}


                infopkts.append(pi)
                #print(pkt[IP].src)
                #print(int(hexadecimal(pkt[IP].src),0))
                #print(pkt[IP].dst)
                #print(int(hexadecimal(pkt[IP].dst),0))
                # 500000 events for training, 1000000 for testing
                #if len(events) >= 500000:
                if len(events) >= 100000:
                    break

        # update last dummy byte, this helps us identify the last pkt
        # (we call different extern on last pkt to write counts to file)
        events[-1]["args"][-1] = 1


        print("PACKETS:",len(events))
        print("FLOWS:",len(self.ground_truth))

        testinfo["packets"] = infopkts
        testinfo["model-output"]=[]


        #with open('cms_sym_test.json','w') as f:
        #    json.dump(testinfo,f,indent=4)

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


