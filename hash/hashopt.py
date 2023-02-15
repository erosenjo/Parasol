# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json, pickle, time
from scapy.all import *

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
        self.ground_truth = 0
        self.pkts = pktpcap


    # this should convert pcap into json format for interp (can write it here, or write in init_iteration, see starflow)
    # call this once before we start optimization
    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 999999999
        info["default input gap"] = 1
        info["random seed"] = 0
        info["python file"] = "hash.py"
        events = []
        pkt_counter = 0
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                if not (pkt.haslayer(IP)):
                    continue
                if not (pkt.haslayer(TCP)):
                    continue
                pkt_counter += 1
                # for testing, use pkts 1000000-2000000
                if pkt_counter < 1000000:
                    continue
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                args = [src_int, dst_int,sport,dport]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                self.ground_truth += 1
                #print(pkt[IP].src)
                #print(int(hexadecimal(pkt[IP].src),0))
                #print(pkt[IP].dst)
                #print(int(hexadecimal(pkt[IP].dst),0))

                # first 100000 events for training
                # 10000000 events for testing
                #if len(events) > 1000000:
                if len(events) > 10000000:
                    break


        info["events"] = events
        with open('hash.json', 'w') as f:
            json.dump(info, f, indent=4)

    # called after every interp run
    # measurement is list of measurements (one measurement for each output file)
    # order in list is same ordered specified in opt json
    def calc_cost(self,measure):
        m = measure[0]  
        
        print("COLLISION RATE", m/self.ground_truth)
        return m/self.ground_truth

    # called before every interp run
    def init_iteration(self, symbs):
        pass


'''
o = Opt("pcap")
o.gen_traffic()
starttime = time.time()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)
endtime = time.time()
measurement = []
outfiles = ["colls.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
print("TESTTIME", endtime-starttime)
'''


