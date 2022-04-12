# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json
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
        self.ground_truth = {}
        self.pkts = pktpcap


    # this should convert pcap into json format for interp (can write it here, or write in init_iteration, see starflow)
    # call this once before we start optimization
    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 9999999
        info["default input gap"] = 100
        info["random seed"] = 0
        info["python file"] = "flowletswitching.py"
        events = []
        starttime = 1261067164.398500 # placeholder until figure out timestamps
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                if not (pkt.haslayer(IP)) or not (pkt.haslayer(TCP)):
                    continue
                timestamp = int((pkt.time-starttime)*1000000000)    # placeholder until figure out timestamps
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                srcport = pkt[TCP].sport
                dstport = pkt[TCP].dport
                protocol = pkt[IP].proto
                args = [timestamp, src_int, dst_int, srcport, dstport, protocol]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                #print(pkt[IP].src)
                #print(int(hexadecimal(pkt[IP].src),0))
                #print(pkt[IP].dst)
                #print(int(hexadecimal(pkt[IP].dst),0))
                if len(events) > 200:
                    break

        info["events"] = events
        with open('flowletswitching.json', 'w') as f:
            json.dump(info, f, indent=4)

    # called after every interp run
    # measurement is list of measurements (one measurement for each output file)
    # order in list is same ordered specified in opt json
    # measure is dict with list of hops and num flowlets sent across each hop
    def calc_cost(self,measure):
        # for cost, we're measuring how far off we are from even distribution
        # compute average of distance from actual num flowlets vs ideal
        hops = measure[0]
        total_flowlets = sum(list(hops.values()))
        even_distr = total_flowlets/len(hops)
        diffs = 0
        for i in hops:
            diffs += abs(hops[i]-even_distr)

        return diffs/total_flowlets


    # called before every interp run
    def init_iteration(self, symbs):
        pass


