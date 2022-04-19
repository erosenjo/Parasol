# how many ns of packet trace to use as a sample
max_packet_sample_time = 4000000000

from scipy import interpolate
import json
# from scapy.all import *
import similaritymeasures
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import math
import pickle


import dpkt
import ipaddress
import struct

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


def events_of_pkts(inPcapFn, maxts = 1000000000):
    def ns_ts(ts):
        return int(ts * 1000000000)
    print ("converting packets to events")
    keys = set()
    events = []
    pcap = dpkt.pcap.Reader(open(inPcapFn, "rb"))
    pktct = 0
    auth_pktct = 0
    ret_pktct = 0    
    fst_ts = 0
    for ts, buf in pcap:
        if (fst_ts == 0):
            fst_ts = ts
        ts = ns_ts(ts - fst_ts)
        if (ts > maxts):
            break
        # if (maxpkts != None):        
        #     if (pktct == maxpkts): 
        #         break
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            authorized = None
            if (type(eth.data) == dpkt.ip.IP):
                src_uint = struct.unpack("!I", eth.ip.src)[0]
                dst_uint = struct.unpack("!I", eth.ip.dst)[0]
                src = str(ipaddress.ip_address(eth.ip.src))
                dst = str(ipaddress.ip_address(eth.ip.dst))
                key = (src, dst)
                reverse_key = (dst, src)
                # case: some packet from return of authorized flow
                if (reverse_key in keys):
                    authorized = 0
                    ret_pktct += 1
                # case: first packet from authorized flow
                elif (key in keys):
                    authorized = 1
                # case: first packet of a flow. Consider it authorized. 
                else:
                    keys.add(key)
                    authorized = 1
                if (authorized == 1):
                    auth_pktct += 1
                event = {"name":"ip_in", "args":[0, src_uint, dst_uint, 64, authorized], "timestamp":ts}
                events.append(event)
                pktct += 1
        except dpkt.dpkt.UnpackError:
            print ("unpack error")
    print ("packet to event conversion done")
    print ("pkts: %s authorized: %s return: %s keys: %s"%(pktct, auth_pktct, ret_pktct, len(keys)))
    return events

class Opt:
    def __init__(self, pktpcap):
        self.ground_truth = None
        self.pcapfn = pktpcap
        self.result_trace = []

    def gen_traffic(self):
        print ("generating traffic")
        info = {}
        info["switches"] = 1
        info["default input gap"] = 1
        info["random seed"] = 0
        info["python file"] = "stateful_firewall.py" # externs
        pktevents = events_of_pkts(self.pcapfn, max_packet_sample_time)
        # final event to trigger dump of stats
        initevent = {"name":"check_timeout", "args":[0], "timestamp":0}
        finalevent = {"name":"final_packet", "args":[], "timestamp":(pktevents[-1]["timestamp"]+1)}
        info["events"] = [initevent] + pktevents + [finalevent] 
        # info["events"] = pktevents + [finalevent] 
        info["max time"] = finalevent["timestamp"]+1
        with open('stateful_firewall.json', 'w') as f:
            json.dump(info, f, indent=4)


    # measurement is dictionary of rtts and correction factors
    def calc_cost(self, measure):     
        measure = measure[-1]
        # cost: 
        # factor 1: number of recirc / number total (minimize)
        overhead = float(measure["counters"]["recirc_pkts"] / (measure["counters"]["data_pkts"] + measure["counters"]["recirc_pkts"]))
        # factor 2: 1 - accuracy (incorrect / total)
        incorrects =  float(measure["counters"]["incorrect_pkts"] / (measure["counters"]["correct_pkts"] + measure["counters"]["incorrect_pkts"]))

        goodput = float(measure["counters"]["data_pkts"] / (measure["counters"]["data_pkts"] + measure["counters"]["recirc_pkts"]))
        accuracy = float(measure["counters"]["correct_pkts"] / (measure["counters"]["correct_pkts"] + measure["counters"]["incorrect_pkts"]))
        cost = overhead + incorrects        
        print ("parameters: %s"%(str(self.cur_params)))
        cost = incorrects
        # low goodput -- bad bad bad
        pps_overhead = measure["counters"]["recirc_pkts"] 
        bps_overhead = 64 * pps_overhead * 8
        self.result_trace.append((self.cur_params, {"accuracy":accuracy, "bps_overhead":bps_overhead}))
        # max of 1 mbps for overhead
        cost = 10
        if (bps_overhead < 1000000):
            cost = incorrects
        else:
            cost = math.pow((bps_overhead / 1000000.0), 2) + (incorrects)
        print ("accuracy: %f bps_overhead: %f cost: %f"%(accuracy, bps_overhead, cost))
        return cost            


    def init_iteration(self, symbs):
        self.cur_params = symbs
        print("init_iteration called")
        pass

#o = Opt("univ1_pt1.pcap")
#o.gen_traffic()
#m = pickle.load(open('rtt_correction.txt','rb'))
#o.calc_cost([m])



