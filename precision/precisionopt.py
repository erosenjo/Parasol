from scipy import interpolate
import json
from scapy.all import *
import similaritymeasures
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import math
import pickle

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
        self.ground_truth = None
        self.pkts = pktpcap

    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 9999999
        info["default input gap"] = 10
        info["random seed"] = 0
        info["python file"] = "precision.py"

        starttime = 0

        counts = {}
        events = []

        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                if not (pkt.haslayer(IP)) or not (pkt.haslayer(TCP)):
                    continue

                if len(events) == 0:
                    starttime = pkt.time
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                # adding last 0 as dummy value for byte alignment 
                args = [src_int, dst_int, sport, dport, 0,0,0]
                p = {"name":"ip_in", "args":args}
                events.append(p)

                key = str(src_int)+str(dst_int)+str(sport)+str(dport)
                if key in counts:
                    counts[key] += 1
                else:
                    counts[key] = 1

                if len(events) > 10000000:
                    totaltime = pkt.time - starttime
                    print("TOTAL TIME NS", totaltime)
                    break



        info["events"] = events
        print(len(events))
        #exit()
        with open('precision.json', 'w') as f:
            json.dump(info, f, indent=4)


        sorted_counts = sorted(counts.items(), key=operator.itemgetter(1),reverse=True)

        self.ground_truth = sorted_counts[0:128]
        pickle.dump(self.ground_truth, open('gt.pkl','wb'))


    def calc_cost(self, measure):
        counts = measure[0]
        #sorted_counts = sorted(counts.items(), key=operator.itemgetter(1),reverse=True)
        errs = []

        #print(self.ground_truth[0])
        #quit()

        for k in range(len(self.ground_truth)):
            fid = self.ground_truth[k][0]
            act = self.ground_truth[k][1]
            est = 0

            if fid in counts:
                est = counts[fid]

            #print(est)
            #print(act)

            errs.append(abs(est-act)/act)

    

        print("AVG ERROR FOR TOP 128: ", sum(errs)/len(errs))              

        return sum(errs)/len(errs)

    def init_iteration(self, symbs):
        pass

#o = Opt("equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
#o.gen_traffic()

'''
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["counts.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
'''

#m = pickle.load(open('counts.txt','rb'))
#o.calc_cost([m])



