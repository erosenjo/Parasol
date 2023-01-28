from scipy import interpolate
import json
from scapy.all import *
import similaritymeasures
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import math
import pickle
import dpkt

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
        #info["max time"] = 9999999999999
        info["default input gap"] = 800
        info["random seed"] = 0
        info["python file"] = "precision.py"

        starttime = 0

        counts = {}
        events = []
        pkt_counter = 0

        pcap = dpkt.pcap.Reader(open(self.pkts,'rb'))
        for ts, buf in pcap:
            try:
                # caida parsing
                ip = dpkt.ip.IP(buf)
                if ip.p != 6:   # not a tcp pkt
                    continue
                tcp = ip.data
                if type(tcp) != dpkt.tcp.TCP: # just double checking tcp
                    continue
                pkt_counter +=1
                # testing
                if pkt_counter <= 5000000:
                    continue
                src_uint = struct.unpack("!I", ip.src)[0]
                dst_uint = struct.unpack("!I", ip.dst)[0]

                args = [src_uint, dst_uint, tcp.sport, tcp.dport, 0,0,0]
                p = {"name":"ip_in", "args":args}
                events.append(p)

                # ground truth calc
                key = str(src_uint)+str(dst_uint)+str(tcp.sport)+str(tcp.dport)
                if key in counts:
                    counts[key] += 1
                else:
                    counts[key] = 1

                # training data: first 1000000 tcp pkts in caida
                # test data: 10000000 - end, caida
                #if len(events) > 1:
                #    break
            except dpkt.dpkt.UnpackError:
                pass

        '''
        # scapy
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

        '''

        info["events"] = events
        print("FLOWS", len(counts.keys()))
        print("PACKETS", len(events))
        # add dummy packet
        args = [0, 0, 0, 0, 0,0,0]
        p = {"name":"ip_in", "args":args}
        events.append(p)

        info["max time"] = len(events) * info["default input gap"] * 1000
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

    

        print("ERRS", errs)
        print("AVG ERROR FOR TOP 128: ", sum(errs)/len(errs))              

        return sum(errs)/len(errs)

    def init_iteration(self, symbs):
        pass

#'''
o = Opt("/media/data/mh43/Lucid4All/traces/equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
o.gen_traffic()

cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["counts.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
#'''

#m = pickle.load(open('counts.txt','rb'))
#o.calc_cost([m])



