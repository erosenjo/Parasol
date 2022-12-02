# (user-provided) class containing functions necessary for optimization:
#   gen traffic (as json), init function called each time we run interp iteration

import json
from scapy.all import *
import pickle
import dpkt
import ipaddress

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
        info["max time"] = 9999999
        info["default input gap"] = 1
        info["random seed"] = 0
        info["python file"] = "flowletswitching.py"
        events = []
        pkt_counter = 0
        #starttime = 1261067164.398500 # placeholder until figure out timestamps, univ1 trace
        #starttime = 1453381151.415037 # caida trace, training
        starttime = 0   # caida trace, testing

        pcap = dpkt.pcap.Reader(open(self.pkts,'rb'))
        for ts, buf in pcap:
            pkt_counter += 1
            if pkt_counter < 2000000:
                continue
            if pkt_counter == 2000000:
            #if pkt_counter == 1:
                starttime = ts
            try:
                ip = dpkt.ip.IP(buf)
                if ip.p != 6:   # not a tcp pkt
                    continue
                tcp = ip.data
                if type(tcp) != dpkt.tcp.TCP:
                    continue
                src_uint = struct.unpack("!I", ip.src)[0]
                dst_uint = struct.unpack("!I", ip.dst)[0]
                timestamp = int((ts-starttime)*1000000000)
                args = [timestamp, src_uint, dst_uint, tcp.sport, tcp.dport, ip.p, ip.len] 
                p = {"name":"ip_in", "args":args}
                events.append(p)
                self.ground_truth += ip.len
                # training: first 500000 pkts of caida trace
                # testing: 3000000 pkts, starting w/ 2000000 of caida trace
                #if len(events) >= 500000:
                if len(events) >= 3000000:
                    print("max events")
                    break
            except dpkt.dpkt.UnpackError:
                pass
                #print("unpack error", pkt_counter)

        '''
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                pkt_counter += 1
                if pkt_counter == 2000068:
                    print("IP?", pkt.haslayer(IP))
                    print("TCP?", pkt.haslayer(TCP))
                    exit()
                if not (pkt.haslayer(IP)) or not (pkt.haslayer(TCP)):
                    continue
                if pkt_counter < 2000000:
                    continue
                timestamp = int((pkt.time-starttime)*1000000000)    # placeholder until figure out timestamps
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                srcport = pkt[TCP].sport
                dstport = pkt[TCP].dport
                pktlen = pkt[IP].len
                protocol = pkt[IP].proto
                args = [timestamp, src_int, dst_int, srcport, dstport, protocol, pktlen]
                p = {"name":"ip_in", "args":args}
                events.append(p)


                self.ground_truth += pktlen

                #print(pkt[IP].src)
                #print(int(hexadecimal(pkt[IP].src),0))
                #print(pkt[IP].dst)
                #print(int(hexadecimal(pkt[IP].dst),0))
                #if len(events) > 10000:
                # training: first 500000 pkts of caida trace
                # testing: pkts 2000000-5000000 of caida trace
                #if len(events) >= 500000:
                if len(events) >= 3000000:
                    break

        '''


        self.ground_truth = self.ground_truth/4


        print(len(events))

        info["events"] = events
        with open('flowletswitching.json', 'w') as f:
            json.dump(info, f, indent=4)


    # called after every interp run
    # measurement is list of measurements (one measurement for each output file)
    # order in list is same ordered specified in opt json
    # measure is dict with list of hops and num flowlets sent across each hop
    def calc_cost(self,measure):
        '''
        # for cost, we're measuring how far off we are from even distribution
        # compute average of distance from actual num flowlets vs ideal
        hops = measure[0]
        total_flowlets = sum(list(hops.values()))
        even_distr = total_flowlets/len(hops)
        diffs = 0
        for i in hops:
            diffs += abs(hops[i]-even_distr)

        return diffs/total_flowlets
        '''

        print(self.ground_truth)

        hops = measure[0]
        errs = []
        for i in hops:
            print(hops[i])
            errs.append(abs(hops[i]-self.ground_truth)/self.ground_truth)

        print(sum(errs)/len(errs))
        return sum(errs)/len(errs)
        


    # called before every interp run
    def init_iteration(self, symbs):
        pass



#o = Opt("equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
#o.gen_traffic()
#m = [pickle.load(open('pkthops.txt','rb'))]
#o.calc_cost(m)

o = Opt("/media/data/mh43/Lucid4All/traces/equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
o.gen_traffic()

cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["pkthops.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)





