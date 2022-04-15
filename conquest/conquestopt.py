from scapy.all import *
import numpy as np
import json

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
        self.ground_truth = None
        self.qlen_arr = None
        self.vect_qlong = None
        self.tracepkts = None
        self.thresh = 1048576
        self.alpha = 0.01
        self.pkts = pktpcap


# (int<<16>> src_port, int<<16>> dst_port, int src, int dst, int qdelay, int<<48>> global_time);
    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 9999999
        info["default input gap"] = 10
        info["random seed"] = 0
        info["python file"] = "conquest.py"
        events = []

        self.ground_truth = np.load("conquestgt.npy")
        self.qlen_arr = np.load("qlen_arr.npy")
        self.vect_qlong = np.load("vect_qlong.npy")
        self.tracepkts = np.load("tracepkts.npy")
        pkt_counter = 0
        events = []
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                sport = 0
                dport = 0
                src_int = 0
                dst_int = 0
                if pkt.haslayer(ARP):
                    src_int = int(hexadecimal(pkt[ARP].psrc),0)
                    dst_int = int(hexadecimal(pkt[ARP].pdst),0)
                elif pkt.haslayer(IP):
                    src_int = int(hexadecimal(pkt[IP].src),0)
                    dst_int = int(hexadecimal(pkt[IP].dst),0)
                    if pkt.haslayer(TCP):
                        sport = pkt[TCP].sport
                        dport = pkt[TCP].dport
                    elif pkt.haslayer(UDP):
                        sport = pkt[UDP].sport
                        dport = pkt[UDP].dport
                elif pkt.haslayer(Ether):
                    print("!!!!!!")
                    print(pkt.src)
                    quit()
                else:
                    print("2!!!!!!")
                    quit()

                global_time=int(self.tracepkts[pkt_counter,5] * 1000000000) # convert s to ns
                qdelay = int((global_time-self.tracepkts[pkt_counter,2]) * 1000000000 )

                args = [sport, dport, src_int, dst_int, qdelay, global_time]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                pkt_counter += 1                

                if len(events)>100:
                    break

        self.ground_truth = self.ground_truth[:pkt_counter]
        self.qlen_arr = self.qlen_arr[:pkt_counter]
        self.vect_qlong = self.vect_qlong[:pkt_counter]
             
        info["events"] = events
        with open('conquest.json', 'w') as f: 
            json.dump(info, f, indent=4)

    def calc_cost(self,measure):
        qlens = np.array(measure[0])
   
        qlen_v=self.qlen_arr[self.vect_qlong]    
        gt_v=self.ground_truth[self.vect_qlong] 
        est_v=qlens[self.vect_qlong]

        decision_v=qlen_v*self.alpha
        est_b=est_v>decision_v
        gt_b=gt_v>decision_v  
 
        TP=np.sum(gt_b * est_b)

        total_est=np.sum(est_b)
        total_gt=np.sum(gt_b)
        if total_est==0:
            precision=1
        else:
            precision=1.0*TP/total_est
        return 1-precision


    def init_iteration(self):
        pass

'''
o = Opt("univ1_pt1.pcap")
o.gen_traffic()
'''


