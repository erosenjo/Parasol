from scapy.all import *
import numpy as np
import json
import pickle

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
        info["default input gap"] = 1
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
                    src_int = pkt[Ether].src.split(":")[-4:]
                    src_int = [str(int(i,16)) for i in src_int]
                    src_int = ".".join(src_int)
                    src_int = int(hexadecimal(src_int),0)
                    dst_int = pkt[Ether].dst.split(":")[-4:]
                    dst_int = [str(int(i,16)) for i in dst_int]
                    dst_int = ".".join(dst_int)
                    dst_int = int(hexadecimal(dst_int),0)
                    sport = pkt[Ether].type
                    dport = pkt[Ether][1].type
                else:
                    src_int = pkt.src.split(":")[-4:]
                    src_int = [str(int(i,16)) for i in src_int]
                    src_int = ".".join(src_int)
                    src_int = int(hexadecimal(src_int),0)
                    dst_int = pkt.dst.split(":")[-4:]
                    dst_int = [str(int(i,16)) for i in dst_int]
                    dst_int = ".".join(dst_int)
                    dst_int = int(hexadecimal(dst_int),0)

                global_time=int(self.tracepkts[pkt_counter,5] * 1000000000) # convert s to ns
                qdelay = int((self.tracepkts[pkt_counter,5]-self.tracepkts[pkt_counter,2]) * 1000000000 )

                size_bytes = int(self.tracepkts[pkt_counter,3])
                args = [sport, dport, src_int, dst_int, qdelay, global_time, size_bytes]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                pkt_counter += 1                

                #break
                if len(events)>300000:
                    break

        if len(events)==1021724:
            events = events[:-1]
            pkt_counter -= 1
        #pkt_counter = 1021723

        #pkt_counter = 300001
        self.qlen_arr = self.qlen_arr[:-1]

        self.ground_truth = self.ground_truth[:pkt_counter]
        self.qlen_arr = self.qlen_arr[:pkt_counter]
        self.vect_qlong = self.vect_qlong[:pkt_counter]
        
        qlen_v=self.qlen_arr[self.vect_qlong]
        gt_v=self.ground_truth[self.vect_qlong]

        decision_v=qlen_v*self.alpha
        gt_b=gt_v>decision_v

        #print(len(gt_b))

        #print(self.qlen_arr)
        #print(self.ground_truth)

        #return
        info["events"] = events
        with open('conquest.json', 'w') as f: 
            json.dump(info, f, indent=4)

    def calc_cost(self,measure):
        qlens = np.array(measure[0])
   
        qlen_v=self.qlen_arr[self.vect_qlong]    
        gt_v=self.ground_truth[self.vect_qlong] 
        est_v=qlens[self.vect_qlong]

        decision_v=qlen_v*self.alpha
        #print(decision_v)
        est_b=est_v>decision_v
        gt_b=gt_v>decision_v  
        #print(est_b)
        #print(np.sum(est_b))
        TP=np.sum(gt_b * est_b)
        #print(TP)

        total_est=np.sum(est_b)
        total_gt=np.sum(gt_b)
        print(total_est)
        print(total_gt)
        print(TP)
        #print(total_gt)
        #print(total_est)
        if total_est==0:
            precision=1
        else:
            precision=1.0*TP/total_est
        print(precision)
        with open('precision.txt','a') as f:
            f.write(str(precision)+"\n")

        # recall
        if total_gt==0:
            recall=1
        else:
            recall=1.0*TP/total_gt

        print(recall)
        with open('recall.txt','a') as f:
            f.write(str(recall)+"\n")

        # f-score
        f_score = (2 * precision * recall) / (precision + recall)

        print(f_score)
        return 1-f_score



    def init_iteration(self,symbs):
        pass


#o = Opt("univ1_pt1.pcap")
#o.gen_traffic()

#test = pickle.load(open('qlens.txt','rb'))
#print(o.calc_cost([test]))


