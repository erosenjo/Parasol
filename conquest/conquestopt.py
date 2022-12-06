import numpy as np
import json
import pickle
import subprocess

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
        info["max time"] = 99999999999
        info["default input gap"] = 1
        info["random seed"] = 0
        info["python file"] = "conquest.py"
        events = []

        '''
        self.ground_truth = np.load("reqfiles_dctraining/conquestgt.npy")
        self.qlen_arr = np.load("reqfiles_dctraining/qlen_arr.npy")
        self.vect_qlong = np.load("reqfiles_dctraining/vect_qlong.npy")
        self.tracepkts = np.load("reqfiles_dctraining/tracepkts.npy")
        '''
        self.ground_truth = np.load("reqfiles_dctest/conquestgt.npy")
        self.qlen_arr = np.load("reqfiles_dctest/qlen_arr.npy")
        self.vect_qlong = np.load("reqfiles_dctest/vect_qlong.npy")
        self.tracepkts = np.load("reqfiles_dctest/tracepkts.npy")
        pkt_counter = 0
        events = []
        for pkt in self.tracepkts[:len(self.vect_qlong)]:
            pkt_counter += 1
            # we can use the original trace to get these vals, but to make it easier we're using numerical flowid in tracepkts
            # flowid is computed in SnappySimNG code, see trace_parsing.ipynb
            sport = int(pkt[4])
            dport = int(pkt[4])
            src_int = int(pkt[4])
            dst_int = int(pkt[4])
            global_time=int(pkt[5] * 1000000000) # convert s to ns
            qdelay = int((pkt[5]-pkt[2]) * 1000000000 )
            size_bytes = int(pkt[3])
            args = [sport, dport, src_int, dst_int, qdelay, global_time, size_bytes]           
            p = {"name":"ip_in", "args":args}
            events.append(p) 

        '''
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
                    #pkt.show()
                    src_int = pkt.src.split(":")[-4:]
                    #src_int = pkt.src.split(":")[:4]
                    #print(src_int)
                    src_int = [str(int(i,16)) for i in src_int]
                    src_int = ".".join(src_int)
                    src_int = int(hexadecimal(src_int),0)
                    dst_int = pkt.dst.split(":")[-4:]
                    #dst_int = pkt.dst.split(":")[:4]
                    dst_int = [str(int(i,16)) for i in dst_int]
                    dst_int = ".".join(dst_int)
                    dst_int = int(hexadecimal(dst_int),0)


                global_time=int(self.tracepkts[pkt_counter,5] * 1000000000) # convert s to ns
                qdelay = int((self.tracepkts[pkt_counter,5]-self.tracepkts[pkt_counter,2]) * 1000000000 )
                if qdelay < 0:
                    qdelay = 0
                size_bytes = int(self.tracepkts[pkt_counter,3])
                args = [sport, dport, src_int, dst_int, qdelay, global_time, size_bytes]
                p = {"name":"ip_in", "args":args}
                events.append(p)
                pkt_counter += 1                
                #break
                #if len(events)>2000000:
                #    break
       
        #if len(events)==1021724:
            #print("MAX EVENTS")
            #events = events[:-1]
            #pkt_counter -= 1
        #pkt_counter = 1021723

        # filter out 0s at tail (this happens in sim code too)
        events = events[:-1]
        pkt_counter -= 1

        #pkt_counter = 300001
        '''


        '''
        print(type(self.vect_qlong))
        print("VECT size", self.vect_qlong.size)
        print(type(self.qlen_arr))
        print("QLEN size", self.qlen_arr.size)
        '''

        #self.qlen_arr = self.qlen_arr[:-1]

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
        print("TOTAL EST", total_est)
        print("TOTAL GT", total_gt)
        print(TP)
        #print(total_gt)
        #print(total_est)
        if total_est==0:
            precision=1
        else:
            precision=1.0*TP/total_est
        print("PRECISION", precision)
        with open('precision.txt','a') as f:
            f.write(str(precision)+"\n")

        # recall
        if total_gt==0:
            recall=1
        else:
            recall=1.0*TP/total_gt

        print("RECALL", recall)
        with open('recall.txt','a') as f:
            f.write(str(recall)+"\n")

        # f-score
        f_score = (2 * precision * recall) / (precision + recall)

        print("FSCORE", f_score)
        return 1-f_score



    def init_iteration(self,symbs):
        pass

'''
#traces = ["univ1_pt8.pcap", "univ1_pt9.pcap", "univ1_pt10.pcap", "univ1_pt11.pcap", "univ1_pt12.pcap"]
#o = Opt("univ1_pt3.pcap")
#o = Opt("univ1_pt1.pcap")
#o = Opt("equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
o = Opt("")
o.gen_traffic()

cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["qlens.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)
'''

#test = pickle.load(open('qlens.txt','rb'))
#print(o.calc_cost([test]))


