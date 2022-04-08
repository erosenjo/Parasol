import json
from scapy.all import *
import similaritymeasures
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

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
        self.cdf_xvals = []

    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 9999999
        info["default input gap"] = 100
        info["random seed"] = 0
        info["python file"] = "rtt.py"
        events = []
        starttime = 1261067164.398500 # placeholder until figure out timestamps
        syns = {}
        gt = []
        with PcapReader(self.pkts) as pcap_reader:
            for pkt in pcap_reader:
                if not (pkt.haslayer(IP)) or not (pkt.haslayer(TCP)):
                    continue
                src_int = int(hexadecimal(pkt[IP].src),0)
                dst_int = int(hexadecimal(pkt[IP].dst),0)
                pktlen = pkt[IP].len
                ihl = pkt[IP].ihl
                offset = pkt[TCP].dataofs
                seq = pkt[TCP].seq
                ack = pkt[TCP].ack
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                timestamp = int((pkt.time-starttime)*1000000000)    # placeholder until figure out timestamps
                # getting int value of flags
                # this is so annoying is there a better way?
                f_str = pkt[TCP].flags
                f_int = 0
                if "F" in f_str:
                    f_int += 1
                if "S" in f_str:
                    f_int += 2
                if "R" in f_str:
                    f_int += 4
                if "P" in f_str:
                    f_int += 8
                if "A" in f_str:
                    f_int += 16
                args = [f_int, pktlen, ihl, offset, seq, ack, src_int, dst_int, sport, dport, timestamp]
                p = {"name":"tcp_in", "args":args}
                events.append(p)

                # ground truth calc
                pkttype=0
                drop = False
                if f_int==2:
                    pkttype=1
                elif f_int==18:
                    pkttype=0
                elif f_int==16 and pktlen <= 80:
                    pkttype=0
                elif f_int==24 and pktlen <= 80:
                    pkttype=0
                elif 80 <= pktlen <= 1600:
                    pkttype=1
                elif f_int==4 or f_int==1:
                    drop=True
                else:
                    pkttype=0

                print(args)
                print("PKT TYPE: "+str(pkttype))
                if pkttype==1 and drop==False:
                    #print("IHL "+str(ihl*4))
                    #print("OFFSET "+str(offset*4))
                    tmp4 = ihl*4+offset*4
                    #print("LEN "+str(pktlen))
                    tmp5 = pktlen-tmp4
                    #print("TMP4 "+str(tmp4))
                    #print("TMP5 "+str(tmp5))
                    eack = seq+tmp5
                    #print("EXPECTED "+str(eack))
                    if f_int==2:
                        eack = eack+1
                    key = str(src_int)+str(dst_int)+str(sport)+str(dport)+str(eack)


                    '''
                    if args==[24, 108, 5, 5, 389537286, 3974322786, 4093882517, 699466720, 1869, 9809, 5524034]:
                        print(f_str)
                        print("EACK: "+str(eack))
                        print("KEY: "+str(key))
                    '''
                    syns[key] = timestamp
                    #print("EACK: "  + str(eack))
                elif drop==False:
                    key = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(ack)

                    '''
                    if args==[16, 40, 5, 5, 3974322862, 389537355, 699466720, 4093882517, 9809, 1869, 6418034]:
                        print("KEY: "+str(key))
                    '''

                    if key in syns:
                        #print("MATCH!!")
                        gt.append(timestamp-syns[key])

                if len(events) > 200:
                    break
        info["events"] = events
        #with open('rtt.json', 'w') as f:
        #    json.dump(info, f, indent=4)

        print(gt)
        #self.ground_truth = len(events)
        #self.ground_truth = gt
        # compute cdf w/ ground truth: (converting to ms from ns)
        ecdf = ECDF([x/1000000 for x in gt])
  
        # x vals go from 0 to max rtt in our data set???
        # should be max in ms, not ns?????
        #self.cdf_xvals = range(0,max(gt)/1000000)
        # TODO: include 0??? idk, maybe; and maybe pad w/ max val
        self.cdf_xvals = [0]+[x/1000000 for x in gt]
        gt_cdf = ecdf(self.cdf_xvals)
        self.ground_truth=np.zeros((len(self.cdf_xvals),2))
        self.ground_truth[:,0]=self.cdf_xvals
        self.ground_truth[:,1] = gt_cdf

    # measurements are num collisions, num timeouts, num rtt samples
    # don't currently calc total num of rtt samples in trace, but can calc in gen_traffic
    def calc_cost(self,measure):
        # placeholder for now, should ideally include all 3 measurements
        # TODO: how to use collisions/timeouts to guide search? timeouts affected by timeout value and struct size?
        #return measure[0]/self.ground_truth
        # measure is list of RTT samples, we're gonna generate CDF from them
        ecdf = ECDF([m/1000000 for m in measure]) 
        samp_vals = ecdf(self.cdf_xvals)
        sv = np.zeros((len(self.cdf_xvals),2))
        sv[:,0]=self.cdf_xvals
        sv[:,1]=samp_vals

        return similaritymeasures.area_between_two_curves(self.ground_truth, sv)



    def init_iteration(self,symbs):
        pass



# compute cdf as cost, diff between ideal and actual cdf is cost
# ground truth is all rtts observed?

# statsmodels ECDF?
# then get curve by doing ecdf([x vals]) - from some min rtt to some max rtt
# then use similaritymeasures.area_between_two_curves
# ecdf = ECDF([measurement])

#o = Opt("univ1_pt1.pcap")
#o.gen_traffic()
