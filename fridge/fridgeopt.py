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
        info["max time"] = 9999999999
        info["default input gap"] = 1
        info["random seed"] = 0
        info["python file"] = "fridge.py"

        events = []
        #starttime = 1261067164.398500 # placeholder until figure out timestamps
        starttime = 0
        syns = {}
        gt = []
        pkt_counter = 0

        pcap = dpkt.pcap.Reader(open(self.pkts,'rb'))
        for ts, buf in pcap:
            if pkt_counter == 0:
                starttime = ts
            pkt_counter += 1
            try:
                # univ parsing:
                eth = dpkt.ethernet.Ethernet(buf)
                #print(type(eth.data))
                if type(eth.data) != dpkt.ip.IP:
                    continue
                #print(type(eth.ip.data))
                if type(eth.ip.data) != dpkt.tcp.TCP:
                    continue
                src_uint = struct.unpack("!I", eth.ip.src)[0]
                dst_uint = struct.unpack("!I", eth.ip.dst)[0]
                timestamp = int((ts-starttime)*1000000000) 
                f_int = eth.ip.tcp.flags
                #print(eth.ip.tcp.flags)
                args = [f_int, eth.ip.len, eth.ip.hl, eth.ip.tcp.off, eth.ip.tcp.seq, eth.ip.tcp.ack, src_uint, dst_uint, eth.ip.tcp.sport, eth.ip.tcp.dport, timestamp]
                '''
                # caida parsing: 
                ip = dpkt.ip.IP(buf)
                if ip.p != 6:   # not a tcp pkt
                    continue
                tcp = ip.data
                if type(tcp) != dpkt.tcp.TCP:
                    continue
                src_uint = struct.unpack("!I", ip.src)[0]
                dst_uint = struct.unpack("!I", ip.dst)[0]
                timestamp = int((ts-starttime)*1000000000)        
                f_int = tcp.flags
                args = [f_int, ip.len, ip.hl, tcp.off, tcp.seq, tcp.ack, src_uint, dst_uint, tcp.sport, tcp.dport, timestamp]
                '''
                p = {"name":"tcp_in", "args":args}
                events.append(p)

                # ground truth calc
                pkttype=0
                drop = False
                if f_int==2:
                    pkttype=1
                elif f_int==18:
                    pkttype=0
                elif f_int==16 and eth.ip.len <= 80:
                    pkttype=0
                elif f_int==24 and eth.ip.len <= 80:
                    pkttype=0
                elif 80 <= eth.ip.len <= 1600:
                    pkttype=1
                elif f_int==4 or f_int==1:
                    drop=True
                else:
                    pkttype=0

                #print(args)
                #print("PKT TYPE: "+str(pkttype))
                if pkttype==1 and drop==False:
                    #print("IHL "+str(ihl*4))
                    #print("OFFSET "+str(offset*4))
                    tmp4 = eth.ip.hl*4+eth.ip.tcp.off*4
                    #print("LEN "+str(pktlen))
                    tmp5 = eth.ip.len-tmp4
                    #print("TMP4 "+str(tmp4))
                    #print("TMP5 "+str(tmp5))
                    eack = eth.ip.tcp.seq+tmp5
                    #print("EXPECTED "+str(eack))
                    if f_int==2:
                        eack = eack+1
                    key = str(src_uint)+str(dst_uint)+str(eth.ip.tcp.sport)+str(eth.ip.tcp.dport)+str(eack)

                    '''
                    if args==[24, 108, 5, 5, 389537286, 3974322786, 4093882517, 699466720, 1869, 9809, 5524034]:
                        print(f_str)
                        print("EACK: "+str(eack))
                        print("KEY: "+str(key))
                    '''
                    syns[key] = timestamp
                    #print("EACK: "  + str(eack))
                elif drop==False:
                    key = str(dst_uint)+str(src_uint)+str(eth.ip.tcp.dport)+str(eth.ip.tcp.sport)+str(eth.ip.tcp.ack)

                    '''
                    if args==[16, 40, 5, 5, 3974322862, 389537355, 699466720, 4093882517, 9809, 1869, 6418034]:
                        print("KEY: "+str(key))
                    '''

                    if key in syns:
                        #print("MATCH!!")
                        gt.append(timestamp-syns[key])

                '''
                if len(events) > 0:
                    return
                '''

                if len(events) > 1000000:
                    break

            except dpkt.dpkt.UnpackError:
                pass

        '''
        # scapy parsing
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

                #print(args)
                #print("PKT TYPE: "+str(pkttype))
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


                    syns[key] = timestamp
                    #print("EACK: "  + str(eack))
                elif drop==False:
                    key = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(ack)

                    if key in syns:
                        #print("MATCH!!")
                        gt.append(timestamp-syns[key])

                if len(events) > 1000000:
                    break
        '''

        print(len(gt))        

        info["events"] = events
        ecdf = ECDF([x/1000000 for x in gt])
        # x vals go from 0 to max rtt in our data set???
        self.cdf_xvals = [x/1000000 for x in gt]
        #gt_cdf = ecdf(self.cdf_xvals)
        #self.ground_truth=np.zeros((len(self.cdf_xvals),2))
        #self.ground_truth[:,0]=self.cdf_xvals
        #self.ground_truth[:,1] = gt_cdf
        #print(ecdf([x/1000000000 for x in gt]))
        #print(list(ecdf([x/1000000000 for x in gt])))
        probs_gt = list(ecdf([x/1000000 for x in gt]))

        # percentile vales
        percentiles_y = [ i/100 for i in range(5,96)]
        # rtt values
        percentiles_x = []
        for y in percentiles_y:
            z = min(probs_gt, key=lambda x:abs(x-y))
            percentiles_x.append(self.cdf_xvals[probs_gt.index(z)])


        self.ground_truth = percentiles_x
        info["events"] = events
        with open('fridge.json', 'w') as f:
            json.dump(info, f, indent=4)


    # measurement is dictionary of rtts and correction factors
    def calc_cost(self, measure):
        rtts = measure[0]

        errs = []

        # aggregate and normalize
        probs = list(rtts.values())
        probs_sum = sum(probs)
        cdf_probs = []
        rtts = dict(sorted(rtts.items()))
        probs = list(rtts.values())
        #print(probs_sum)
        for r in rtts:
            # normalize
            prob_norm = rtts[r]/probs_sum
            #print(rtts[r])
            #print(prob_norm)
            # aggregate
            #for c in cdf_probs:
            #    prob_norm += c
            if len(cdf_probs) > 0:
                prob_norm += cdf_probs[-1]
            if prob_norm > 1:
                prob_norm = 1
            cdf_probs.append(prob_norm)
            #print(cdf_probs)


        keys = list(rtts.keys())

        percentiles_y = [ i/100 for i in range(5,96)]
        sampled_percentiles_x = []
        for y in percentiles_y:
            z = min(cdf_probs, key=lambda x:abs(x-y))
            sampled_percentiles_x.append(keys[cdf_probs.index(z)]/1000000)

        # max horizontal distance
        for sv in range(len(sampled_percentiles_x)):
            #err1 = abs(math.log2(sampled_percentiles_x[sv]/self.ground_truth[sv]))
            err1 = abs((sampled_percentiles_x[sv]-self.ground_truth[sv])/self.ground_truth[sv])
            errs.append(err1)

        print(max(errs))
        return max(errs)

        '''
        # OLD, area between
        # linear interpolation to get continuouos func
        cdf_sample = interpolate.interp1d(list(rtts.keys()), cdf_probs)

        keys = list(rtts.keys())
        #print(keys[0])
        #print(self.ground_truth([keys[0]]))
        #print(cdf_probs[0])
        gt_cdf = self.ground_truth(list(rtts.keys()))
        testvals = np.zeros((len(list(rtts.keys())),2))
        testvals[:,0] = [x/1000000000 for x in list(rtts.keys())]
        testvals[:,1] = gt_cdf

        # compute cdf on some values to compare to ground truth
        test_yvals = []
        for x in list(rtts.keys()):

        sv = np.zeros((len(list(rtts.keys())),2))
        sv[:,0]=[x/1000000000 for x in list(rtts.keys())]
        sv[:,1]=cdf_probs

        print(similaritymeasures.area_between_two_curves(testvals, sv))
        # find area between ground truth and sampled
        return similaritymeasures.area_between_two_curves(testvals, sv)
        '''

    def init_iteration(self, symbs):
        pass

#o = Opt("univ1_pt1.pcap")
#o.gen_traffic()
#m = pickle.load(open('rtt_correction.txt','rb'))
#o.calc_cost([m])

o = Opt("/media/data/mh43/Lucid4All/traces/univ_pcap/univ1_pt1.pcap")
o.gen_traffic()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["rtt_correction.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)


