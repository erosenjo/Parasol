import json
from scapy.all import *
import similaritymeasures
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
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
        self.cdf_xvals = []

    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 999999999
        info["default input gap"] = 10
        info["random seed"] = 0
        info["python file"] = "rtt.py"
        events = []
        # univ1_pt1 start time
        #starttime = 1261067164.398500 # placeholder until figure out timestamps
        # univ1_pt8 start time
        #starttime = 1261069030.761055
        starttime = 0
        syns = {}
        gt = []
        ps = {}
        pkt_counter = 0
        rtts_orig = {}
        tests = {}

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

                added = False

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
                    key2 = str(dst_uint)+str(src_uint)+str(eth.ip.tcp.dport)+str(eth.ip.tcp.sport)+str(eack)

                    if key2 in ps:  # already got the match for this
                        ps.pop(key2)
                        #ps.pop(key)

                    if key in ps:
                        events.remove(ps[key])
                        ps[key] = p
                        added = True

                    else:
                        ps[key] = p
                        added = True

                    #if eack == 2211554069:
                        #print("EACK: "+str(timestamp))
                        #if key in ps:
                            #print("PS "+str(timestamp))


                    '''
                    if args==[24, 108, 5, 5, 389537286, 3974322786, 4093882517, 699466720, 1869, 9809, 5524034]:
                        print(f_str)
                        print("EACK: "+str(eack))
                        print("KEY: "+str(key))
                    '''
                    #if key not in syns:
                        #syns[key] = timestamp
                    #print("EACK: "  + str(eack))
                elif drop==False:
                    key = str(dst_uint)+str(src_uint)+str(eth.ip.tcp.dport)+str(eth.ip.tcp.sport)+str(eth.ip.tcp.ack)
                    key2 = str(src_uint)+str(dst_uint)+str(eth.ip.tcp.sport)+str(eth.ip.tcp.dport)+str(eth.ip.tcp.ack)
                    if key2 in ps:
                        events.remove(p)
                        #ps[key2] = p
                        added = True
                    else:
                        ps[key2] = p
                        added = True

                    #if ack == 2211554069:
                        #print("ACK: "+str(timestamp))
                        #if key2 in ps:
                            #print("PSACK "+str(timestamp))

                    '''
                    if args==[16, 40, 5, 5, 3974322862, 389537355, 699466720, 4093882517, 9809, 1869, 6418034]:
                        print("KEY: "+str(key))
                    '''

                    '''
                    if key in syns:
                        #print("MATCH!!")
                        gt.append(timestamp-syns[key])
                        tests[ack] = timestamp-syns[key]
                        syns.pop(key,None)
                    '''
                #if not added:
                #    ps[key] = p

                '''
                if eack == 1872138008:
                    print("EACK TIME: "+str(timestamp))

                if ack==1872138008:
                    print("ACK TIME: "+str(timestamp))
                '''


                # 1000000 pkts for training, full trace for testing
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
                #print(len(events))
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
                #print(timestamp)

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
                added = False
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
                    key2 = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(eack)
                    if key2 in ps:  # already got the match for this
                        ps.pop(key2)
                        #ps.pop(key)
                    if key in ps:
                        events.remove(ps[key])
                        ps[key] = p
                        added = True
                    else:
                        ps[key] = p
                        added = True
                    #if eack == 2211554069:
                        #print("EACK: "+str(timestamp))
                        #if key in ps:
                            #print("PS "+str(timestamp))

                    #if key not in syns:
                        #syns[key] = timestamp
                    #print("EACK: "  + str(eack))
                elif drop==False:
                    key = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(ack)
                    key2 = str(src_int)+str(dst_int)+str(sport)+str(dport)+str(ack)

                    if key2 in ps:
                        events.remove(p)
                        #ps[key2] = p
                        added = True
                    else:
                        ps[key2] = p
                        added = True
                    #if ack == 2211554069:
                        #print("ACK: "+str(timestamp))
                        #if key2 in ps:
                            #print("PSACK "+str(timestamp))

                #if not added:
                #    ps[key] = p

                # 1000000 pkts for training, full trace for testing
                if len(events) > 1000000:
                    break
        '''

        info["events"] = events
        with open('rtt.json', 'w') as f:
            json.dump(info, f, indent=4)


        for p in events:
            f_int = p["args"][0]
            pktlen = p["args"][1]
            ihl = p["args"][2]
            offset = p["args"][3]
            seq = p["args"][4]
            ack = p["args"][5]
            src_int = p["args"][6]
            dst_int = p["args"][7]
            sport = p["args"][8]
            dport = p["args"][9]
            timestamp = p["args"][10]

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

            if pkttype==1 and drop==False:
                tmp4 = ihl*4+offset*4
                tmp5 = pktlen-tmp4
                eack = seq+tmp5
                if f_int==2:
                    eack = eack+1
                key = str(src_int)+str(dst_int)+str(sport)+str(dport)+str(eack)
                syns[key] = timestamp

                #if eack==2211554069:
                    #print("EACK2 "+str(timestamp))


            elif drop==False:
                key = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(ack)

                if key in syns:
                    #print("MATCH!!")

                    if (timestamp-syns[key])<(2000*1000*1000):
                        gt.append(timestamp-syns[key])
                        tests[ack] = timestamp-syns[key]
                        if tests[ack] < 0:
                            print(ack)
                    #syns.pop(key,None)
                #if ack==2211554069:
                    #print("ACK2 "+str(timestamp))



        with open('gttest.pkl','wb') as f:
            pickle.dump(tests,f)
        
        print(len(gt))
        with open('gtnumsamples.pkl','wb') as f:
            pickle.dump(len(gt),f)

        '''
        # CDF RELATIVE %-ILE ERROR

        #self.ground_truth = len(events)
        #self.ground_truth = gt
        # compute cdf w/ ground truth: (converting to ms from ns)
        ecdf = ECDF([x/1000000 for x in gt])
  
        #print("ACT")
        #print(ecdf([161.627]))

        # x vals go from 0 to max rtt in our data set???
        # should be max in ms, not ns?????
        #self.cdf_xvals = range(0,max(gt)/1000000)
        # TODO: include 0??? idk, maybe; and maybe pad w/ max val
        self.cdf_xvals = [x/1000000 for x in gt]
        self.cdf_xvals.sort()
        self.cdf_xvals = list(set(self.cdf_xvals))

        #print(self.cdf_xvals)

        probs_gt = list(ecdf(self.cdf_xvals))

        percentiles_y = [ i/100 for i in range(5,96)]
        percentiles_x = []
        for y in percentiles_y:
            #print(y)
            z = min(probs_gt, key=lambda x:abs(x-y))
            #print(z)
            #print(self.cdf_xvals[probs_gt.index(z)])
            percentiles_x.append(self.cdf_xvals[probs_gt.index(z)])


        self.ground_truth = percentiles_x

        #self.ground_truth=np.zeros((len(self.cdf_xvals),2))
        #self.ground_truth[:,0]=self.cdf_xvals
        #self.ground_truth[:,1] = gt_cdf

        '''
        self.ground_truth = len(gt)


    # measurements are num collisions, num timeouts, num rtt samples
    # don't currently calc total num of rtt samples in trace, but can calc in gen_traffic
    def calc_cost(self,measure):

        # read success rate (num samples/gt num samples)
        nums = measure[0]

        print("1 - READ SUCCESS RATE", (1-nums/self.ground_truth))

        return (1-nums/self.ground_truth)

        '''
        # CDF RELATIVE %-ILE ERROR
        # measure is list of RTT samples, we're gonna generate CDF from them
        rtts = measure[0]
        print(len(rtts))

        with open('evalnumsamples.txt','ab') as f:
            pickle.dump(len(rtts),f)

        keys = [m/1000000 for m in rtts]
        #keys.sort()

        #print(keys)

        ecdf = ECDF(keys)
        x_val_keys = [i/1000000 for i in list(set(rtts))]
        x_val_keys.sort()
        samp_vals = list(ecdf(x_val_keys))

        percentiles_y = [ i/100 for i in range(5,96)]

        #print("EST")
        #print(ecdf([161.627]))


        sampled_percentiles_x = []
        for y in percentiles_y:
            #print(y)
            z = min(samp_vals, key=lambda x:abs(x-y))
            #print(z)
            #print(keys[samp_vals.index(z)])
            sampled_percentiles_x.append(x_val_keys[samp_vals.index(z)])


        errs = []
        # max horizontal distance
        for sv in range(len(sampled_percentiles_x)):
            #err1 = abs(math.log2(sampled_percentiles_x[sv]/self.ground_truth[sv]))
            #print(sampled_percentiles_x[sv])
            #print("EST")
            #print(self.ground_truth[sv])
            #print("GT")
            #print(percentiles_y[sv])
            err1 = abs((sampled_percentiles_x[sv]-self.ground_truth[sv])/self.ground_truth[sv])
            #print(err1)
            #print(sv)
            errs.append(err1)

        print(max(errs))
        return max(errs)

        '''
        '''
        sv = np.zeros((len(self.cdf_xvals),2))
        sv[:,0]=self.cdf_xvals
        sv[:,1]=samp_vals

        return similaritymeasures.area_between_two_curves(self.ground_truth, sv)
        '''

        


    def init_iteration(self,symbs):
        pass



# compute cdf as cost, diff between ideal and actual cdf is cost
# ground truth is all rtts observed?

# statsmodels ECDF?
# then get curve by doing ecdf([x vals]) - from some min rtt to some max rtt
# then use similaritymeasures.area_between_two_curves
# ecdf = ECDF([measurement])

o = Opt("/media/data/mh43/Lucid4All/traces/univ_pcap/univ1_pt8.pcap")
o.gen_traffic()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["numsamples.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
o.calc_cost(measurement)

#m = [pickle.load(open('samples.txt','rb'))]
#o.calc_cost(m)



