from scipy import interpolate
import json

class Opt:
    def __init__(self, pktpcap):
        self.ground_truth = None
        self.pkts = pktpcap

    def gen_traffic(self):
        info = {}
        info["switches"] = 1
        info["max time"] = 9999999
        info["default input gap"] = 100
        info["random seed"] = 0
        info["python file"] = "fridge.py"

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
                args = [f_int, pktlen, ihl, offset, seq, src_int, dst_int, sport, dport, timestamp, ack]
                p = {"name":"tcp_in", "args":args}
                events.append(p)


                if f_int==2:
                    tmp4 = ihl*4+offset*4
                    tmp5 = pktlen-tmp4
                    eack = seq+tmp5
                    eack = eack+1

                    key = str(src_int)+str(dst_int)+str(sport)+str(dport)+str(eack)

                    syns[key] = timestamp

                elif f_int==18:
                    key = str(dst_int)+str(src_int)+str(dport)+str(sport)+str(ack)
                    if key in syns:
                        gt.append(timestamp-syns[key])

                if len(events) > 200:
                    break

        info["events"] = events
        ecdf = ECDF([x/1000000 for x in gt])
        # x vals go from 0 to max rtt in our data set???
        self.cdf_xvals = [0]+[x/1000000 for x in gt]
        gt_cdf = ecdf(self.cdf_xvals)
        self.ground_truth=np.zeros((len(self.cdf_xvals),2))
        self.ground_truth[:,0]=self.cdf_xvals
        self.ground_truth[:,1] = gt_cdf

    # measurement is dictionary of rtts and correction factors
    def calc_cost(self, measure):
        rtts = measure[0]
        # aggregate and normalize
        probs = list(rtts.values())
        probs_sum = sum(probs)
        cdf_probs = []
        for r in rtts:
            # normalize
            prob_norm = rtts[r]/probs_sum
            # aggregate
            for c in cdf_probs:
                prob_norm += c
            cdf_probs.append(prob_norm)

        # linear interpolation to get continuouos func
        cdf_sample = interpolate.interp1d(list(rtts.keys()), cdf_probs)

        # compute cdf on some values to compare to ground truth
        test_yvals = []
        for x in self.cdf_xvals:
            test_yvals.append(cdf_sample(x))

        sv = np.zeros((len(self.cdf_xvals),2))
        sv[:,0]=self.cdf_xvals
        sv[:,1]=test_yvals

        # find area between ground truth and sampled
        return similaritymeasures.area_between_two_curves(self.ground_truth, sv)


    def init_iteration(self):
        pass

#o = Opt("univ1_pt1.pcap")
#o.calc_cost([{1:0.9, 2: 0.8}])



