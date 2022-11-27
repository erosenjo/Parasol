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

        self.ground_truth = np.load("reqfiles_dc/conquestgt.npy")
        self.qlen_arr = np.load("reqfiles_dc/qlen_arr.npy")
        self.vect_qlong = np.load("reqfiles_dc/vect_qlong.npy")
        self.tracepkts = np.load("reqfiles_dc/tracepkts.npy")
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

                '''
                if src_int == 12032676479472000745:
                    src_int = 12032676475
                if dst_int == 12535319967083608017:
                    dst_int = 1253531997
                if src_int == 12535315807007091835:
                    src_int = 1253531585
                if src_int == 11961444146725396848:
                    src_int = 119614448
                if dst_int == 12535319967353978895:
                    dst_int = 1253531995
                if src_int == 12032676479469912034:
                    src_int = 1203267644
                if src_int == 12032676479472025606:
                    src_int = 120326766
                if dst_int == 12535319967083608016:
                    dst_int = 1253531906
                if src_int == 12032676479637585436:
                    src_int = 1203267646
                if src_int == 12535319966947295125:
                    src_int = 1253531966
                if dst_int == 12032676479204466929:
                    dst_int = 120326929
                if dst_int == 12032676479203507957:
                    dst_int = 120326957
                if src_int == 12032676479634480874:
                    src_int = 120326874
                if src_int == 12535319967083608053:
                    src_int = 1253538053
                if dst_int == 12032676479204499742:
                    dst_int = 12032742
                if src_int == 12032676479472033433:
                    src_int = 12032433
                if src_int == 12032676479469916188:
                    src_int = 12032188
                if src_int == 12032676479469940583:
                    src_int = 12032583
                if src_int == 12032676479634464278:
                    src_int = 12032278
                if dst_int == 12535319967353978893:
                    dst_int = 1253531893
                if src_int == 12032676479637577578:
                    src_int = 120326578
                if dst_int == 12032676479204499230:
                    dst_int = 120326230
                if src_int == 12535319967083608054:
                    src_int = 1253531054
                if src_int == 12032676479637565542:
                    src_int = 12032542
                if src_int == 12535319967083608055:
                    src_int = 1253531055
                if src_int == 12180531642118256664:
                    src_int = 121805664
                if dst_int == 762084771789878161:
                    dst_int = 762084161
                if src_int == 12535342678898689011:
                    src_int = 125353011
                if src_int == 12535319967083608050:
                    src_int = 1253531050
                if src_int == 12535319967083607997:
                    src_int = 125353997
                if src_int == 12535319967083608058:
                    src_int = 125353058
                if src_int == 12032967233313817498:
                    src_int = 120329498
                if dst_int == 12535319966685065323:
                    dst_int = 125353323
                if src_int == 12535319966951460848:
                    src_int = 12535848
                if dst_int == 12180531616619295720:
                    dst_int = 121805720
                if src_int == 12318744102555500276:
                    src_int = 123187276
                if src_int == 761282302629247993:
                    src_int = 761282993
                if dst_int == 12535319800658326499:
                    dst_int = 125353499
                if src_int == 12180514035597899752:
                    src_int = 121805752
                if src_int == 12032676479471992925:
                    src_int = 120326925
                if dst_int == 12319309325620548571:
                    dst_int = 12319571
                if src_int == 761283220342937992:
                    src_int = 761283992
                if src_int == 12032920321643329430:
                    src_int = 12032430
                if src_int == 12032676480138763899:
                    src_int = 12032899
                if src_int == 12318355918839399425:
                    src_int = 123183425
                if dst_int == 752042279940978572:
                    dst_int = 75204572
                if src_int == 12535319807374005269:
                    src_int = 12535269
                if dst_int == 12535319808673371056:
                    dst_int = 125353056
                if dst_int == 12535319808673370943:
                    dst_int = 12535943
                if src_int == 12535319796861293546:
                    src_int = 12535546
                if src_int == 12032625956492344461:
                    src_int = 120326461
                if src_int == 12032676480138960499:
                    src_int = 120326499
                if src_int == 11961444136698910998:
                    src_int = 11961998
                if src_int == 12535327497294772231:
                    src_int = 12535231
                if dst_int == 783457458657390360:
                    dst_int = 78345360
                if src_int == 11961444145305782228:
                    src_int = 11961228
                if src_int == 12032676480138961172:
                    src_int = 12032172
                if src_int == 12535319518800431211:
                    src_int = 12535211
                if src_int == 12535319236672632943:
                    src_int = 125352943
                if src_int == 11961444149019821418:
                    src_int = 11961418
                if src_int == 11961444145444328423:
                    src_int = 11961423
                if src_int == 11961444138418801767:
                    src_int = 119611767
                if src_int == 11961444146207419401:
                    src_int = 11961401
                if src_int == 11961444138840066596:
                    src_int = 11961596
                if src_int == 12319309325620532179:
                    src_int = 12319179
                if dst_int == 12193356827185001493:
                    dst_int = 121933493
                if src_int == 11961444154145338144:
                    src_int = 11961144
                if src_int == 12535319098968622076:
                    src_int = 12535076
                if dst_int == 761497065489398908:
                    dst_int = 76149908
                if src_int == 12535317132156717182:
                    src_int = 12535182
                if src_int == 11961444138401804439:
                    src_int = 11964439
                if src_int == 11961439327132565113:
                    src_int = 11961113
                if src_int == 12535332006013843427:
                    src_int = 12535427
                if dst_int == 12032676479203491347:
                    dst_int = 12032347
                if src_int == 12535319967083608057:
                    src_int = 12538057
                if src_int == 12033377386998385641:
                    src_int = 120333641
                if src_int == 12534890221118084492:
                    src_int = 12534492
                if dst_int == 752042279940978586:
                    dst_int = 75204586
                if src_int == 12319189410607201240:
                    src_int = 123191240
                if src_int == 12535320041781800977:
                    src_int = 12535977
                if dst_int == 12193356827185315734:
                    dst_int = 12193734
                if src_int == 11960183538686375838:
                    src_int = 11960838
                if dst_int == 12193356566046259970:
                    dst_int = 12193970
                if src_int == 783458689199061292:
                    src_int = 78345292
                if dst_int == 12193356827185856387:
                    dst_int = 12193387
                if src_int == 12193356866378857469:
                    src_int = 12193469
                if dst_int == 761376679822135218:
                    dst_int = 76137218
                if src_int == 761283226613893221:
                    src_int = 761283221
                if dst_int == 12193356345172239431:
                    dst_int = 12193431
                if dst_int == 2974157483010309:
                    dst_int = 29741309
                if src_int == 12180109762349888618:
                    src_int = 121801618
                if dst_int == 47584021866692271:
                    dst_int = 47584271
                if src_int == 12180108392416711694:
                    src_int = 12180694
                if src_int == 12033173831852868583:
                    src_int = 12033583
                if src_int == 12033170912835662824:
                    src_int = 12033824
                if src_int == 12033171457944109155:
                    src_int = 12033155
                if src_int == 783458689199061277:
                    src_int = 78345277
                if dst_int == 12535319800656326640:
                    dst_int = 12535640
                if src_int == 783458395230292983:
                    src_int = 78345983
                if src_int == 12535319967083607988:
                    src_int = 125353988
                if dst_int == 12032929010128319571:
                    dst_int = 120329571
                if src_int == 783456528933926786:
                    src_int = 783456786
                if src_int == 783458689199061240:
                    src_int = 783458240
                if src_int == 12535319966685058034:
                    src_int = 125353034
                if dst_int == 12535319966685058034:
                    dst_int = 125353034
                if src_int == 12535319967083120791:
                    src_int = 12535791
                if dst_int == 783457487541305231:
                    dst_int = 78345231
                if src_int == 12535316458900477951:
                    src_int = 12535951
                if src_int == 12319165840269127065:
                    src_int = 12319065
                if dst_int == 12032676479204466930:
                    dst_int = 120326930
                if src_int == 783458110288873845:
                    src_int = 78345845
                if dst_int == 12180531616619295114:
                    dst_int = 121805114
                if src_int == 12180114385343014960:
                    src_int = 12180960
                if src_int == 12180114386282513319:
                    src_int = 12180319
                if src_int == 48966073859972627:
                    src_int = 48966627
                if dst_int == 12032676479203491862:
                    dst_int = 120326862
                if src_int == 12535320001566921321:
                    src_int = 125353321
                if src_int == 12032923011396008836:
                    src_int = 12032836
                if src_int == 12535319967083608035:
                    src_int = 12535035
                if dst_int == 783452023981734090:
                    dst_int = 78345090
                if dst_int == 12183952788117911573:
                    dst_int = 121831573
                if src_int == 12535319967622427658:
                    src_int = 1253531658
                if dst_int == 783452024487283739:
                    dst_int = 783452739
                if dst_int == 12535232389147670655:
                    dst_int = 12535655
                if src_int == 12318703365734399995:
                    src_int = 12318995
                if dst_int == 12183952788117714023:
                    dst_int = 121834023
                if src_int == 12180531525485540350:
                    src_int = 12180350
                if dst_int == 12180531616619295900:
                    dst_int = 12180900
                if src_int == 12535325498066157531:
                    src_int = 12537531
                if src_int == 12535327327587269743:
                    src_int = 12535743
                if src_int == 12535320041781801441:
                    src_int = 12531441
                if dst_int == 12535319967083121257:
                    dst_int = 12535257
                if dst_int == 12535319967083121254:
                    dst_int = 12535254
                if src_int == 12535327497297868259:
                    src_int = 1253259
                if dst_int == 783457833527079903:
                    dst_int = 78349903
                if src_int == 12535320783393730566:
                    src_int = 12535566
                if src_int == 783457502669711881:
                    src_int = 78311881
                if dst_int == 12032676479204499400:
                    dst_int = 12032400
                if src_int == 12535316459841978249:
                    src_int = 125378249
                if dst_int == 12032676479204499738:
                    dst_int = 120399738
                if dst_int == 12535319800660804709:
                    dst_int = 125354709
                if src_int == 761282127224880142:
                    src_int = 76128142
                if src_int == 12180531595815220241:
                    src_int = 12220241
                if dst_int == 752058063134914933:
                    dst_int = 75214933
                if src_int == 12180531595815220243:
                    src_int = 12120243
                if src_int == 12535319807374005357:
                    src_int = 125355357
                if dst_int == 12535319808673371121:
                    dst_int = 125351121
                if src_int == 12535333058815014896:
                    src_int = 125354896
                if src_int == 752057270719603768:
                    src_int = 75203768
                if dst_int == 12535340539043563812:
                    dst_int = 12533812
                if src_int == 12535319967085182825:
                    src_int = 12532825
                if dst_int == 761497049257359483:
                    dst_int = 76149483
                if src_int == 12180531635011044365:
                    src_int = 12180365
                if src_int == 12180514054500107257:
                    src_int = 12180257
                if src_int == 11961444138953496195:
                    src_int = 11966195
                if src_int == 11961444138953538031:
                    src_int = 11961031
                if src_int == 12535319967083608049:
                    src_int = 125358049
                if dst_int == 12183952788117911571:
                    dst_int = 12181571
                if src_int == 12535319967083515009:
                    src_int = 125355009
                if dst_int == 12180531616619294877:
                    dst_int = 12184877
                if src_int == 12535319800658381797:
                    src_int = 12535797
                if src_int == 12317706711996613759:
                    src_int = 12313759
                if dst_int == 12032676479204495119:
                    dst_int = 120325119
                '''


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
                if len(events)>2000000:
                    break

        if len(events)==1021724:
            print("MAX EVENTS")
            events = events[:-1]
            pkt_counter -= 1
        #pkt_counter = 1021723

        #pkt_counter = 300001

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
#o = Opt("univ1_pt1.pcap")
o = Opt("equinix-chicago.dirA.20160121-125911.UTC.anon.pcap")
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


