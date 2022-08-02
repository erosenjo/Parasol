import json
from scapy.all import *

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

# create pkt events (but don't write json yet, need to do that for each interp run in starflow)
# return ground truth (num pkts for starflow) and list of pkt events
def gen_traffic(pkts):
    info = {}
    info["switches"] = 1
    info["max time"] = 9999999
    info["default input gap"] = 100
    info["random seed"] = 0
    info["python file"] = "fridge.py"
    events = []
    starttime = 1261067164.398500
    with PcapReader(pkts) as pcap_reader:
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
            timestamp = int((pkt.time-starttime)*1000000000)
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
            args = [f_int, pktlen, ihl, offset, seq, src_int, dst_int, sport, dport, ack, timestamp]
            p = {"name":"tcp_in", "args":args}
            events.append(p)
            if f_int==18:
                print(f_str)
            if len(events) > 200:
                #print(int((pkt.time-starttime)*1000000000))
                break
            #print(int((pkt.time-starttime)*1000000000))

    info["events"] = events
    with open('fridge.json', 'w') as f:
        json.dump(info, f, indent=4)

    #return ground_truth = [0]*100
    return len(events) 


#exact = gen_traffic("univ1_pt1.pcap")
#print(exact)


gen_traffic("univ1_pt1.pcap")

