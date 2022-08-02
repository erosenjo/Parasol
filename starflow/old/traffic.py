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
def gen_traffic_events(pkts):
    '''
    info = {}
    info["switches"] = 1
    info["max time"] = 9999999
    info["default input gap"] = 100
    info["random seed"] = 0
    info["python file"] = "starflow.py"
    '''
    events = []
    with PcapReader(pkts) as pcap_reader:
        for pkt in pcap_reader:
            if not (pkt.haslayer(IP)):
                continue
            src_int = int(hexadecimal(pkt[IP].src),0)
            dst_int = int(hexadecimal(pkt[IP].dst),0)
            pktlen = pkt[IP].len
            tos = pkt[IP].tos
            args = [128, src_int, dst_int, pktlen, tos]
            p = {"name":"ip_in", "args":args}
            events.append(p)
            if len(events) > 2000:
                break

    '''
    info["events"] = events
    with open('cms_sym.json', 'w') as f:
        json.dump(info, f, indent=4)

    #return ground_truth = [0]*100
    return len(events) 
    '''
    return len(events), events


# write json file w events
# call this before we run interp each time, bc num of free_block events may change each time
def gen_traffic(events, l_slots):
    info = {}
    info["switches"] = 1
    info["max time"] = 9999999
    info["default input gap"] = 100
    info["random seed"] = 0
    info["python file"] = "starflow.py"
    fb_events = []
    for i in range(1,l_slots):
        fb_events.append({"name":"free_block","args":[i,0]})
    
    info["events"] = fb_events+events

    with open('starflow.json','w') as f:
        json.dump(info, f, indent=4)

#exact = gen_traffic("univ1_pt1.pcap")
#print(exact)


#e,evtest = gen_traffic_events("univ1_pt1.pcap")
#gen_traffic(evtest,32)


