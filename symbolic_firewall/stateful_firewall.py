# track 
# track number of rtt samples we get and record how many inserts happen during each sample
import pickle
measurements = {
    "authorized_flows":set(),
    "insert":[],
    "cycle":[],
    "delete":[],
    "access":[],
    "busy":[],
    "counters": {
        "data_pkts":0,
        "recirc_pkts":0,
        "correct_pkts":0,
        "incorrect_pkts":0,
        "incorrect_bits":0,
        "early_delete_pkts":0
    }
}

def count_data_packet():
    measurements["counters"]["data_pkts"] += 1

def count_recirc_packet():
    measurements["counters"]["recirc_pkts"] += 1

def insert_attempt(src, dst, success):
    # every insert attempt should be a real insert
    # print ("adding authorized flow: (%s,%s)"%(src, dst))
    measurements["authorized_flows"].add((src, dst))

def log_insert(src, dst, time):
    pass
    measurements["insert"].append((src, dst, time))
def log_cycle(src, dst, time):
    pass
    measurements["cycle"].append((src, dst, time))
def log_delete(src, dst, time):
    pass
    measurements["delete"].append((src, dst, time))
def log_access(src, dst, time):
    pass
    measurements["access"].append((src, dst, time))
def log_busy(src, dst, time):
    pass
    measurements["busy"].append((src, dst, time))
def log_arrival(src, dst, time):
    pass
    measurements["arrivals"].append((src, dst, time))


def decide(src, dst, time, permitted, pktlen):
    reverse_key = (dst, src)
    should_permit = False
    # 1. figure out correct decision
    if (reverse_key in measurements["authorized_flows"]):
        should_permit = True
    # 2. increment correct or incorrect counter accordingly
    if (should_permit and permitted):
        measurements["counters"]["correct_pkts"] += 1
    else:
        measurements["counters"]["incorrect_pkts"] += 1
        measurements["counters"]["incorrect_bits"] += (pktlen * 8)
        # if incorrect, why? 
        for (s, d, t) in measurements["delete"]:
            if (d == src and s == dst and t < time):
                measurements["counters"]["early_delete_pkts"] += 1
                break

def write_logs():
    print("------final event stats------")


    with open('stateful_firewall_out_trace.pkl','wb') as f:
        pickle.dump(measurements,f)