import pickle

counts = {}
pkt_counter = [0]

def log_count(srca,dsta,sport,dport,c):
    pkt_counter[0] += 1
    fid = str(srca)+str(dsta)+str(sport)+str(dport)

    if fid not in counts:
        counts[fid] = c

    elif c==0:
        counts[fid] += 1

    else:
        counts[fid] = c


    #with open('test.pkl','wb') as f:
    #    pickle.dump(pkt_counter,f)

    # ok this is ridiculous we simply cannot write after every pkt
    # gonna try after last pkt?
    # TODO: this feels like cheating a little, can we have a better way of know when we've reached the end?
    if pkt_counter[0] > 665590:
        with open('counts.pkl','wb') as f:
            pickle.dump(counts,f)    


