import pickle

counts = {}
fid3_mapping = {}

def log_count(srca, dsta, sport, dport, fid3, c, oldsrca, olddsta, oldfid3):

    if srca==0 and dsta==0 and sport==0 and dport==0 and fid3==0 and c==0 and oldsrca==0 and olddsta==0 and oldfid3==0:
        return

    if fid3 not in fid3_mapping:
        fid3_mapping[fid3] = (sport, dport)

    fid = str(srca)+str(dsta)+str(sport)+str(dport)

    if oldsrca!=0 or olddsta!=0 or oldfid3!=0:    # eviction + recirc event
        oldsport = fid3_mapping[oldfid3][0]
        olddport = fid3_mapping[oldfid3][1]
        oldfid = str(oldsrca) + str(olddsta) + str(oldsport) + str(olddport)
        # if we evict a key, set its count to 0 bc it's no longer in the struct
        counts[oldfid] = 0

    if fid not in counts:
        if c==0:
            counts[fid] = 1
        else:
            counts[fid] = c

    elif c==0:
        counts[fid] += 1

    else:
        counts[fid] = c

    #print("PACKET COUNT", pkt_counter[0])
    #with open('test.pkl','wb') as f:
    #    pickle.dump(pkt_counter,f)

    # ok this is ridiculous we simply cannot write after every pkt
    # gonna try after last pkt?
    # TODO: this feels like cheating a little, can we have a better way of know when we've reached the end?
    '''
    if pkt_counter[0] > 976545:
        with open('counts.pkl','wb') as f:
            pickle.dump(counts,f)    
    '''
    #print(counts)


def write_to_file():
    with open('counts.pkl','wb') as f:
        pickle.dump(counts,f)

