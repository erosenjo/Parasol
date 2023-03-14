# track number of rtt samples we get and record how many inserts happen during each sample
import pickle
samples = [0]
insert_measure = []
rtt_correction = {}
# rtts collected
raw_rtt = []
# ids of collected rtts
sample_ids = []
# accept is the number of syn pkts where entropy = 0 (i.e., added to the fridge)
accept_counter = [0]
# ids of syns that get accepted
accept_ids = []
# reject is the number of syn pkts where entropy != 0 (i.e., NOT added to fridge)
reject_counter = [0]
# number of syn packets
syn_counter = [0]
# number of other pkts that are not syn (pkt_type = 2) but can still trigger insert to fridge
otherpkt_counter = [0]
# every time we add a pkt to the fridge, subtract 1 from empty regs
# every time we collect a sample, add 1 to empty regs
# (we can get a negative number bc we can have collisions --> we overwrite when we add to fridge)
empty_regs = [2048]
# measure how many fridge collisions
# these occur when the id of the (syn) ack packet don't match the id at the hashed location in the fridge
# (either bc we've already collected the sample or we've overwritten it)
collisions = [0]

# insrtdiff is the number of pkts inserted in fridge between seq and ack
def log_rttsample(rtt, insrtdiff, entry_prob, array_size, src_uint, dst_uint, sport, dport, ack):
    samples[0]+=1
    insert_measure.append(insrtdiff)

    empty_regs.append(empty_regs[-1]+1)

    entry_prob = 2**-entry_prob

    # correction = p^-1 * (1 - p/m) ^ -x
    #p = insert prob, M = size of array, x = # insertions
    correction_factor1 = entry_prob**-1
    correction_factor2 = (1-entry_prob/array_size)**-insrtdiff
    correction_factor = correction_factor1 * correction_factor2

    if rtt in rtt_correction:
        rtt_correction[rtt] += correction_factor
    else:
        rtt_correction[rtt] = correction_factor

    raw_rtt.append(rtt)
    sample_ids.append(str(dst_uint)+str(src_uint)+str(dport)+str(sport)+str(ack))

    '''
    with open('numsamples.pkl','wb') as f:
        pickle.dump(samples[0],f)
    with open('insertdiffs.pkl','wb') as f:
        pickle.dump(insert_measure,f)

    with open('rtt_correction.pkl','wb') as f:
        pickle.dump(rtt_correction,f)
    '''

def log_synpkt_entropy(entropy, pkt_type, src_uint, dst_uint, sport, dport, eack):
    if entropy==0:
        accept_counter[0]+=1
        empty_regs.append(empty_regs[-1]-1)
        accept_ids.append(str(src_uint)+str(dst_uint)+str(sport)+str(dport)+str(eack))
    else:
        reject_counter[0]+=1

    if pkt_type==1:
        syn_counter[0]+=1
    else:
        otherpkt_counter[0]+=1

def log_collision():
    collisions[0]+=1

def write_to_file():
    with open('numsamples.pkl','wb') as f:
        pickle.dump(samples[0],f)
    with open('insertdiffs.pkl','wb') as f:
        pickle.dump(insert_measure,f)

    with open('rtt_correction.pkl','wb') as f:
        pickle.dump(rtt_correction,f)

    with open('accepts.pkl','wb') as f:
        pickle.dump(accept_counter,f)
    with open('acceptids.pkl','wb') as f:
        pickle.dump(accept_ids,f)
    with open('rejects.pkl','wb') as f:
        pickle.dump(reject_counter,f)
    with open('syns.pkl','wb') as f:
        pickle.dump(syn_counter,f)
    with open('otherpkts.pkl','wb') as f:
        pickle.dump(otherpkt_counter,f)

    with open('empty_regs.pkl','wb') as f:
        pickle.dump(empty_regs,f)
    with open('collisions.pkl','wb') as f:
        pickle.dump(collisions,f)

    with open('rawrttsamples.pkl','wb') as f:
        pickle.dump(raw_rtt,f)
    with open('sampleids.pkl','wb') as f:
        pickle.dump(sample_ids,f)


