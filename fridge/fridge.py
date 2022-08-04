# track number of rtt samples we get and record how many inserts happen during each sample
import pickle
samples = [0]
insert_measure = []
rtt_correction = {}

# insrtdiff is the number of pkts inserted in fridge between seq and ack
def log_rttsample(rtt, insrtdiff, entry_prob, array_size):
    samples[0]+=1
    insert_measure.append(insrtdiff)

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

    with open('numsamples.pkl','wb') as f:
        pickle.dump(samples[0],f)
    with open('insertdiffs.pkl','wb') as f:
        pickle.dump(insert_measure,f)

    with open('rtt_correction.pkl','wb') as f:
        pickle.dump(rtt_correction,f)

