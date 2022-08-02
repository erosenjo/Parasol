import pickle

counts = {}


def log_count(srca,dsta,sport,dport,c):
    fid = str(srca)+str(dsta)+str(sport)+str(dport)

    if fid not in counts:
        counts[fid] = c

    elif c==0:
        counts[fid] += 1

    else:
        counts[fid] = c


    with open('counts.txt','wb') as f:
        pickle.dump(counts,f)    


