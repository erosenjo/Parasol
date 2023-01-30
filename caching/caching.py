# count the cache hits and misses for precision-based cache 
# we have access to the keys for each hit and miss, but for now just counting totals

import pickle

misses = [0]
hits = [0]

def log_miss(key):
    misses[0]+=1
    '''
    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)
    '''

def log_hit(key):
    hits[0]+=1
    '''
    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)
    '''

def write_to_file():
    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)

    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)



