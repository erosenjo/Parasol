# count the cache hits and misses for precision-based cache 
# we have access to the keys for each hit and miss, but for now just counting totals

import pickle

misses = [0]
hits = [0]

def log_miss(key):
    misses[0]+=1
    with open('5misses.txt','wb') as f:
        pickle.dump(misses[0],f)


def log_hit(key):
    hits[0]+=1
    with open('5hits.txt','wb') as f:
        pickle.dump(hits[0],f)


# creating sep ones for cms, bc apparently can't have same for both?
def log_miss_cms(key):
    misses[0]+=1
    with open('5misses.txt','wb') as f:
        pickle.dump(misses[0],f)


def log_hit_cms(key):
    hits[0]+=1
    with open('5hits.txt','wb') as f:
        pickle.dump(hits[0],f)

