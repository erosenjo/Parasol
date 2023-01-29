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

# creating sep ones for cms, bc apparently can't have same for both?
def log_miss_cms(key):
    misses[0]+=1
    '''
    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)
    '''

def log_hit_cms(key):
    hits[0]+=1
    '''
    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)
    '''


# separate externs for hash
def log_miss_hash(key):
    misses[0]+=1

def log_hit_hash(key):
    hits[0]+=1

def write_to_file_cms():
    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)

    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)

def write_to_file_precision():
    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)

    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)

def write_to_file_hash():
    print("HASH WRITE")
    with open('misses.pkl','wb') as f:
        pickle.dump(misses[0],f)

    with open('hits.pkl','wb') as f:
        pickle.dump(hits[0],f)



