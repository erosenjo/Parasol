# gen traffic
# compute zipf, 1000 unique items, param of 2 and 3?
# for each item in result, gen pkts for each request
# if 10 req for item, gen 10 pkts
# might need to scale up reqs
# shuffle up pkts so requests for items are interleaved


import numpy as np
import pickle
import random
import json


# reqs generated using np.random.zipf(4, 1000) for not skewed and zipf(1.8,1000) for skew

skewed = pickle.load(open('skewed.txt','rb'))
skewed = [s*10 for s in skewed]
notskew = pickle.load(open('notskew.txt','rb'))
notskew = [n*70 for n in notskew]
#print(skewed)
#quit()

notskew.sort()
print(notskew)

quit()
info = {}
info["switches"] = 1
info["max time"] = 9999999
info["default input gap"] = 100
info["random seed"] = 0
info["python file"] = "precision.py"

global_time = 0

def get_events(reqs):
    ev = []
    k = 1
    for s in reqs:
        for p in range(s):
            # choose random timestamp, between 0ms (0ns)) and 1ms (1000000ns) from last time
            #timestamp = global_time + int(random.uniform(0,1)*1000000)
            ev.append({"name":"ip_in", "args":[k,0,0,0]})
        k += 1
        #break
    return ev

events_skew = get_events(skewed)
random.shuffle(events_skew)

# assign timestamp to each pkt
# choose random timestamp, between 0ms (0ns)) and 1ms (1000000ns) from last time
counter = 0
for e in events_skew:
    timestamp = global_time + int(random.uniform(0,0.01)*1000000)
    global_time = timestamp
    e["args"].append(timestamp)
    #print(timestamp)
    #print(global_time)
    #print("NEXT")
    counter += 1

# take timestamps from existing trace (caida)

events_notskew = get_events(notskew)
random.shuffle(events_notskew)

counter = 0
global_time = 0
for e in events_notskew:
    timestamp = global_time + int(random.uniform(0,0.01)*1000000)
    global_time = timestamp
    e["args"].append(timestamp)
    #print(timestamp)
    #print(global_time)
    #print("NEXT")
    counter += 1

#print(len(events_notskew))
#quit()
'''
info["events"] = events_skew
with open('precisionskew.json','w') as f:
    json.dump(info,f,indent=4)
'''

info["events"] = events_notskew
with open('precisionnotskew.json','w') as f:
    json.dump(info,f,indent=4)


'''
info["events"] = events_notskew
with open('cmscachetestnotskew.json','w') as f:
    json.dump(info,f,indent=4)
'''

