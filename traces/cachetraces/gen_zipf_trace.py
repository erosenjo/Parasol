import json, random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# entry event ip_in(int key, int<<1>> insert, int min_stg, int cacheval, int timestamp);
'''
{
    "switches": 1,
    "max time": 9999999,
    "default input gap": 100,
    "random seed": 0,
    "python file": "caching.py",
    "events": [
        {
            "name": "ip_in",
            "args": [
                830,
                0,
                0,
                0,
                3859
            ]
        },
'''
# skewed events: 87380
# not skewed events: 78400

# distr_param is float > 1, higher means more skewed, lower means more even
# not skew = 1.2
# not skew 2 = 1.3
# not skew 3 = 1.1
# not skew 4 = 1.05
# not skew 5 = 1.03
#distr_param = 1.2
#distr_param = 1.3
#distr_param = 1.1
distr_param = 1.05
#distr_param = 1.03
# skew = 1.4
#distr_param = 1.4
# large trace 80000000 events
# med traces 1000000 events
#num_events = 80000000
#num_events = 1000000
#num_events = 10000000
# num unique keys in medskew1mil = 27012
num_events = 1000000

key_list = list(np.random.zipf(distr_param, num_events))
#key_list = list(np.random.randint(1, high=27013, size=num_events))
# figure out number of unique keys
counts = Counter(key_list)
unique = list(counts.keys())
counts_y = []
for k in unique:
    counts_y.append(counts[k])

print("num unique:", len(unique))

#exit()

#plt.scatter(unique, counts_y)
#plt.show()

trace = {}
trace["switches"]=1
trace["max time"]=99999999999
trace["default input gap"]=100
trace["random seed"]=0
trace["python file"]="caching.py"
events = []
# basing timestamp values off of univ pcap trace
# on average, time between pkts was 517948 ns
timestamp = 0
key_dict = {}
key_counter = 1
for k in key_list:
    if int(k) not in key_dict:
        key_dict[int(k)] = key_counter
        key_counter += 1
    args = [key_dict[int(k)], 0,0,0,timestamp]
    e = {"name":"ip_in", "args":args}
    events.append(e)
    timestamp += random.randint(50000, 800000) 

# dummy packet
e = {"name":"ip_in", "args":[0,0,0,0,0]}
events.append(e)

trace["events"] = events
print(key_counter)

with open("mednotskew1mil1-05.json",'w') as f:
    json.dump(trace,f,indent=4)



