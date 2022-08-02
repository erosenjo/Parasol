import json

skew_bad = json.load(open("badtraces/time_cmscacheskew.json"))
sb = skew_bad["events"]
nskew_bad = json.load(open("badtraces/time_cmscachenotskew.json")) 
nsb = nskew_bad["events"]

skew = json.load(open("cachingskew.json"))
s = skew["events"]
nskew = json.load(open("cachingnotskew.json"))
ns = nskew["events"]

counter = 0
for se in s:
    se["args"].append(sb[counter]["args"][-1])
    counter += 1

counter = 0
for nse in ns:
    nse["args"].append(nsb[counter]["args"][-1])
    counter += 1

skew["events"] = s
nskew["events"] = ns

with open("newskew.json",'w') as f:
    json.dump(skew,f,indent=4)

with open("newnotskew.json",'w') as f:
    json.dump(nskew,f,indent=4)

