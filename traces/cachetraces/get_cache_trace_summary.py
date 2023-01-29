import json, sys

trace = json.load(open(sys.argv[1],'r'))

keys = {}

for pkt in trace["events"]: 
    key = pkt["args"][0]

    if key in keys:
        keys[key] += 1
    else:
        keys[key] = 1

key_max = max(keys, key=keys.get)
req_max = keys[key_max]

print("UNIQUE KEYS:", len(keys.keys()))
print("MAX REQUESTS FROM KEY:", req_max)



