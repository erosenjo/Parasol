# Applications
This is a summary of the data structures implemented in Parasol, along with their associated (symbolic) parameters and optimization objective function.

## Bloom filter
Approximate set membership. Assume single register array.
### parameters:
- bits (+)
- number of hashes (?)
### optimization objective/measurement:
- minimize false positive rate (record query answer per packet and ground truth answers)
### parameter search
- max out bits, then find number of hashes (with fixed bits)

## [Caching](https://www.cs.cornell.edu/~jnfoster/papers/netcache.pdf)
A key value store that tracks popularity of uncached keys. Implemented as either a hash table (cache) and cms (key popularity), or with precision.
### parameters:
- cache entries (columns in reg array) (+ if precision/by itself, ? w/ cms)
- cache tables (num reg arrays) (+ if precision/by itself, ? w/ cms)
- cms columns (?, but + by itself)
- cms rows (?, but + by itself)
- cms expiring threshold (in ns) (?)
- cms heavy hitter threshold (?)
- boolean for structure choice (?)
### optimization objective/measurement:
- maximize hit rate (extern that keeps counter of hits/misses)
### parameter search
- if we know skewness, then choose data structure first; else, just optimize for each one separately, in parallel
- if precision: max number of counters (then max out num tables, then max out cols) (does this also depend on k?)
- if cms: cache size, fill up avail with cms, hh threshold, timeout 
### performance
- compile time: ~2min, ~3.5min
- interpreter time: (externs!!)

## CMS
Simple sketch for identifying heavy hitters. Can include timeout/count threshold as parameters (see caching example).
### parameters
- rows (+)
- columns (+)
###optimization objective/measurement:
- minimize mean absolute error (record approximate count for each packet and ground truth counts)
### parameter search
- for general heavy hitter identification app
- max cols, then max rows (with fixed cols) --> cols affect error rate of sketch, rows affect confidence
### performance
- compile time: ~3s 
- interpreter time: ~35s (500000 pkts, 55095 flows)


## [Conquest](https://www.cs.princeton.edu/~xiaoqic/documents/paper-ConQuest-CoNEXT19.pdf)
Queue length measurement in the data plane, to identify which flows are contributing to queue. Implemented as a series of sketches that are periodically rotated through.
### parameters:
- snapshots(H) (cms) (?, but + w/o resource constraints)
- rows in snapshot(R) (?, but + w/o resource constraints)
- columns in snapshot(C) (?, but + w/o resource constraints)
- time window(LOG\_CQ\_T) (in ns, determines how often we cycle through snaps) (?)
### optimization objective/measurement:
- maximize f-score (precision, recall) (extern records qlength estimation when qlength is large enough, compute how often we correctly identify a large queue and how good the estimation was using ground truth qlength -> see conquest paper/Danny's simulation code for more detail)
### parameter search
- time window is last
- max number of reg accesses?
### performance
- compile time: >22min (this is worst case, maxing out all values-32 sketches, 4 rows, 65536 cols), ~25s (realistic case) 
- interpreter time: ~32s (300001 pkts)


## [Flowlet switching](http://web.mit.edu/domino/domino-sigcomm.pdf)
Load-balancing application that sends flowlets on randomly chosen next hop, all packets in same flowlet will go to same hop. Assume single register array that saves the hop and most recent arrival time for each flowlet.
### parameters:
- flowlet slots (columns in reg array) (+)
- IPG threshold (in ns; packets with IPG smaller than threshold considered part of same flowlet) (?)
### optimization objective/measurement:
- minimize gap to even distribution (measure the pkts in bytes sent along each hop and compute error with an even distribution of bytes across each hop)
### parameter search
- give as many columns as possible, then find threshold with fixed size
- only compile once
### performance
- compile time: ~1s 
- interpreter time: ~45s (500000 pkts)


## [Fridge](https://www.cs.princeton.edu/~jrex/papers/apocs22.pdf)
Unbiased delay measurements. Hash-indexed array storing time of syn packets, to get matched when corresponding ack arrives. Entries added to fridge with probability p; aggressively overwrite and correct bias as samples collected. Assume single fridge (register array).
### parameters
- entry probability (?)
- fridge size (columns in reg array) (+)
### optimization objective/measurement:
- minimze max percentile error (measure rtt sample (in ns) and correction factor, compute cdf from measurements and ground truth, compute error at various percentiles -> see fridge paper for more details)
### parameter search
- choose (max) fridge size, then find probability (with fixed size)
- only compile once
### performance
- compile time: ~25s 
- interpreter time: ~30s (666096 pkts, 212377 samples) 


## Hash table
Simple multi-stage hash table. Assume entries are never evicted.
### parameters:
- tables (num reg arrays) (+)
- entries (per table) (columns in reg array) (+)
### optimization objective/measurement:
- minimize number of collisions (extern that keeps counter of collisions)
### parameter search
### performance


## [Precision](https://www.cs.princeton.edu/~xiaoqic/documents/paper-PRECISION-ToN.pdf)
Heavy hitter detection. Implemented as a series of tables that store flow key and counter. Probabilistically insert new entries upon collisions.
### parameters:
- tables (+)
- entries (columns in reg array) (+)
### optimization objective/measurement:
- minimize mean absolute error (same as cms)
### parameter search
- max out tables (most limited, can't have 2 tables in same stg), then max out entries 
### performance
- compile time: ~4.5 min (MAT entries)
- interpreter time:  >35 min(666096 pkts, flows) (recirculation) (externs!!!)

## [RTT](https://www.cs.princeton.edu/~xiaoqic/documents/paper-P4RTT-SPIN20.pdf)
Measure rtt in the data plane. Store timestamp of syn packets and match with corresponding ack. Implemented as multi-stage hash table; lazily evict expired records.
### parameters:
- tables (+)
- table size (columns in reg array) (+)
- timeout threshold (in ns) (?)
### optimization objective/measurement:
- maximize read success rate - number of samples generated in data plane / ground truth number of samples (record each successful sample and ground truth samples -> see rtt paper for more details)
### parameter search
- max tables (most limiting), max table size, find timeout threshold (fixed size)
- only compile when finding size
### performance
- compile time: ~40s
- interprer time: ~30s (619720 pkts) (w/o externs)

## [Starflow](https://www.usenix.org/system/files/atc18-sonchack.pdf)
Cache for telemetry data (aka grouped packet vectors) (implements select and grouping for a query). Implemented as 2 caches -- narrow and wide buffers. Narrow buffers can store more flows but smaller amount of data per flow than wide. If flow fills up its narrow buffer, attempt to allocate wide buffer. When buffer is full, flush the entries to software.
### parameters:
- slots in narrow buffer (?, but + w/o resource constraints)
- slots in wide buffer (?, but + w/o resource constraints)
- rows in narrow buffer (?, but + w/o resource constraints)
- rows in wide buffer (?, but + w/o resource constraints)
### optimization objective/measurement:
- minimize eviction ratio (evicted GPVs / total number of pkts -> see starflow paper for more details) (measure every time entry gets flushed from either narrow/wide buffer - from eviction or bc full)
### parameter search
- choose narrow buffer size (within some bounds), then fill up the available memory with wide buffer
- (or do the reverse, the idea is just to use up all the memory)
### performance
- compile time: ~1-2min
- interpreter time: ~10s (20000 pkts)


## Stateful firewall
Firewall with stages and cuckoo insert operation to mitigate collisions
### parameters:
- stages (rows?) (+)
- entries (columns in reg array) (+)
- timeout (in ns?) (?)
- delay (in ns?) (how long to wait before checking for timeouts) (?)
### optimization objective/measurement:
- minimize sum of bytes processed by recirculation and bytes that endhosts resend when legit traffic gets dropped (count recirculations and count how many incorrect packets get dropped)
### parameter search
### performance
- compile time: ~1.5min
- interpreter time: ~5s



