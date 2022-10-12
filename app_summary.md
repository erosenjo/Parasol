# Applications
This is a summary of the data structures implemented in Parasol, along with their associated (symbolic) parameters and optimization objective function.

## Bloom filter
Approximate set membership. Assume single register array.
### parameters:
- bits (+)
- number of hashes (?)
### optimization objective/measurement:
- minimize false positive rate (record query answer per packet and ground truth answers)
### parameter search order
- bits, hashes

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
### parameter search order
- if we know skewness, then choose data structure first; else, just optimize for each one separately, in parallel
- if precision: tables, entries
- if cms: tables, entries, rows, columns, expiring threshold, hh threshold 
### performance
- compile time: ~2min, ~3.5min
- dataflow graph + layout time:  
- interpreter time: (externs!!)
- tofino compile time: ????
### tofino?
- phv allocation error

## CMS
Simple sketch for identifying heavy hitters. Can include timeout/count threshold as parameters (see caching example).
### parameters
- rows (+)
- columns (+)
### optimization objective/measurement:
- minimize mean absolute error (record approximate count for each packet and ground truth counts)
### parameter search order
- rows, cols
### performance
- compile time: ~3s
- dataflow graph + layout time: ~0.7s 
- interpreter time: ~35s (500000 pkts, 55095 flows)
- tofino compile time: 
### tofino?
- lucid compiler estimates 12 stgs for 11 rows
- tofino compiler uses 14 stgs for 11 rows
- 10 rows fits on tofino

## [Conquest](https://www.cs.princeton.edu/~xiaoqic/documents/paper-ConQuest-CoNEXT19.pdf)
Queue length measurement in the data plane, to identify which flows are contributing to queue. Implemented as a series of sketches that are periodically rotated through.
### parameters:
- snapshots(H) (cms) (?, but + w/o resource constraints)
- rows in snapshot(R) (?, but + w/o resource constraints)
- columns in snapshot(C) (?, but + w/o resource constraints)
- time window(LOG\_CQ\_T) (in ns, determines how often we cycle through snaps) (?)
### optimization objective/measurement:
- maximize f-score (precision, recall) (extern records qlength estimation when qlength is large enough, compute how often we correctly identify a large queue and how good the estimation was using ground truth qlength -> see conquest paper/Danny's simulation code for more detail)
### parameter search order
- snapshots, rows, columns, time window
- max number of reg accesses/stages/hashes
### performance
- compile time: ~5s (realistic case), >22min (this is worst case, maxing out all values-32 sketches, 4 rows, 65536 cols)
- dataflow graph + layout time: ERROR, layout stages?
- interpreter time: ~32s (300001 pkts)
- tofino compile time: ~13min
### tofino?
- lucid compiler stg estimation accurate
- no compiler issues

## [Flowlet switching](http://web.mit.edu/domino/domino-sigcomm.pdf)
Load-balancing application that sends flowlets on randomly chosen next hop, all packets in same flowlet will go to same hop. Assume single register array that saves the hop and most recent arrival time for each flowlet.
### parameters:
- flowlet slots (columns in reg array) (+)
- IPG threshold (in ns; packets with IPG smaller than threshold considered part of same flowlet) (?)
### optimization objective/measurement:
- minimize gap to even distribution (measure the pkts in bytes sent along each hop and compute error with an even distribution of bytes across each hop)
### parameter search order
- flowlet slots, IPG threshold
- max memory (other resources are fixed)
### performance
- compile time: ~0.7s
- dataflow graph + layout time: ~0.5s 
- interpreter time: ~45s (500000 pkts)
- tofino compile time: ????
### tofino?
- table could not fit w/in a single input crossbar

## [Fridge](https://www.cs.princeton.edu/~jrex/papers/apocs22.pdf)
Unbiased delay measurements. Hash-indexed array storing time of syn packets, to get matched when corresponding ack arrives. Entries added to fridge with probability p; aggressively overwrite and correct bias as samples collected. Assume single fridge (register array).
### parameters
- entry probability (?)
- fridge size (columns in reg array) (+)
### optimization objective/measurement:
- minimze max percentile error (measure rtt sample (in ns) and correction factor, compute cdf from measurements and ground truth, compute error at various percentiles -> see fridge paper for more details)
### parameter search order
- fridge size, entry probability
- max memory (other resources are fixed)
### performance
- compile time: ~25s 
- dataflow graph + layout time: ~3s 
- interpreter time: ~30s (666096 pkts, 212377 samples)
- tofino compile time: ???? 
### tofino?
- phv allocation error

## Hash table
Simple multi-stage hash table. Assume entries are never evicted.
### parameters:
- tables (num reg arrays) (+)
- entries (per table) (columns in reg array) (+)
### optimization objective/measurement:
- minimize number of collisions (extern that keeps counter of collisions)
### parameter search order
- tables, entries
### performance
### tofino?

## [Precision](https://www.cs.princeton.edu/~xiaoqic/documents/paper-PRECISION-ToN.pdf)
Heavy hitter detection. Implemented as a series of tables that store flow key and counter. Probabilistically insert new entries upon collisions.
### parameters:
- tables (+)
- entries (columns in reg array) (+)
### optimization objective/measurement:
- minimize mean absolute error (same as cms)
### parameter search order
- tables, entries
### performance
- compile time: ~2min (MAT entries)
- dataflow graph + layout time: ~2min
- interpreter time:  ~4min (666096 pkts)
- tofino compile time: 37s
### tofino?
- lucid compiler stg estimate accurate
- no compile issues

## [RTT](https://www.cs.princeton.edu/~xiaoqic/documents/paper-P4RTT-SPIN20.pdf)
Measure rtt in the data plane. Store timestamp of syn packets and match with corresponding ack. Implemented as multi-stage hash table; lazily evict expired records.
### parameters:
- tables (+)
- table size (columns in reg array) (+)
- timeout threshold (in ns) (?)
### optimization objective/measurement:
- maximize read success rate - number of samples generated in data plane / ground truth number of samples (record each successful sample and ground truth samples -> see rtt paper for more details)
### parameter search order
- tables, table size, timeout threshold
### performance
- compile time: ~25s
- dataflowgraph + layout time: ~3s
- interprer time: ~30s (619720 pkts)
- tofino compile time: ???? (7s)
### tofino?
- doesn't compile, internal compiler bug

## [Starflow](https://www.usenix.org/system/files/atc18-sonchack.pdf)
Cache for telemetry data (aka grouped packet vectors) (implements select and grouping for a query). Implemented as 2 caches -- narrow and wide buffers. Narrow buffers can store more flows but smaller amount of data per flow than wide. If flow fills up its narrow buffer, attempt to allocate wide buffer. When buffer is full, flush the entries to software.
### parameters:
- slots in narrow buffer (?, but + w/o resource constraints)
- slots in wide buffer (?, but + w/o resource constraints)
- rows in narrow buffer (?, but + w/o resource constraints)
- rows in wide buffer (?, but + w/o resource constraints)
### optimization objective/measurement:
- minimize eviction ratio (evicted GPVs / total number of pkts -> see starflow paper for more details) (measure every time entry gets flushed from either narrow/wide buffer - from eviction or bc full)
### parameter search order
- rows in narrow, rows in wide, slots in narrow, slots in wide
### performance
- compile time: ~1-2min
- dataflow graph + layout time: ERROR
- interpreter time: ~10s (20000 pkts)
- tofino compile time: ???? (17hrs)
### tofino?
- doesn't compile, internal compiler bug

## Stateful firewall
Firewall with stages and cuckoo insert operation to mitigate collisions
### parameters:
- stages (rows?) (+)
- entries (columns in reg array) (+)
- timeout (in ns?) (?)
- delay (in ns?) (how long to wait before checking for timeouts) (?)
### optimization objective/measurement:
- minimize sum of bytes processed by recirculation and bytes that endhosts resend when legit traffic gets dropped (count recirculations and count how many incorrect packets get dropped)
### parameter search order
- stages, entries, timeout, delay
### performance
- compile time: ~1.5min
- interpreter time: ~5s
- tofino compile time: ????
### tofino?
- doesn't compile, internal compiler bug


