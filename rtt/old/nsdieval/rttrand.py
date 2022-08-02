

import pickle


collisions = [0]
timeouts = [0]
numsamples = [0]
samples = []
tests = {}

def log_collision():
    collisions[0] += 1
    with open('collisionsrand.txt','wb') as f:
        pickle.dump(collisions[0],f)

def log_timeout(t):
    if t:
        timeouts[0] += 1
        with open('timeoutsrand.txt','wb') as f:
            pickle.dump(timeouts[0],f)

def log_rttsample(sample):
    numsamples[0]+=1
    with open('numsamplesrand.txt','wb') as f:
        pickle.dump(numsamples[0],f)

    samples.append(sample)
    with open('samplesrand.txt','wb') as f:
        pickle.dump(samples,f)


def log_test(ack, rtt):
    tests[ack] = rtt
    with open('testsrand.txt','wb') as f:
        pickle.dump(tests,f)




