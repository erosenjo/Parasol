import subprocess, pickle

class Opt:
    def __init__(self, pktpcap):
        self.ground_truth = {}
        self.pkts = pktpcap

    def gen_traffic(self):
        return    

    def calc_cost(self,measure):
        #return measure[0]/1000000
        miss_rates = [x/1000000 for x in measure]
        print("MISS RATES", miss_rates)
        return sum(miss_rates)/len(miss_rates)

    def init_iteration(self,symbs):
        pass


'''
o = Opt("")
o.gen_traffic()
cmd = ["make", "interp"]
ret = subprocess.run(cmd)

measurement = []
outfiles = ["misses.pkl"]
for out in outfiles:
    measurement.append(pickle.load(open(out,"rb")))
print(o.calc_cost(measurement))
'''


