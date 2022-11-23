import subprocess, pickle

class Opt:
    def __init__(self, pktpcap):
        self.ground_truth = {}
        self.pkts = pktpcap

    def gen_traffic(self):
        return    

    def calc_cost(self,measure):
        return measure[0]/80000000

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


