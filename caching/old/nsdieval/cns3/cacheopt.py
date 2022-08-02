class Opt:
    def __init__(self, pktpcap):
        self.ground_truth = {}
        self.pkts = pktpcap

    def gen_traffic(self):
        return    

    def calc_cost(self,measure):
        return measure[0]/78400

    def init_iteration(self,symbs):
        pass

