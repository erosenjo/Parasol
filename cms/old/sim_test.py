import subprocess, pickle
from traffic import *

def run_interp():
    # run the interpreter
    cmd = ["../../dpt", "cms_sym.dpt"]
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print("err")

    print("output")

    '''
    outfile = open("test.txt", "rb")
    try: # Hacky, but quick way to unpickle the whole file
        while True:
            print(pickle.load(outfile))
    except EOFError:
        pass
    '''
    outfile = "test.txt"
    measurement = pickle.load(open(outfile,"rb"))
    print(measurement)

ground_truth = gen_traffic("univ1_pt1.pcap")

run_interp()

#print(ground_truth)


