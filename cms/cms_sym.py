# keep track of measurements from interpreter
import pickle

interp_measure = {}
counter = [0]

def update_count(src, dst, count):
    counter[0]+=1
    interp_measure[str(src)+str(dst)] = count
    if counter[0] > 887640:
        with open("test.txt",'wb') as f:
            pickle.dump(interp_measure, f)


