# keep track of measurements from interpreter
import pickle

interp_measure = {}
counter = [0]

def update_count(src, dst, count):
    counter[0]+=1
    interp_measure[str(src)+str(dst)] = count
    '''
    if counter[0] > 499999:
        with open("test.pkl",'wb') as f:
            pickle.dump(interp_measure, f)
    '''

def write_to_file():
    with open("test.pkl",'wb') as f:
        pickle.dump(interp_measure, f)

    interp_measure.clear()

