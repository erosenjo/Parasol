# keep track of measurements from interpreter
import pickle

interp_measure = {}
clean_counter = [0]
# NOTE: hard coding number of bits in cms for now, [0]*cols*rows for each cms
#   need to be careful to update if size of sketch changes
cms_bits = [[0]*128*2, [0]*128*2]

def update_count(src, dst, count):
    interp_measure[str(src)+str(dst)] = count
    clean_counter[0] += 1

# keep track of how many bits are set in each cms
# update when we write AND clean from cms
# TODO: should we clean one register total each pkt? or one register in EACH row per pkt?
#   for now, we're cleaning one register TOTAL per pkt
#   to do one reg in each row per pkt, move clean_counter[0] += 1 to this function (out of update_count)
def update_bits_set(cms_write, row, index):
    cms_clean = 1-cms_write
    # sketch we're writing to (set bits to 1)
    write_bit = 128*row+index
    cms_bits[cms_write][write_bit] = 1
    # sketch we're cleaning (set bits to 0)
    cms_bits[cms_clean][clean_counter[0]] = 0

def write_to_file():
    with open("estimated_counts.pkl",'wb') as f:
        pickle.dump(interp_measure, f)
    interp_measure.clear()

    bits_set = [sum(bits) for bits in cms_bits]
    with open("bits_set.pkl", 'wb') as f:
        pickle.dump(bits_set, f)
    # reset vars we use to keep track of bits cleaned
    clean_counter[0] = 0



