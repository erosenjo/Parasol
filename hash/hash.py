import pickle

collisions = [0]

def log_collision():
    collisions[0] += 1
    with open("colls.txt",'wb') as f:
        pickle.dump(collisions[0],f)



