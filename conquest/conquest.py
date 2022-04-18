import pickle

qlens = []

def track_qlen(qlen_est):
    qlens.append(qlen_est)
    if len(qlens) > 299997:
        with open('qlens.txt','wb') as f:
            pickle.dump(qlens,f)



