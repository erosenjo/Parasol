import pickle

qlens = []

def track_qlen(qlen_est):
    qlens.append(qlen_est)
    if len(qlens) > 1021721:
        with open('qlens.pkl','wb') as f:
            pickle.dump(qlens,f)



