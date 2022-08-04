import pickle

qlens = []

def track_qlen(qlen_est):
    qlens.append(qlen_est)
    if len(qlens) > 299997:
        with open('qlens.pkl','wb') as f:
            pickle.dump(qlens,f)



