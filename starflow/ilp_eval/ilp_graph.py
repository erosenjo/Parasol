import matplotlib.pyplot as plt
import numpy as np
import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



f = open('testing_eval_rand.pkl','rb')
eval_list = pickle.load(f)
eval_list = list(filter(lambda a: a != -1, eval_list))

print(eval_list)
print(len(eval_list))

print(sum(eval_list)/len(eval_list))

plt.figure(figsize=(8,5))
x_vals = [i for i in range(len(eval_list))]
plt.plot(x_vals, eval_list)

plt.savefig('starflow_ilp.pdf')


