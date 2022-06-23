from copy import deepcopy as dc
import json
from scipy.stats import sem
import numpy as np
import os

META_PATH = "/users/pyu12/data/pyu12/model/{:s}/coop_models/{:d}/soft_embeddings_epoch_{:d}.pt"

cc_name = '{:s}_{:s}_seed_{:d}__{:s}.json'
clip_name = 'clip_{:s}__{:s}.json'

output_template = "{:.1f},{:.1f},{:.1f},{:.1f},,{:.1f},{:.1f},{:.1f},{:.1f},,{:.1f},{:.1f},{:.1f},{:.1f}"
clip_template = "{:.1f},,{:.1f},,,{:.1f},,{:.1f},,,{:.1f},,{:.1f}"


coop_template = """
  & COOP 
  && 36.8$_{0.1}$  & 16.5 $_{{0.1}}$  & 16.1 $_{0.1}$ & 4.7 $_{0.0}$
  && 61.8$_{0.5}$  & 39.3 $_{1.3}$  & 35.6 $_{0.7}$ & 19.5 $_{0.6}$
  && 20.9$_{0.3}$ & 4.5 $_{0.2}$  & 5.7 $_{0.2}$ & 0.73 $_{0.0}$\\ 
  """
coop_dataset_template = """
&& 36.8$_{0.1}$  & 16.5 $_{{0.1}}$  & 16.1 $_{0.1}$ & 4.7 $_{0.0}$"""
ckpts = [1,5,10,20]
datasets = ['mit-states', 'ut-zappos', 'cgqa']
tasks = ['attr', 'obj']
seeds = range(5)

epochs = {
    'coop' : {
        'mit-states': 19,
        'ut-zappos': 17,
        'cgqa': 9
    },
    'csp' : {
        'mit-states': 20,
        'ut-zappos': 13,
        'cgqa': 20
    },
}

def load_acc(path):
    if os.path.exists(path):
        try:
            results = json.load(open(path))
            return results['AUC']*100, results['best_hm']*100, results['best_unseen']*100, results['best_seen']*100
        except Exception as e:
            print(e)

def coop_eval():
    res = []
    output_str = ""

    for dataset in datasets:
        aucs, hms, us, ss = [], [],[],[]
        line_template = "{:.1f} $_{{{:.1f}}}$ &"
        for seed in seeds:
            auc, bhm, bu, bs = load_acc(dc(META_PATH).format(dataset, seed, epochs['coop'][dataset]))
            aucs.append(auc)
            hms.append(bhm)
            us.append(bu)
            ss.append(bs)
        output_str += dc(line_template).format(np.mean(us), sem(us))
        output_str += dc(line_template).format(np.mean(ss), sem(ss))
        output_str += dc(line_template).format(np.mean(hms), sem(hms))
        output_str += dc(line_template).format(np.mean(aucs), sem(aucs))
        output_str += '&'
    print()
    print(output_str)
    print()
    print("= " * 50)

if __name__ == '__main__':
    coop_eval()