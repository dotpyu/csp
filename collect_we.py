from copy import deepcopy as dc
import json
from scipy.stats import sem
import numpy as np
import os

default_path = './vocab_results/'
cc_name = '{:s}{:s}_{:s}_seed_{:d}__{:s}.json'
clip_name = '{:s}clip_{:s}__{:s}.json'

output_template = "{:.1f},{:.1f},{:.1f},{:.1f},,{:.1f},{:.1f},{:.1f},{:.1f},,{:.1f},{:.1f},{:.1f},{:.1f}"
clip_template = "{:.1f},,{:.1f},,,{:.1f},,{:.1f},,,{:.1f},,{:.1f}"

ckpts = [1,5,10,20]
datasets = ['mit-states', 'ut-zappos', 'cgqa']
tasks = ['attr', 'obj']
seeds = range(5)

def load_acc(path, checkpoint):
    if os.path.exists(path):
        try:
            results = json.load(open(path))
            return results[str(checkpoint)]
        except Exception as e:
            print(e)


def clip_eval(checkpoint, prefix):
    print(f'Performance @ {checkpoint} with CLIP {prefix}')
    res = []
    for dataset in datasets:
        for task in tasks:
            res.append(float(load_acc(default_path+dc(clip_name).format(prefix, dataset, task), checkpoint))*100)
    print()
    print(dc(clip_template).format(*res))
    print()
    print("= "*50)

def reg_eval(method, checkpoint, prefix):
    print(f'Performance @ {checkpoint} with {method} {prefix}')
    res = []
    for dataset in datasets:
        for task in tasks:
            ir = []
            for seed in seeds:
                ir.append(float(load_acc(default_path + dc(cc_name).format(prefix, method, dataset, seed, task), checkpoint))*100)
            res += [np.mean(ir), sem(ir)]
    print()
    print(dc(output_template).format(*res))
    print()
    print("= " * 50)

if __name__ == '__main__':
    ckpts = [1]
    for ckpt in ckpts:
        for prefix in ['train', '']:
            print(f"Running checkpoint @ {ckpt}, {prefix}")
            clip_eval(ckpt, prefix)
            reg_eval('coop', ckpt, prefix)
            reg_eval('csp', ckpt, prefix)
            print()
