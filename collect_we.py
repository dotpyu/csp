from copy import deepcopy as dc
import json
from scipy.stats import sem
import numpy as np
import os

default_path = './vocab_results/'
cc_name = '{:s}_{:s}_seed_{:d}__{:s}.json'
clip_name = 'clip_{:s}__{:s}.json'

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


def clip_eval(checkpoint):
    print(f'Performance @ {checkpoint} with CLIP')
    res = []
    for dataset in datasets:
        for task in tasks:
            res.append(float(load_acc(default_path+dc(clip_name).format(dataset, task), checkpoint)))
    print()
    print(dc(clip_template).format(*res))
    print()
    print("= "*50)

def reg_eval(method, checkpoint):
    print(f'Performance @ {checkpoint} with {method}')
    res = []
    for dataset in datasets:
        for task in tasks:
            ir = []
            for seed in seeds:
                ir.append(float(load_acc(default_path + dc(cc_name).format(method, dataset, seed, task), checkpoint)))
            res += [np.mean(ir), sem(ir)]
    print()
    print(dc(output_template).format(*res))
    print()
    print("= " * 50)

if __name__ == '__main__':
    for ckpt in ckpts:
        print(f"Running checkpoint @ {ckpt}")
        clip_eval(ckpt)
        reg_eval('coop', ckpt)
        reg_eval('csp', ckpt)
        print()
