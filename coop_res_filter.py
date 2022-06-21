import pandas as pd
import json
import os
import argparse
import sys
import numpy as np
from scipy.stats import sem
from copy import deepcopy as dc

datasets = ['ut-zappos', 'cgqa', 'mit-states']
data_cap = ['ut', 'c', 'mit']
models = ['vitb16', 'vitl14', 'r50', 'r101']
formal_models = ['ViT-B/16', 'ViT-L/14', 'RN50', 'RN101']
seeds = range(5)

META_PATH = "/users/pyu12/data/pyu12/model/{:s}/coop_models/{:d}/soft_embeddings_epoch_{:d}.closed.json"

def mmain(args):
    result_list = []
    total_epochs = 51
    crit = 'best_unseen'#'best_hm'
    val_results = -np.ones([51, 5])
    test_results = -np.ones([51, 5, 4])

    for i in [0, 1, 2, 3, 4]:
        for epoch in range(21):
            path = dc(META_PATH).format(args.dataset, i, epoch)
            if os.path.exists(path):
                try:
                    results = json.load(open(path))
                except Exception as e:
                    print(e)
                    test_results[epoch, i, :] = np.zeros([4])
                    continue
                val_results[epoch, i] = results['val'][crit]
                test = results['test']
                auc, bhm, bu, bs = test['AUC']*100, test['best_hm']*100, test['best_seen']*100, test['best_unseen']*100
                test_results[epoch, i, :] = np.array([bs, bu, bhm, auc])
    best_epoch_val = np.argmax(np.mean(val_results, axis=1))
    best_test = test_results[best_epoch_val, :, :]
    print(best_test)
    avg_results = np.mean(best_test, axis=0) # 4
    print('='*50)
    print(args.dataset)
    print('best epoch: ', best_epoch_val)
    output_string = "{:.2f} , {:.2f},  {:.2f}  " \
                    ", {:.2f},   {:.2f}  , {:.2f}, " \
                    "{:.2f}  , {:.2f} ".format(avg_results[0], sem(best_test[:, 0]),
                                                      avg_results[1], sem(best_test[:, 1]),
                                                      avg_results[2], sem(best_test[:, 2]),
                                                      avg_results[3], sem(best_test[:, 3]),
                                                      )

    # output_string = "{:.2f}, {:.2f}, {:.2f}, {:.2f};".format(avg_results[1], avg_results[0], avg_results[2], avg_results[3])
    print(output_string)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of the dataset', type=str)
    # parser.add_argument('--model', help='name of the model', type=str)
    args = parser.parse_args()
    mmain(args)

    print('done!')
