import subprocess
import sys
from os.path import exists
import tqdm



META_PATH = "/users/pyu12/data/pyu12/model/{:s}/finetune_amp_models/{:d}/finetune_model_{:d}.pth"

datasets = ['ut-zappos', 'mit-states'] #, 'cgqa'
data_cap = ['ut', 'c', 'mit']
models = ['vitb16', 'vitl14', 'r50', 'r101']
formal_models = ['ViT-B/16', 'ViT-L/14', 'RN50', 'RN101']
seeds = range(5)


def check(dataset, seed, epoch=1):
    return exists(META_PATH.format(dataset, seed, epoch))

cl = ["16", "12"]

def subproc(dataset, seed, epoch):
    #abbr = data_cap[datasets.index(dataset)]
    #fmodel = formal_models[models.index(model)]
    pt_file = META_PATH.format(dataset, seed, epoch)
    subprocess.run(["/users/pyu12/.conda/envs/gpu/bin/python3.8", "evaluate_full.py",
                    "--dataset", dataset,
                    "--experiment_name", f"finetune",
                    "--clip_model", formal_models[1],
                    "--model_path", pt_file,
                    "--context_length", cl[datasets.index(dataset)],
                    "--text_encoder_batch_size", "36",
                    "--eval_batch_size", "16",
                    "--finetune"
                    ])


def vmain():
    seed = int(sys.argv[1])
    for dataset in datasets:
        for epoch in tqdm.tqdm(range(21)):
            if check(dataset, seed, epoch):
                subproc(dataset, seed, epoch)


def cmain():
    missing = []
    print('='*50)
    for seed in seeds:
        for dataset in datasets:
            for model in models:
                if not check(dataset, model, seed):
                    missing.append([dataset, model, seed])
                    print([dataset, model, seed])


if __name__ == '__main__':
    vmain()

