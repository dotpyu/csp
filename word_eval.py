import argparse
import copy
import json
import os
from itertools import product

import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from sklearn.metrics import top_k_accuracy_score as topk
from copy import deepcopy as dc

cudnn.benchmark = True


se_path = {
    "mit-states": "/users/pyu12/data/bats/projects/clip-labeler/mit-states/{:s}/soft_embeddings_epoch_{:d}.pt",
    "ut-zappos": "/users/pyu12/data/bats/projects/clip-labeler/vitl14_utzappos/{:s}/soft_embeddings_epoch_{:d}.pt",
    "cgqa": "/users/pyu12/data/bats/projects/clip-labeler/vitl14_cgqa/{:s}/soft_embeddings_epoch_{:d}.pt"
}

coop_path = "/users/pyu12/data/pyu12/model/{:s}/coop_models/{:s}/soft_embeddings_epoch_{:d}.pt"

epochs = {
    'coop' : {
        'mit-states': 19,
        'ut-zappos': 17,
        'cgqa': 19
    },
    'csp' : {
        'mit-states': 20,
        'ut-zappos': 13,
        'cgqa': 20
    },
}

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

attribute_template = 'A photo of X object'
object_template = 'A photo of X'
alternative_attr_template = 'The object in the photo is X'


def predict_logits(model, text_rep, dataset, device, config):

    model.eval()

    true_labels = []
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()


    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_img_feat = model.encode_image(batch_img)
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            logits = (
                model.clip_model.logit_scale.exp()
                * normalized_img
                @ text_rep.t()
            )

            attr_truth, obj_truth, _ = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            true_labels.append(obj_truth if config.eval_obj else attr_truth)

    return all_logits, torch.cat(true_labels).to("cpu")


def compute_coop_representations(model, test_dataset, config, device):


    target = test_dataset.objs if config.eval_obj else test_dataset.attrs
    template = object_template if config.eval_obj else attribute_template
    if config.alter_attr:
        target = test_dataset.attrs
        template = alternative_attr_template
    prompts = [dc(template).replace('X', t) for t in target]

    # checked: "object" is single token id at 14115

    class_token_ids = clip.tokenize(
        prompts,
        context_length=config.context_length,
    )
    token_embedding = model.clip_model.token_embedding(class_token_ids.to(device))
    token_embedding[
    :, 1: len(model.soft_embeddings) + 1, :
    ] = model.soft_embeddings.type(model.clip_model.dtype)

    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for tidx, t in enumerate(tqdm(target)):

            token_tensor = token_embedding[tidx]
            token_tensor = token_tensor.unsqueeze(0)
            text_features = model.text_encoder(
                model.token_ids,
                token_tensor,
                enable_pos_emb=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            rep = torch.cat([rep, text_features], dim=0)

    return rep

def compute_csp_representations(model, test_dataset, config, device):

    target = test_dataset.objs if config.eval_obj else test_dataset.attrs
    template = object_template if config.eval_obj else attribute_template
    offset = len(test_dataset.attrs) if config.eval_obj else 0
    back_offset = -1 if config.eval_obj else -2
    if config.alter_attr:
        target = test_dataset.attrs
        template = alternative_attr_template
        back_offset = -1

    # checked: "object" is single token id at 14115

    class_token_ids = clip.tokenize(
        [template],
        context_length=config.context_length,
    )

    token_embedding = model.clip_model.token_embedding(class_token_ids.to(device))
    replacement_idx = int(model.token_ids[0].argmax()) + back_offset

    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for tidx, t in enumerate(tqdm(target)):

            token_tensor = dc(token_embedding).to(device).to(model.clip_model.dtype)
            token_tensor[:, replacement_idx, :] = model.soft_embeddings[offset + tidx].to(device).to(model.clip_model.dtype)

            text_features = model.text_encoder(
                model.token_ids,
                token_tensor,
                enable_pos_emb=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            rep = torch.cat([rep, text_features], dim=0)

    return rep


def clip_baseline(model, test_dataset, config, device):
    """Function to get the clip representations.

    Args:
        model (nn.Module): the clip model
        test_dataset (CompositionDataset): the test/validation dataset
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0

    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations with clip model.
    """
    target = test_dataset.objs if config.eval_obj else test_dataset.attrs
    template = object_template if config.eval_obj else attribute_template
    if config.alter_attr:
        target = test_dataset.attrs
        template = alternative_attr_template
    prompts = [dc(template).replace('X', t) for t in target]

    tokenized_prompts = clip.tokenize(
        prompts, context_length=config.context_length)

    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_tokens in tokenized_prompts:
            batch_tokens = batch_tokens.unsqueeze(0)
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)

    return rep



def save_results(test_dataset, config, prefix=""):
    if config.experiment_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length)

        model = CLIPInterface(
            clip_model,
            config,
            token_ids=None,
            device=device,
            enable_pos_emb=True)
        test_text_rep = clip_baseline(model, test_dataset, config, device)
    else:
        model, optimizer = get_model(test_dataset, config, device)
        soft_embs = torch.load(se)['soft_embeddings']
        model.set_soft_embeddings(soft_embs)
        if config.experiment_name == 'csp':
            test_text_rep = compute_csp_representations(
                model, test_dataset, config, device)
        elif config.experiment_name == 'coop':
            test_text_rep = compute_coop_representations(
                model, test_dataset, config, device)

    logits, gt = predict_logits(model, test_text_rep, test_dataset, device, config)
    target = test_dataset.objs if config.eval_obj else test_dataset.attrs
    top_res = {i: topk(gt, logits, k=i, labels=list(range(len(target)))) for i in [1, 2, 3, 5, 10, 20]}
    suffix = '_obj' if config.eval_obj else '_attr'

    if config.alter_attr: suffix += '_alter'
    if config.experiment_name != 'clip':
        result_path = './vocab_results/{:s}{:s}_{:s}_seed_{:d}_{:s}.json'.format(prefix, config.experiment_name, config.dataset, config.seed, suffix)
    else:
        result_path = './vocab_results/{:s}clip_{:s}_{:s}.json'.format(prefix, config.dataset, suffix)

    with open(result_path, 'w+') as fp:
        json.dump(top_res, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=1e-04
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-L/14"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=16, type=int
    )

    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--eval_obj",
        help="evaluate object vocab",
        action="store_true",
    )
    parser.add_argument(
        "--alter_attr",
        help="alternative attr prompt",
        action="store_true",
    )

    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help="optional threshold"
    )

    parser.add_argument(
        '--threshold_trials',
        type=int,
        default=50,
        help="how many threshold values to try"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="seed"
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )

    config = parser.parse_args()

    # set the seed value
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")



    # soft_embedding path assemble here:

    if config.experiment_name == 'csp':
        se = dc(se_path[config.dataset]).format(
            "coop_" + str(config.seed) if config.experiment_name == 'coop' else str(config.seed),
            epochs[config.experiment_name][config.dataset])
        if not os.path.exists(se):
            print(f'{se} not found')
            print('code exiting!')
            exit(0)
    if config.experiment_name == 'coop':
        se = dc(coop_path).format(config.dataset, str(config.seed), epochs[config.experiment_name][config.dataset])

    dataset_path = DATASET_PATHS[config.dataset]

    print('loading train dataset')
    train_dataset = CompositionDataset(dataset_path,
                                     phase='train',
                                     split='compositional-split-natural',
                                     open_world=False)
    save_results(train_dataset, config, prefix='train')


    print('loading test dataset')
    # test_dataset = CompositionDataset(dataset_path,
    #                                   phase='test',
    #                                   split='compositional-split-natural',
    #                                   open_world=False)

    # True attr/obj labels loaded from test/val dataset
    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural',
                                       open_world=False)
    save_results(test_dataset, config, prefix='')



    print("done!")
