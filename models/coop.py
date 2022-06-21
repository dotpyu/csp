import argparse
import os

import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from copy import deepcopy as dc

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def coop(train_dataset, config, device, prompt_template="a photo of x x"):
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    ctx_init = "a photo of "
    n_ctx = len(ctx_init.split())
    prompt = clip.tokenize(ctx_init,
                           context_length=config.context_length).to(device)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt)

    ctx_vectors = embedding[0, 1: 1 + n_ctx, :]

    soft_embedding = nn.Parameter(ctx_vectors).to(device)

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    concerned_pairs = train_dataset.concerned_pairs
    compositions = [f"{ctx_init}{attributes[pair[0]]} {classes[pair[1]]}" for pair in concerned_pairs]

    tokenized = torch.cat(
        [
            clip.tokenize(compositions, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )

    comp_token_embedding = clip_model.token_embedding(tokenized.to(device))


    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)


    optimizer = torch.optim.Adam(
        [soft_embedding], lr=config.lr, weight_decay=config.weight_decay
    )

    offset = len(attributes)

    coop = COOP(
        clip_model,
        config,
        offset,
        soft_embedding,
        comp_token_embedding,
        token_ids,
        device=device,
        enable_pos_emb=True,
    )

    return coop, optimizer


class COOP(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        soft_embeddings: torch.nn.Parameter,
        comp_token_embedding,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.comp_token_embedding = comp_token_embedding#.type(clip_model.dtype)
        self.offset = offset
        self.ctx_len = len(self.soft_embeddings)



    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)

        token_tensor = dc(self.comp_token_embedding).to(self.device)

        token_tensor[
            :, 1 :self.ctx_len + 1, :
        ] = self.soft_embeddings#.type(self.clip_model.dtype)

        return token_tensor
