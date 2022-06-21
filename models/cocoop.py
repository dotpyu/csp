import argparse
import os

import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from collections import OrderedDict
from models.cocsp import VisualCtxEncoder, CoCSPInterface
from copy import deepcopy as dc

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class CoCOOP(CoCSPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        vctx_encoder,
        soft_embeddings: torch.nn.Parameter,
        comp_token_embedding,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            offset,
            soft_embeddings,
            vctx_encoder,
            class_token_ids=token_ids,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        # self.comp_token_embedding = comp_token_embedding#.type(self.clip_model.dtype)
        # self.soft_embeddings = soft_embeddings.to(device)
        self.start_token_embedding = comp_token_embedding[0, :]
        self.comp_token_embedding = comp_token_embedding[1:, :]  # .type(clip_model.dtype)
        self.offset = offset
        self.ctx_len = len(self.soft_embeddings)

    def construct_token_tensors(self, batch_img, pair_idx):

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]

        vctx = self.vctx_encoder(batch_img)#.to(self.dtype))  # (batch, vocab_sz)
        vctx = vctx.unsqueeze(1)  # (batch, 1, vocab_dim)
        soft_embeddings = self.soft_embeddings.unsqueeze(0)  # (1, n_ctx, vocab_dim)
        vctx_soft_embeddings = soft_embeddings + vctx  # (batch, vocab_sz, vocab_dim)
        #vctx_soft_embeddings = vctx_soft_embeddings.type(self.clip_model.dtype) # (batch, n_ctx, vocab_dim)

        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)

        # Problem here
        # token_tensor = self.clip_model.token_embedding(
        #     class_token_ids.to(self.device)
        # ).unsqueeze(0)  #.type(self.clip_model.dtype) (batch, prompt_len, vocab_dim)
        #
        #
        # eos_idx = int(self.token_ids[0].argmax())
        # token_tensor[:,:, eos_idx - 2, :] = self.frozen_embeddings[
        #     attr_idx
        # ]#.type(self.clip_model.dtype) # (batch, prompt_len, vocab_dim)
        # token_tensor[:,:, eos_idx - 1, :] = self.frozen_embeddings[
        #     obj_idx + self.offset
        #     ]#.type(self.clip_model.dtype) # (batch, prompt_len, vocab_dim)

        # adding the correct learnable context
        # print(vctx_soft_embeddings.shape)
        token_tensor = torch.hstack([self.start_token_embedding.data, self.soft_embeddings.unsqueeze(0).expand(self.comp_token_embedding.shape[0],-1,-1), self.comp_token_embedding.data])

        token_tensor = token_tensor.unsqueeze(0)
        token_tensor = token_tensor.repeat(len(batch_img), 1, 1, 1)
        token_tensor[:,:, 1: len(self.soft_embeddings) + 1, :] = token_tensor[:,:, 1: len(self.soft_embeddings) + 1, :] + vctx_soft_embeddings.unsqueeze(1).expand(-1, len(attr_idx), -1, -1)  # [BS, CAND_SZ, vocab_sz, vocab_dim]

        return token_tensor


def get_cocoop(train_dataset, config, device, prompt_template="a photo of x x"):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

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

    with torch.no_grad():
        comp_token_embedding = clip_model.token_embedding(tokenized.to(device))

    token_ids = clip.tokenize(prompt_template,
                              context_length=config.context_length).to(device)


    vocab_sz = soft_embedding.shape[-2]
    vis_dim = soft_embedding.shape[-1]

    vctx_encoder = VisualCtxEncoder(vis_dim, ctx_vectors.shape[-1], dtype=torch.float32).to(device)

    offset = len(attributes)

    optimizer = torch.optim.Adam(
        [soft_embedding]+ list(vctx_encoder.parameters()), lr=config.lr, weight_decay=config.weight_decay
    )

    coop = CoCOOP(
        clip_model,
        config,
        offset,
        vctx_encoder,
        soft_embedding,
        comp_token_embedding,
        token_ids,
        device=device,
        enable_pos_emb=True,
    )

    return coop, optimizer
