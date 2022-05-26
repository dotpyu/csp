import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from models.csp import csp_init
from clip_modules.model_loader import load
from collections import OrderedDict

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class CoCSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        vctx_encoder,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

        self.offset = offset
        self.vctx_encoder = vctx_encoder

    def construct_token_tensors(self, pair_idx):

        vctx = self.vctx_encoder(batch_img)  # (batch, vocab_sz)
        vctx = vctx.unsqueeze(1)  # (batch, 1, vocab_sz)
        vctx_soft_embeddings = self.soft_embeddings + vctx  # (batch, vocab_dim, vocab_sz)

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = vctx_soft_embeddings[
            :, attr_idx,:
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = vctx_soft_embeddings[
            :, obj_idx + self.offset, :
        ].type(self.clip_model.dtype)

        return token_tensor


def get_cocsp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    vocab_sz = soft_embedding.shape[-2]
    vis_dim = soft_embedding.shape[-1]

    vctx_encoder = nn.Sequential(OrderedDict([
        ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
        ("relu", nn.ReLU(inplace=True)),
        ("linear2", nn.Linear(vis_dim // 16, vocab_sz))
    ]))

    optimizer = torch.optim.Adam(
        [soft_embedding, vctx_encoder.parameters()],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    interface = CoCSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        vctx_encoder,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface, optimizer

