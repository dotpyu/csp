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


class VisualCtxEncoder(nn.Module):
    """
    Visual Context Encoder
    """
    def __init__(self, vis_dim, prompt_vocab_sz=2):
        super(VisualCtxEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, prompt_vocab_sz))
        ])).half()

    def forward(self, x):
        return self.encoder(x)


def logits_compute(vc_text, img):
    """
    vc_text is a pytorch tensor of (batch_size, candidate_size, embed_dim)
    img is a pytorch tensor of (batch_size, embed_dim)

    this function computes the matrix multiplication for each batch of the vc_text of the respective image
    """
    batch_size = vc_text.size(0)
    candidate_size = vc_text.size(1)
    embed_dim = vc_text.size(2)
    img = img.unsqueeze(1).repeat(1, candidate_size, 1)
    img = img.view(batch_size * candidate_size, embed_dim)
    vc_text = vc_text.view(batch_size * candidate_size, embed_dim)
    logits = torch.bmm(img, vc_text.transpose(1, 2))
    logits = logits.view(batch_size, candidate_size, -1)
    return logits



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
        soft_embeddings = soft_embeddings.unsqueeze(0).repeat(len(config.train_batch_size), 1, 1)
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
        self.vctx_encoder = vctx_encoder.to(self.device)

    def construct_token_tensors(self, batch_img, pair_idx):

        vctx = self.vctx_encoder(batch_img)  # (batch, 2)

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype).unsqueeze(0).repeat(len(batch_img),1,1,1)

        eos_idx = int(self.token_ids[0].argmax())

        # soft_embeddings = self.attr_dropout(self.soft_embeddings)
        # print(soft_embeddings.shape)
        # print(token_tensor.shape)
        # print(vctx.shape)

        # torch.Size([360, 768])
        # torch.Size([64, 1262, 8, 768])
        # torch.Size([64, 2]) # scalar bias
        # vctx_soft_embeddings = soft_embeddings.unsqueeze(0) + vctx  # (batch, vocab_sz, vocab_dim)

        # Token Tensors old: (label_sz, vocab_sz, vocab_dim) -> (batch, label_sz, vocab_sz, vocab_dim)

        '''
        RuntimeError: The size of tensor a (768) must match the size of tensor b (64) at non-singleton dimension 2

        '''
        # reshape vctx to be the same shape of soft_embeddings
        soft_embeddings = self.soft_embeddings.to(self.device)
        soft_embeddings[:, :self.offset, :] += vctx[:, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, self.offset, 1)
        soft_embeddings[:, self.offset:, :] += vctx[:, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, soft_embeddings.shape[1] - self.offset, 1)

        attr_emb = soft_embeddings[:, attr_idx, :].type(self.clip_model.dtype)
        obj_emb = soft_embeddings[:, obj_idx + self.offset, :].type(self.clip_model.dtype)
        # print(attr_emb.expand(len(batch_img),-1).shape)

        token_tensor[:, :, eos_idx - 2, :] = attr_emb
        token_tensor[:, :, eos_idx - 1, :] = obj_emb
        return token_tensor

    def forward(self, batch_img, idx):
        batch_img = batch_img.to(self.device)

        token_tensors = self.construct_token_tensors(batch_img, idx)

        cand_sz = token_tensors.shape[1]
        logits = torch.empty([len(batch_img), cand_sz], device=self.device, dtype=self.clip_model.dtype)
        # token_tensors => (batch_sz, prompt_len, vocab_dim)
        _batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        # TODO: Parallelize without loop
        for img_id, img_feat in enumerate(_batch_img):
            text_features = self.text_encoder(
                self.token_ids,
                token_tensors[img_id],
                enable_pos_emb=self.enable_pos_emb,
            )
            _text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits[img_id] = img_feat @ _text_features.t()

        logits *= self.clip_model.logit_scale.exp()

        return logits


def get_cocsp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    vocab_sz = soft_embedding.shape[-2]
    vis_dim = soft_embedding.shape[-1]

    vctx_encoder = VisualCtxEncoder(vis_dim).to(device)

    optimizer = torch.optim.Adam(
        [soft_embedding] + list(vctx_encoder.parameters()),
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

