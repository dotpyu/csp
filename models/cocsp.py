import os
import gc
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
    def __init__(self, vis_dim, prompt_vocab_sz=2, dtype=torch.float16):
        super(VisualCtxEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, prompt_vocab_sz))
        ])).to(dtype)

    def forward(self, x):
        return self.encoder(x)


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
        dtype=torch.float32,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            dtype=dtype,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)
        self.vctx_encoder = vctx_encoder.to(self.device)
        #self.text_encoder = self.text_encoder.to(self.dtype)
        # self.soft_embeddings = self.soft_embeddings.to(self.device)

    def reset_trainables(self):
        self.vctx_encoder = None
        self.soft_embeddings = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_vctx_encoder(self, vctx_encoder):
        self.vctx_encoder = vctx_encoder.to(self.device)

    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs)

    def construct_token_tensors(self, batch_img, pair_idx):

        vctx = self.vctx_encoder(batch_img)  # (batch, 2)

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).unsqueeze(0).repeat(len(batch_img),1,1,1)#.type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())

        '''
        RuntimeError: The size of tensor a (768) must match the size of tensor b (64) at non-singleton dimension 2

        '''
        # reshape vctx to be the same shape of soft_embeddings
        soft_embeddings = self.soft_embeddings.unsqueeze(0).expand(batch_img.shape[0], -1, -1)
        soft_embeddings = soft_embeddings.to(self.device)
        attr_visual_ctx = vctx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, self.offset, -1)
        obj_visual_ctx = vctx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, soft_embeddings.shape[1] - self.offset, -1)
        attr_emb = soft_embeddings[:, :self.offset, :]
        obj_emb = soft_embeddings[:, self.offset:, :]

        attr_emb = attr_emb + attr_visual_ctx
        obj_emb = obj_emb + obj_visual_ctx

        attr_emb = attr_emb[:, attr_idx, :]#.type(self.clip_model.dtype)
        obj_emb = obj_emb[:, obj_idx, :]#.type(self.clip_model.dtype)
        # print(attr_emb.expand(len(batch_img),-1).shape)

        token_tensor[:, :, eos_idx - 2, :] = attr_emb
        token_tensor[:, :, eos_idx - 1, :] = obj_emb

        return token_tensor

    def forward(self, batch_img, idx):

        batch_img = batch_img.to(self.device)#.to(self.dtype)
        token_tensors = self.construct_token_tensors(batch_img, idx)

        # token_tensors => # [BS, CAND_SZ, vocab_sz, vocab_dim]
        emb_dim = batch_img.shape[-1]
        _batch_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        _batch_img = _batch_img.unsqueeze(-1)

        batch_size, prompt_len, vocab_sz, vocab_dim = token_tensors.shape
        flat_token_tensors = token_tensors.view(batch_size * prompt_len, vocab_sz, vocab_dim)
        text_features = self.text_encoder(
            self.token_ids,
            flat_token_tensors.float(),
            enable_pos_emb=self.enable_pos_emb,
        )
        _text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        _text_features = _text_features.view(batch_size, prompt_len, emb_dim)

        logits = torch.matmul(_text_features, _batch_img).squeeze(-1)
        logits *= self.clip_model.logit_scale.exp()

        return logits


def get_cocsp(train_dataset, config, device):


    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    vis_dim = soft_embedding.shape[-1]

    vctx_encoder = VisualCtxEncoder(vis_dim, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        [soft_embedding] + list(vctx_encoder.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    with torch.autocast(device_type="cuda" if "cuda" in device or "gpu" in device else "cpu"): interface = CoCSPInterface(
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

