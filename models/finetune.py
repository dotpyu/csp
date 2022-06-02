import argparse
import os
from copy import deepcopy as dc
import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_ft(train_dataset, config, device, prompt_template="a photo of [attr] [obj]"):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    offset = len(attributes)

    ft = Finetune(
        clip_model,
        config,
        offset,
        attributes,
        classes,
        None,
        prompt_template=prompt_template,
        device=device,
        enable_pos_emb=True,
    )

    optimizer = torch.optim.Adam(
        ft.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if config.amp:
        ft, optimizer = amp.initialize(
            ft, optimizer, opt_level="O2", loss_scale="dynamic"
        )

    return ft, optimizer


class Finetune(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset,
        attributes,
        objects,
        token_ids: torch.tensor,
        prompt_template="a photo of [attr] [obj]",
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = False,
    ):
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=torch.zeros([len(attributes)]),
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.offset = offset
        for params in self.text_encoder.parameters():
            params.requires_grad = True
        self.attributes = attributes
        self.objects = objects
        self.prompt_template = prompt_template
        self.text_encoder = self.text_encoder.type(torch.float32)

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        # class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        # token_tensor = self.clip_model.token_embedding(
        #     class_token_ids.to(self.device)
        # ).type(self.clip_model.dtype)
        prompts = [dc(self.prompt_template).replace('[attr]', self.attributes[attr_idx]).replace('[obj]', self.objects[obj_idx]) for attr_idx, obj_idx in zip(attr_idx, obj_idx)]
        tokenized = clip.tokenize(prompts, context_length=self.config.context_length)
        return tokenized

    def forward(self, batch_img, idx):
        batch_img = batch_img.to(self.device).to(self.dtype)

        tokenized = self.construct_token_tensors(idx).to(self.device)
        text_features = self.text_encoder(
            tokenized,
            None,
            enable_pos_emb=True,
        ).to(self.device)

        _text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        logits = (
            self.clip_model.logit_scale.exp()#.to(self.dtype)
            * normalized_img
            @ _text_features.t()
        )
        return logits
