import argparse
import os
from copy import deepcopy as dc
import clip
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

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
        self.clip_model.text_projection.requires_grad = True
        self.attributes = attributes
        self.objects = objects
        self.prompt_template = prompt_template

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        # class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        # token_tensor = self.clip_model.token_embedding(
        #     class_token_ids.to(self.device)
        # ).type(self.clip_model.dtype)
        prompts = [dc(self.prompt_template).replace('[attr]', self.attributes[attr_idx]).replace('[obj]', self.objects[obj_idx]) for attr_idx, obj_idx in zip(attr_idx, obj_idx)]
        tokenized = torch.cat([clip.tokenize(prompts, context_length=self.config.context_length)])
        token_tensor = self.text_encoder(tokenized.to(self.device))

        return token_tensor
