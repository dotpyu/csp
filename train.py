#
import argparse
import os
import pickle
import pprint
from copy import deepcopy as dc
import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from utils import set_seed

try:
    from torch.cuda import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train_model(model, optimizer, train_dataset, config, device, epoch_offset=0):
    """Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataset (CompositionDataset): the train dataset
        config (argparse.ArgumentParser): the config
        device (...): torch device

    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()
    #
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(device)
    i = 0
    train_losses = []

    # torch.autograd.set_detect_anomaly(True)

    if config.amp:
        scaler = amp.GradScaler()

    for i in range(epoch_offset, epoch_offset+config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img, batch_target = batch[0], batch[3]
            batch_target = batch_target.to(device)
            batch_img = batch_img.to(device)
            batch_feat = model.encode_image(batch_img)
           
            with torch.autocast(device_type="cuda"): logits = model(batch_feat, train_pairs)

            loss = loss_fn(logits, batch_target)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            if config.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # weights update[
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                    (bid + 1 == len(train_dataloader)):
                if config.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": np.mean(epoch_train_losses[-50:])}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, epoch=i + 1)

    return model, optimizer


def save_soft_embeddings(model, config, epoch=None):
    """Function to save soft embeddings.

    Args:
        model (nn.Module): the CSP/COOP module
        config (argparse.ArgumentParser): the config
        epoch (int, optional): epoch number for the soft embedding.
            Defaults to None.
    """
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    if config.finetune:
        if epoch:
            model_save_path = os.path.join(
                config.save_path, f"finetune_model_{epoch}.pth"
            )
        else:
            model_save_path = os.path.join(
                config.save_path, f"finetune_model.pth"
            )
        torch.save(model.state_dict(), model_save_path)
        return

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
            vctx_path = os.path.join(
                config.save_path, f"vctx_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )
            vctx_path = os.path.join(
                config.save_path, f"vctx_{epoch}.pt"
            )

        torch.save({"soft_embeddings": model.soft_embeddings}, soft_emb_path)
        if config.experiment_name == 'cocsp' or config.experiment_name == 'cocoop':
            torch.save({"vis_context_encoder": model.vctx_encoder}, vctx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )

    parser.add_argument(
        "--continue_ckpt",
        help="continue previous training course",
        action="store_true",
    )

    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )

    parser.add_argument("--finetune", help="finetune the model", action="store_true")
    parser.add_argument("--amp", help="mix precision training", action="store_true")
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )

    config = parser.parse_args()
    config.amp = config.amp and APEX_AVAILABLE
    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    if os.path.exists(config.save_path):
        print('file already exists')
        #print('exiting!')
        #exit(0)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    model, optimizer = get_model(train_dataset, config, device)

    print("model dtype", model.dtype)
    print("soft embedding dtype", model.soft_embeddings.dtype)

    if not config.evaluate_only:
        epoch_offset = 0
        if config.continue_ckpt:
            # automatically discover ckpt, only supports co* for now
            vctx_version = -1
            se_version = -1
            se_path_template = config.save_path + "/soft_embeddings_epoch_{:d}.pt"
            for i in range(config.epochs):
                if os.path.exists(dc(se_path_template).format(i+1)):
                    se_version = i+1 # 1-indexed epoch
                else: break
            if se_version != -1:
                if config.experiment_name == 'cocoop' or 'cocsp':
                    model.reset_trainables()
                    vctx_template = config.save_path + "/vctx_epoch_{:d}.pt"
                    for i in range(config.epochs):
                        if os.path.exists(dc(vctx_template).format(i + 1)):
                            vctx_version = i + 1  # 1-indexed epoch
                        else:
                            break
                    final_version = min(se_version, vctx_version)
                    vctx_path = dc(vctx_template).format(final_version)
                    model.load_vctx_encoder(torch.load(vctx_path, map_location='cpu')['vis_context_encoder'])
                else:
                    final_version = se_version
                    model.self_embeddings = None
                    torch.cuda.empty_cache()
                se_path = dc(se_path_template).format(final_version)
                se = torch.load(se_path, map_location='cpu')['soft_embeddings']
                model.soft_embeddings = torch.nn.Parameter(se).to(model.device)
            epoch_offset = final_version

        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            device,
            epoch_offset = epoch_offset,
        )

    save_soft_embeddings(
        model,
        config,
    )

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)

    if config.save_model:
        torch.save(
            model.dict(),
            os.path.join(
                config.save_path,
                'final_model.pt'))

    print("done!")
