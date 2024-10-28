import argparse

import os
import torch
import wandb

import numpy as np
from itertools import chain

from tqdm import tqdm, trange
import torchvision.transforms as T

from models.ad_models import FeatureExtractors
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from dataset2D import *
from models.dataset import BaseAnomalyDetectionDataset
def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project='AD',
        name=model_name
    )
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    common =   T.Compose([
            SquarePad(),
            T.Resize((224, 224), interpolation = T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
            ])


    # Dataloader.
    train_loader = get_dataloader(args.dataset_path, common, common, 64, 16, True)

    # Feature extractors.
    feature_extractor = FeatureExtractors(device=device)

    # Model instantiation.
    FAD_LLToClean = FeatureProjectionMLP(in_features=768, out_features=768)

    optimizer = torch.optim.Adam(params=chain(FAD_LLToClean.parameters()))

    FAD_LLToClean.to(device)
    feature_extractor.to(device)

    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    for epoch in trange(args.epochs_no, desc=f'Training Feature Transfer Net.{args.class_name}'):
        FAD_LLToClean.train()
        epoch_cos_sim = []
        for i, (images, lowlight) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs_no}')):
            images, lowlight = images.to(device), lowlight.to(device)
            with torch.no_grad():
                features, features_lowlight = feature_extractor(
                    images, lowlight)

            transfer_features = FAD_LLToClean(features_lowlight)

            loss = 1 - metric(features, transfer_features).mean()

            epoch_cos_sim.append(loss.item())
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": np.mean(epoch_cos_sim),
        })
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                FAD_LLToClean.state_dict(),
                f'{args.checkpoint_folder}/{args.class_name}/FAD_LLToClean_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Crossmodal Feature Networks (FADs) on a dataset.')

    parser.add_argument('--dataset_path', default='data/Normal', type=str,
                        help='Dataset path.')

    parser.add_argument('--checkpoint_folder', default='checkpoints', type=str,
                        help='Where to save the model checkpoints.')

    parser.add_argument('--class_name', default = "Bowl", type = str, choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"],
                        help = 'Category name.')

    parser.add_argument('--epochs_no', default=10, type=int,
                        help='Number of epochs to train the FADs.')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--save_interval', default=5, type=int,
                        help='Number of epochs to train the FADs.')

    args = parser.parse_args()
    train(args)
