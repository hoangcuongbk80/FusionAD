import argparse
import os
import torch
from torchvision import transforms as T
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.features import MultimodalFeatures
# from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from models.ad_models import FeatureExtractors
from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score
from dataset2D import SquarePad, TestDataset


def set_seeds(sid=42):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def infer(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs"

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    common = T.Compose(
        [
            SquarePad(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Dataloader.
    test_loader = get_dataloader(
        "Anomaly", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path
    )

    # Feature extractors.
    feature_extractor = FeatureExtractors(device=device)
    

    # Model instantiation.
    FAD_LLToClean = FeatureProjectionMLP(in_features=768, out_features=768)
    

    FAD_LLToClean.to(device)
    feature_extractor.to(device)
    FAD_LLToClean.load_state_dict(torch.load(f"{args.checkpoint_folder}/{args.class_name}/FAD_LLToClean_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth"))

    FAD_LLToClean.eval()
    feature_extractor.eval()

    # Metrics.
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # Use box filters to approximate gaussian blur (https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf).
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device = device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device = device)/(w_u**2)

    predictions, gts = [], []
    pixel_labels = [], []
    image_preds, pixel_preds = [], []


    # Inference.
    # Assuming FeatureExtractor is for 2D images, and we use one FeatureProjectionMLP for projection.
# ------------ [Testing Loop] ------------ #

# * Return (img1, img2), gt[:1], label, img_path where img1 and img2 are both 2D images.
    for (img1, img2), gt in  tqdm(test_loader, desc=f'Extracting feature from class: {args.class_name}.'):

        # Move data to the GPU (or whatever device is being used)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            # Extract features from both 2D images
            img1_features,img2_features = feature_extractor(img1)  # Features from img2 (e.g., low-light image)
            
            # Project features from img2 into the same space as img1 using the FeatureProjectionMLP
            projected_img2_features = FAD_LLToClean(img2_features)  # FeatureProjectionMLP now projects between 2D features

            # Mask invalid features (if necessary)
            feature_mask = (img2_features.sum(axis=-1) == 0)  # Mask for img2 features that are all zeros.

            # Cosine distance between img1 features and projected img2 features
            cos_img1 = (torch.nn.functional.normalize(img1_features, dim=1) - 
                        torch.nn.functional.normalize(img1_features, dim=1)).pow(2).sum(1).sqrt()  
            cos_img2 = (torch.nn.functional.normalize(projected_img2_features, dim=1) - 
                        torch.nn.functional.normalize(img2_features, dim=1)).pow(2).sum(1).sqrt()  

            # Apply the mask to ignore zeroed out features
            cos_img1[feature_mask] = 0.
            cos_img1 = cos_img1.reshape(224, 224)

            cos_img2[feature_mask] = 0.
            cos_img2 = cos_img2.reshape(224, 224)

            # Combine the cosine distances from both feature sets
            cos_comb = (cos_img1 * cos_img2)
            cos_comb.reshape(-1)[feature_mask] = 0.

            # Apply smoothing (similarly as before) using conv2d
            cos_comb = cos_comb.reshape(1, 1, 224, 224)
            
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            
            cos_comb = cos_comb.reshape(224, 224)
            
            # Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy())  # (224, 224)
            predictions.append((cos_comb / (cos_comb[cos_comb != 0].mean())).cpu().detach().numpy())  # (224, 224)
            
            # GTs
            # image_labels.append()  # (1,)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())  # (50176,)

            # Predictions
            image_preds.append((cos_comb / torch.sqrt(cos_comb[cos_comb != 0].mean())).cpu().detach().numpy().max())  # single number
            pixel_preds.extend((cos_comb / torch.sqrt(cos_comb.mean())).flatten().cpu().detach().numpy())  # (224, 224)

   
                