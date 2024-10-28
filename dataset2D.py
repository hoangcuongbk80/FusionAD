from email.contentmanager import maintype
from utils.general_utils import SquarePad
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import kornia as K
import kornia.augmentation as Aug
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image

class LowLightDataset(Dataset):
    def __init__(self, image_folder, transform=None, low_light_transform=None, extensions=".jpg"):
        """
        Args:
            image_paths (list): List of paths to the image files.
            transform (callable, optional): Optional transform to be applied to both original and low-light images.
            low_light_transform (callable, optional): Transform to apply to simulate low-light conditions.
        """
        self.low_light_path = os.path.join(image_folder, "Low_light")
        self.well_light_path = os.path.join(image_folder, "Well_light")

        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform = transform

        self.low_light_transform = low_light_transform

        self.well_lit_images = [f for f in os.listdir(self.well_light_path) if f.lower().endswith(extensions)]
        self.low_light_images = [f for f in os.listdir(self.low_light_path) if f.lower().endswith(extensions)]
    def __len__(self):
        return len(os.listdir(self.low_light_path))

    def __getitem__(self, idx):
        # Load the image


        img_low_path =  self.low_light_images[idx]
        image_low = Image.open(os.path.join(self.low_light_path, img_low_path)).convert('RGB')
        # image_low = T.ToTensor()(image_low)  # Convert to tensor (C, H, W)



        img_high_path =  self.well_lit_images[idx]
        image_high = Image.open(os.path.join(self.well_light_path, img_high_path)).convert('RGB')
        # image_high = T.ToTensor()(image_high)

        # Apply the transform to the original image if specified
        if self.transform:
            original_image = self.transform(image_high)
        else:
            original_image = image_high

        # Apply the low-light augmentation to simulate low light conditions
        if self.low_light_transform:
            low_light_image = self.low_light_transform(image_low)
        else:
            low_light_image = image_low

        return original_image, low_light_image
    

class TestDataset(Dataset):
    def __init__(self, data_folder, gt_folder, transform=None, low_light_transform=None, extensions=".jpg"):
        """
        Args:
            data_folder (str): Path to the folder containing 'low_light' and 'well_light' images.
            gt_folder (str): Path to the folder containing ground truth images in PNG format.
            transform (callable, optional): Transform to be applied to well-lit images.
            low_light_transform (callable, optional): Transform to apply to low-light images.
            extensions (str): Image file extension for low-light and well-lit images.
        """
        self.low_light_path = os.path.join(data_folder, "low_light")
        self.well_light_path = os.path.join(data_folder, "well_light")
        self.gt_folder = gt_folder

        self.transform = transform
        self.low_light_transform = low_light_transform
        self.gt_transform = T.Compose([
            SquarePad(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()])

        # Get the list of images in the well-lit and low-light folders
        self.well_lit_images = sorted(
            [f for f in os.listdir(self.well_light_path) if f.lower().endswith(extensions)]
        )
        self.low_light_images = sorted(
            [f for f in os.listdir(self.low_light_path) if f.lower().endswith(extensions)]
        )

        # Ensure both folders contain the same number of images
        assert len(self.well_lit_images) == len(self.low_light_images), \
            "Mismatch in number of images between well-light and low-light folders."

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        # Load the low-light and well-lit images
        img_low_name = self.low_light_images[idx]
        img_high_name = self.well_lit_images[idx]

        image_low = Image.open(os.path.join(self.low_light_path, img_low_name)).convert('RGB')
        image_high = Image.open(os.path.join(self.well_light_path, img_high_name)).convert('RGB')

        # Load the ground truth mask (assumes corresponding .png file in gt_folder)
        gt_name = os.path.splitext(img_low_name)[0] + '.png'
        gt_path = os.path.join(self.gt_folder, gt_name)
        gt_mask = Image.open(gt_path).convert('L')  # Load as grayscale

        # Apply transforms to the images if specified
        if self.transform:
            original_image = self.transform(image_high)
        else:
            original_image = image_high  # Default to tensor conversion

        if self.low_light_transform:
            low_light_image = self.low_light_transform(image_low)
        else:
            low_light_image = image_low  # Default to tensor conversion

        # Convert ground truth mask to tensor
        gt_mask = self.gt_transform(gt_mask)

        return original_image, low_light_image, gt_mask

def show_images(original, low_light):
    # Convert from Tensor to NumPy and reshape for display
    original_np = original.permute(1, 2, 0).cpu().numpy()
    low_light_np = low_light.permute(1, 2, 0).cpu().numpy()

    # Plot the images using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Display the low-light image
    axs[1].imshow(low_light_np)
    axs[1].set_title("Low-Light Image")
    axs[1].axis("off")

    plt.show()

def get_dataloader(image_folder, transform, low_light_transform, batch_size=1, num_workers=1, shuffle=False):
    data_loader = DataLoader(dataset=LowLightDataset(image_folder, transform=transform, low_light_transform=transform), batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=False, pin_memory=True)

    return data_loader
if __name__ == '__main__':
    # Sample image paths
    image_paths = "data/Normal"

    # Create the common transform
    common_transform = T.Compose([
        T.Resize((256, 256))
    ])

    # Create the dataset
    dataset = LowLightDataset(
        image_paths, transform=common_transform, low_light_transform=common_transform)


    # Example usage
    # for original, low_light in dataset:
    #     # original is the original image
    #     # low_light is the image with simulated low-light conditions
    #     print("Original Image Shape:", original.shape)
    #     print("Low-Light Image Shape:", low_light.shape)
    #
    #     show_images(original, low_light)