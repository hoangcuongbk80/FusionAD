import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import kornia as K
import kornia.augmentation as Aug
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os


class LowLightDataset(Dataset):
    def __init__(self, image_folder, transform=None, low_light_transform=None, extensions=".jpg"):
        """
        Args:
            image_paths (list): List of paths to the image files.
            transform (callable, optional): Optional transform to be applied to both original and low-light images.
            low_light_transform (callable, optional): Transform to apply to simulate low-light conditions.
        """
        self.image_paths = image_folder
        self.transform = transform
        self.low_light_transform = low_light_transform

        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                            if f.lower().endswith(extensions)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = T.ToTensor()(image)  # Convert to tensor (C, H, W)

        # Apply the transform to the original image if specified
        if self.transform:
            original_image = self.transform(image)
        else:
            original_image = image

        # Apply the low-light augmentation to simulate low light conditions
        if self.low_light_transform:
            low_light_image = self.low_light_transform(image)
        else:
            low_light_image = image

        return original_image, low_light_image


# Kornia-based low-light transform that darkens part of the image
class KorniaLowLightWithShadowTransform:
    def __init__(self, brightness_factor_range=(-0.5, -0.3), roughness=(0.1, 0.7), shade_intensity=(-1.0, 0.0), shade_quantity=(0.0, 1.0), p=0.5):
        self.brightness_factor_range = brightness_factor_range
        self.shadow_transform = Aug.RandomPlasmaShadow(
            roughness=roughness,
            shade_intensity=shade_intensity,
            shade_quantity=shade_quantity,
            p=p
        )

    def __call__(self, image):
        # Random brightness reduction for low-light simulation
        brightness_factor = torch.FloatTensor(1).uniform_(
            *self.brightness_factor_range).item()
        low_light_image = K.enhance.adjust_brightness(image, brightness_factor)

        # Apply RandomPlasmaShadow for shadow effect
        low_light_image_with_shadow = self.shadow_transform(
            low_light_image.unsqueeze(0)).squeeze(0)

        return low_light_image_with_shadow


class GlobalLowLightTransform:
    def __init__(self, brightness_factor_range=(0.3, 0.7)):
        self.brightness_factor_range = brightness_factor_range

    def __call__(self, image):
        brightness_factor = torch.FloatTensor(1).uniform_(
            *self.brightness_factor_range).item()
        low_light_image = K.enhance.adjust_brightness(image, brightness_factor)
        return low_light_image


class AdditiveGaussianNoiseTransform:
    # Additive Noise: Adding noise to simulate camera sensor noise.
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = torch.clamp(
            image + noise, 0.0, 1.0)  # Clamp to valid range
        return noisy_image


class VignetteLowLightTransform:
    def __init__(self, strength=0.5):
        self.strength = strength

    def __call__(self, image):
        _, h, w = image.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        center_y, center_x = h // 2, w // 2
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        dist = dist / torch.max(dist)

        vignette_mask = 1 - self.strength * dist.unsqueeze(0)
        vignette_image = torch.clamp(image * vignette_mask, 0.0, 1.0)
        return vignette_image


class ColorShiftLowLightTransform:
    def __call__(self, image):
        # Convert to grayscale and blend to reduce saturation
        grayscale = K.color.rgb_to_grayscale(image).repeat(3, 1, 1)
        desaturated_image = 0.7 * image + 0.3 * grayscale

        # Add a slight blue tint
        blue_tint = torch.tensor([0.9, 0.9, 1.1], dtype=image.dtype).view(
            3, 1, 1).to(image.device)
        low_light_image = torch.clamp(desaturated_image * blue_tint, 0.0, 1.0)

        return low_light_image


class ContrastReductionLowLightTransform:
    # Contrast Reduction: Lowering contrast for a more washed-out appearance.
    def __init__(self, contrast_factor_range=(0.5, 0.8)):
        self.contrast_factor_range = contrast_factor_range

    def __call__(self, image):
        contrast_factor = torch.FloatTensor(1).uniform_(
            *self.contrast_factor_range).item()
        low_contrast_image = K.enhance.adjust_contrast(image, contrast_factor)
        return low_contrast_image


class PatchLowLightTransform:
    # Patch-Based Dimming: Simulating uneven lighting by darkening random patches of the image.
    def __init__(self, patch_size_range=(0.2, 0.5), brightness_factor_range=(0.3, 0.7)):
        self.patch_size_range = patch_size_range
        self.brightness_factor_range = brightness_factor_range

    def __call__(self, image):
        _, h, w = image.shape
        patch_size = int(torch.FloatTensor(1).uniform_(
            *self.patch_size_range).item() * min(h, w))
        top_left_x = torch.randint(0, w - patch_size, (1,)).item()
        top_left_y = torch.randint(0, h - patch_size, (1,)).item()
        brightness_factor = torch.FloatTensor(1).uniform_(
            *self.brightness_factor_range).item()

        # Create a mask for the patch and apply brightness reduction
        mask = torch.ones_like(image)
        mask[:, top_left_y:top_left_y + patch_size,
             top_left_x:top_left_x + patch_size] *= brightness_factor
        low_light_image = image * mask

        return torch.clamp(low_light_image, 0.0, 1.0)


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
    data_loader = DataLoader(dataset=LowLightDataset(image_folder, transform=transform, low_light_transform=low_light_transform), batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=False, pin_memory=True)

    return data_loader


if __name__ == '__main__':
    # Sample image paths
    image_paths = "models"

    # Instantiate low-light transform
    low_light_transform = T.Compose([

        # ColorShiftLowLightTransform(),
        # ContrastReductionLowLightTransform(),
        # PatchLowLightTransform(),
        KorniaLowLightWithShadowTransform(p=1.0),
        # RandomLowLightTransform(),
        # VignetteLowLightTransform(),

    ])

    # Create the common transform
    common_transform = T.Compose([
        T.Resize((256, 256))
    ])

    # Create the dataset
    dataset = LowLightDataset(
        image_paths, transform=common_transform, low_light_transform=low_light_transform)

    # Example usage
    for original, low_light in dataset:
        # original is the original image
        # low_light is the image with simulated low-light conditions
        print("Original Image Shape:", original.shape)
        print("Low-Light Image Shape:", low_light.shape)

        show_images(original, low_light)
