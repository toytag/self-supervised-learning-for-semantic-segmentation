import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import ImageFilter


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, crop_size=512):
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.crop_size = crop_size
        self.random_resized_crop = transforms.RandomResizedCrop(
            crop_size, scale=(0.5, 1.))
        self.random_color_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.random_grayscale = transforms.RandomGrayscale(p=0.2)
        self.random_gaussian_blur = transforms.RandomApply(
            [GaussianBlur([.1, 2.])], p=0.5)
        self.random_horizontal_flip = CustomRandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _transform(self, x):
        x = self.random_color_jitter(x)
        x = self.random_grayscale(x)
        x = self.random_gaussian_blur(x)
        x, hflip = self.random_horizontal_flip(x)
        x = self.to_tensor(x)
        x = self.normalize(x)
        return x, hflip

    def __call__(self, x):
        x = self.random_resized_crop(x)
        q, hflip_q = self._transform(x)
        k, hflip_k = self._transform(x)
        return (q, k), hflip_q != hflip_k


class CustomRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Custom version of transforms.RandomHorizontalFlip that also outputs 
    an indicator showing whether the horizontal flip has taken place.
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
            Flipped: whether the horizontal flip has taken place.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
