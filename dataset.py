import os
import torch
from glob import glob
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms


IMG_EXTENSIONS = ['.png', '.jpg']


class ToSquare:
    """Crop a ``PIL Image`` at the center to make it square. This transform does not support torchscript.

    Crops a PIL Image (H x W x C) to (R x R x C) where R=min(H,W).
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Square PIL Image.
        """
        R = min(pic.size)
        return transforms.functional.center_crop(pic, (R, R))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class HighPass(torch.nn.Module):
    """Apply high pass filter to Image.
    This transform does not support PIL Image.
    """

    def __init__(self):
        super().__init__()
        self.filter = torch.tensor([[1, -1], [-1, 1]])
        self.padding = (1, 1, 1, 1)


    def forward(self, x):
        """
        Args:
            tensor (Tensor): Tensor image to be high pass filtered.

        Returns:
            Tensor: Filtered Tensor image.
        """
        # Pad input with reflection padding
        C, H, W = x.shape
        x = x.unsqueeze(0)
        x = torch.nn.functional.pad(x, self.padding, mode='reflect')


        # Convolve with the filter to filter high frequencies.
        f = self.filter.view(1, 1, 2, 2).repeat(C, 1, 1, 1).to(x.device, x.dtype)
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=C).squeeze(0)[:, :H, :W]
        return x


    def __repr__(self):
        return self.__class__.__name__ + '(filter={0}, padding={1})'.format(self.filter.tolist(), self.padding)


def get_dataset(cfg):
    tf = transforms.Compose([
        ToSquare(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))          # [0, 1] -> [-1, 1]
    ])
    if cfg.data.get('highpass', False):
        tf.transforms.append(HighPass())
    z_dim = cfg.model.get('z_dim', 1)
    dset = ImageLatentDataset(cfg.data.root, z_dim, transform=tf)
    H, W = dset[0][0].shape[1:]
    if H != W:
        raise RuntimeError(f'Images need to be square but have H={H} and W={W}.')
    resolution = cfg.data.get('resolution', H)
    if H != cfg.data.resolution:
        print(f'Resize images from {H} to {cfg.data.resolution} using Lanczos filter.')
        tf.transforms.insert(1, transforms.Resize(cfg.data.resolution, interpolation=transforms.InterpolationMode.LANCZOS))
    assert dset[0][0].shape[1] == resolution
    dset.resolution = resolution
    dset.highpass = cfg.data.get('highpass', False)

    if z_dim == 1:      # replace z by index, used for discriminator testbed
        dset.z = torch.arange(len(dset))

    if cfg.data.subset is not None:
        subset_idcs = range(cfg.data.subset)
        subset = torch.utils.data.Subset(dset, subset_idcs)

        # inherit attributes
        subset.root = dset.root
        subset.z = dset.z[subset.indices]
        subset.resolution = dset.resolution
        subset.highpass = dset.highpass

        dset = subset

    return dset


class ImageFolder(VisionDataset):
    """Loads all images in given root
    """
    def __init__(self, root, transform=None):
        super(ImageFolder, self).__init__(root, transform=transform)
        self.img_paths = [p for p in glob(os.path.join(root, '*')) if os.path.splitext(p)[-1] in IMG_EXTENSIONS]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageLatentDataset(ImageFolder):
    """Wrapper class which pairs each image with a fixed latent code."""
    def __init__(self, root, z_dim, transform=None):
        super(ImageLatentDataset, self).__init__(root, transform=transform)
        self.z = torch.randn(len(self), z_dim)

    def __getitem__(self, idx):
        img = super(ImageLatentDataset, self).__getitem__(idx)
        return img, self.z[idx]
