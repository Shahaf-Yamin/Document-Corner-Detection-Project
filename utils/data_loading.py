import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import utils
from utils.Transforms import UniformRandomCrop, Resize, GaussianBlur, Normalize, RandomCropNearCorner
from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class DocumentDataset(object):
    def __init__(self):
        self.images, self.masks, self.labels = utils.generate_dataset()
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return {
            'image': torch.as_tensor(self.images[idx].copy()).float().contiguous(),
            'mask': torch.as_tensor(self.masks[idx].copy()).long().contiguous(),
            'label': torch.as_tensor(self.labels[idx].copy()).long().contiguous()
        }

class MaskDocumentDataset(object):
    def __init__(self, path):
        self.masks, self.labels = utils.LoadMaskDocumentDataset(path)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return {
            'mask': torch.as_tensor(self.masks[idx].copy()).long().contiguous(),
            'label': torch.as_tensor(self.labels[idx].copy()).long().contiguous()
        }

class MaskCornerDataset(object):
    def __init__(self, path):
        self.masks, self.labels = utils.LoadMaskCornerDataset(path)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return {
            'mask': torch.as_tensor(self.masks[idx].copy()).long().contiguous(),
            'label': torch.as_tensor(self.labels[idx].copy()).long().contiguous()
        }

class FullResolutionCroppedDocumentDataset(object):
    def __init__(self, CropSize, Path):
        self.images, self.masks, self.labels = utils.generate_full_resolution_partial_dataset(Path)
        self.Cropper = UniformRandomCrop(CropSize)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.Cropper(self.images[idx], self.masks[idx], self.labels[idx])
        return {
            'image': torch.as_tensor(image).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(self.labels[idx].copy()).long().contiguous()
        }

class FullResolutionResizedDocumentDataset(object):
    def __init__(self, ResizeSize, Path):
        self.images, self.masks, self.labels = utils.generate_full_resolution_partial_dataset(Path)
        self.Resizer = Resize(ResizeSize)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask, label = self.Resizer(self.images[idx], self.masks[idx], self.labels[idx])
        return {
            'image': torch.as_tensor(image).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(label.copy()).long().contiguous()
        }

TransformsMapping = {'Resize': Resize, 'UniformRandomCrop': UniformRandomCrop, 'GaussianBlur': GaussianBlur, 'Normalize': Normalize, 'RandomCropNearCorner': RandomCropNearCorner}

class PartialDocumentDatasetMaskSegmentation(object):
    def __init__(self, Path, Transforms: dict, Size : int):
        self.images, self.masks, self.labels, self.stats = utils.generate_full_resolution_partial_dataset(Path, size=Size)
        self.withTransforms = len(Transforms) != 0
        try:
            if Transforms['Normalize'] is None:
                Transforms['Normalize'] = (self.stats['mean'], self.stats['std'])
        except KeyError:
            pass
        transform_list = [TransformsMapping[transform](value) for transform, value in Transforms.items()]
        self.transforms = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.withTransforms:
            image, mask, label = self.transforms((self.images[idx], self.masks[idx], self.labels[idx]))
        else:
            image, mask, label = self.images[idx], self.masks[idx], self.labels[idx]
        return {
            'image': (torch.as_tensor(image).float().contiguous()),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'label': torch.as_tensor(label.copy()).float().contiguous()
        }

class FullyConnectedDocumentDataset(object):
    def __init__(self, masks, labels):
        self.masks, self.labels = masks, labels
    def __len__(self):
        return len(self.masks)
    def __getitem__(self, idx):
        return {
            'mask': torch.as_tensor(self.masks[idx].copy()).long().contiguous(),
            'label': torch.as_tensor(self.labels[idx].copy()).long().contiguous()
        }

def normalize_data_set(loader):
    '''
    Calculates the mean and stadart deviation of the data from loader
    '''
    total_sum = 0
    # Assuming that the data is an image with size of 64 x 64 pixels
    num_of_pixels = 4096
    dataset_size = len(loader.dataset.labels)
    for batch in loader:
        total_sum += batch['mask'].sum()
    mean = total_sum / (num_of_pixels * dataset_size)
    sum_of_squared_error = 0
    for batch in loader:
        sum_of_squared_error += ((batch['mask'] - mean).pow(2)).sum()
    std = torch.sqrt(sum_of_squared_error / (num_of_pixels * dataset_size))
    return mean, std