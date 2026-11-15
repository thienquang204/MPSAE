"""Image/mask alignment, resizing and patch-extraction collation."""
import io
import numpy as np
import torch
from PIL import Image
from ..benchmark.data import open_rgb
from .datasets import VOC

def open_mask(value):
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as image:
                return image.copy()
        value = value.get('path')
    with Image.open(value) as image:
        return image.copy()


class ImageRows:
    def __init__(self, dataset, pending, size, normalize, spec=VOC):
        self.dataset, self.pending, self.size = dataset, pending, size
        self.normalize = normalize
        self.spec = spec

    def __len__(self):
        return len(self.pending)

    def __getitem__(self, position):
        from torchvision.transforms.functional import normalize, pil_to_tensor
        index = self.pending[position]
        row = self.dataset[index]
        image, mask = open_rgb(row['image']), open_mask(row[self.spec.mask_column])
        if image.size != mask.size:
            raise ValueError('image and segmentation mask are misaligned')
        truth = self.spec.remap(mask)
        height, width = (self.size, self.size) if isinstance(self.size, int) else self.size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        scaled = mask.resize((width, height), Image.Resampling.NEAREST)
        image = normalize(pil_to_tensor(image).float()/255, *self.normalize)
        return index, image, truth, self.spec.remap(scaled)


def collate(rows):
    return [r[0] for r in rows], torch.stack([r[1] for r in rows]), [r[2] for r in rows], [r[3] for r in rows]


