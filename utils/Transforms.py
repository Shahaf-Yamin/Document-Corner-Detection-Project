import collections
import numbers
import random
from PIL import Image
import numpy as np
from torchvision import transforms

def _iterate_transforms(transforms, x):
    if isinstance(transforms, collections.Iterable):
        for i, transform in enumerate(transforms):
            x[i] = _iterate_transforms(transform, x[i])
    else:
        x = transforms(x)
    return x

# we can pass nested arrays inside Compose
# the first level will be applied to all inputs
# and nested levels are passed to nested transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = _iterate_transforms(transform, x)
        return x

class RandomCropGenerator(object):
    def __call__(self, img):
        self.x1 = random.uniform(0, 1)
        self.y1 = random.uniform(0, 1)
        return img

class RandomCropNearCorner(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.T.astype('uint8'))
        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img
        corner = random.randint(0, 3)
        x_center = corners[(corner*2)] * w
        y_center = corners[(corner*2)+1] * h
        randTw = random.randint(0, tw)
        randTh = random.randint(0, th)
        x1 = max(x_center - randTw, 0)
        y1 = max(y_center - randTh, 0)
        x2 = min(x1 + tw, w)
        y2 = min(y1 + th, h)
        if x2 == w:
            x1 = w - tw
        if y2 == h:
            y1 = h - th
        cropped_img = np.array(img.crop((x1, y1, x2, y2))).T
        cropped_mask = np.array(mask.crop((x1, y1, x2, y2))).T
        new_cor = [(corners[(corner*2)] * w - x1) / tw, (corners[(corner*2)+1] * h - y1) / th]

        return cropped_img, cropped_mask, new_cor

class UniformRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *args):
        img, mask, corners =tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.T.astype('uint8'))
        w, h = np.array(img.size) - 1
        th, tw = self.size

        if w+1 == tw and h+1 == th:
            return img

        rand_W = random.randint(0, w)
        rand_H = random.randint(0, h)
        randTw = random.randint(0, tw)
        randTh = random.randint(0, th)

        x1 = max(rand_W - randTw, 0)
        y1 = max(rand_H - randTh, 0)
        x2 = min(x1 + tw, w)
        y2 = min(y1 + th, h)

        if x2 == w:
            x1 = w - tw
        if y2 == h:
            y1 = h - th

        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_mask = mask.crop((x1, y1, x2, y2))
        # plt.figure()
        # plt.imshow(cropped_img)
        # plt.show()
        # plt.figure()
        # plt.imshow(cropped_mask)
        # plt.show()
        return np.array(cropped_img).T, np.array(cropped_mask).T, corners

class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.T.astype('uint8'))

        corners = np.array(list(map(lambda x: np.array((x[0] * self.size[0] / img.size[0], x[1] * self.size[1] / img.size[1])), corners)))

        resized_img = img.resize(self.size)
        resized_mask = mask.resize(self.size)

        return np.array(resized_img).T, np.array(resized_mask).T, corners

class GaussianBlur(object):
    def __init__(self, size):
        self.filter = transforms.GaussianBlur(size)

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = Image.fromarray(img.T.astype('uint8'), 'RGB')
        img = np.array(self.filter(img)).T
        return img, mask, corners

class Normalize(object):
    def __init__(self, *args):
        self.mean, self.std = tuple(*args)

    def __call__(self, *args):
        img, mask, corners = tuple(*args)
        img = (img - self.mean) / self.std

        return img, mask, corners

