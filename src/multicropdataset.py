# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()


def build_label_index(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label2inds.get(label) is None:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        subset=-1,
        ratio_minority_class=0.,
        ratio_step_size=1.,
        random_seed=0
    ):
        super(MultiCropDataset, self).__init__(data_path)

        # Subset
        if subset > 0:
            all_indices = []
            label2inds = build_label_index(self.targets)
            # Unbalanced dataset
            np.random.seed(random_seed)
            number_minority_class = int(ratio_minority_class * len(label2inds))
            minority_class_idx = set(np.random.choice(len(label2inds), number_minority_class, replace=False))
            for label, img_indices in label2inds.items():
                assert len(img_indices) >= subset
                if label in minority_class_idx:
                    minority_subset = int(subset / ratio_step_size)
                    all_indices += img_indices[:minority_subset]
                else:
                    all_indices += img_indices[:subset]
            self.imgs = [self.imgs[idx] for idx in all_indices]
            self.samples = [self.samples[idx] for idx in all_indices]
            self.targets = [self.targets[idx] for idx in all_indices]
            # assert len(self) == (subset * 1000)
        # print('Size dataset: ', len(self))

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
