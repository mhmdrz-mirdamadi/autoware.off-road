#! /usr/bin/env python3

import pathlib
from PIL import Image
import numpy as np


DATASETS = ['CaSSeD', 'Goose', 'OFFSED', 'ORFD', 'Rellis_3D', 'Yamaha_CMU']


class LoadDataFreespaceSeg:
    def __init__(self, labels_filepath, images_filepath, dataset: str):

        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError('Dataset type is not correctly specified')

        self.labels = sorted(
            [f for f in pathlib.Path(labels_filepath).glob('*.png')])
        self.images = sorted(
            [f for f in pathlib.Path(images_filepath).glob('*.png')])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if (self.num_images != self.num_labels):
            raise ValueError(
                'Number of images and ground truth labels are mismatched')

        if (self.num_images == 0):
            raise ValueError('No images found - check the root path')

        if (self.num_labels == 0):
            raise ValueError(
                'No ground truth masks found - check the root path')

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.num_train_samples = 0
        self.num_val_samples = 0

        for count in range(self.num_images):
            if (count + 1) % 10 == 0:
                self.val_images.append(str(self.images[count]))
                self.val_labels.append(str(self.labels[count]))
                self.num_val_samples += 1
            else:
                self.train_images.append(str(self.images[count]))
                self.train_labels.append(str(self.labels[count]))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def createGroundTruth(self, input_label):
        road_colour = (0, 255, 220)

        # Convert input PIL image to NumPy array
        input_np = np.array(input_label)
        row, col, _ = input_np.shape

        # Initialize masks and apply
        ground_truth = np.zeros((row, col), dtype=np.uint8)
        road_mask = np.isin(input_np.reshape(-1, 3),
                            road_colour).all(axis=1).reshape(row, col)
        ground_truth[road_mask] = 255

        return ground_truth

    def getItemTrain(self, index):
        self.train_image = Image.open(
            str(self.train_images[index])).convert("RGB")
        self.train_label = Image.open(str(self.train_labels[index]))
        self.train_ground_truth = self.createGroundTruth(self.train_label)
        self.train_ground_truth = np.expand_dims(
            self.train_ground_truth, axis=-1)

        return np.array(self.train_image), self.train_ground_truth

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])

    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index])).convert("RGB")
        self.val_label = Image.open(str(self.val_labels[index]))
        self.val_ground_truth = self.createGroundTruth(self.val_label)
        self.val_ground_truth = np.expand_dims(
            self.val_ground_truth, axis=-1)

        return np.array(self.val_image), self.val_ground_truth

    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
