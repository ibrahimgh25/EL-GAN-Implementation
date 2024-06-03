import unittest
import json
from unittest.mock import patch, mock_open

import cv2
import numpy as np

from src.data.dataset import TuSimpleDataset
from src.data.utils import makedir_if_not_exists, save_annotated_image


class TestTuSimpleDataset(unittest.TestCase):
    def setUp(self):
        self.ds_dir = "test/resources/Dummy Dataset"
        self.labels_file = "labels.json"

    def test_length_asExpected(self):
        ds = TuSimpleDataset(self.labels_file, self.ds_dir)
        self.assertEquals(len(ds), 2)

    def test_getitem_correctChannels(self):
        img, gt = self.get_zeroth_sample()
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(gt.shape[0], 1)

    def get_zeroth_sample(self):
        ds = TuSimpleDataset(self.labels_file, self.ds_dir)
        img, gt = ds[0]
        return img, gt

    def test_getitem_matchingSize(self):
        img, gt = self.get_zeroth_sample()
        self.assertEqual(img.shape[1], gt.shape[1])
        self.assertEqual(img.shape[2], gt.shape[2])

    def test_saveAnnotatedImage_noException(self):
        # This is just here for sanity check rather than testing save_annotated_image, it still needs checking
        # the output file manually
        img, gt = self.get_zeroth_sample()
        makedir_if_not_exists("test/output")
        save_annotated_image(img, gt, "test/output/combined_img.jpg")


if __name__ == "__main__":
    unittest.main()
