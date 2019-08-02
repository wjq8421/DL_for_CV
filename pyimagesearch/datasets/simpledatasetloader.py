"""
load small image datasets from disk
"""

import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        # first need to resize an image to a fixed size,
        # then perform some sort of scaling,
        # followed by converting the image array to a format suitable for Keras
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # imagePaths: specifying the file paths to the images in our dataset residing on disk
        # verbose: used to print updates to a console
        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            # 目录结构: /dataset_name/class/image.jpg
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)

            # handle printing updates to our console
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        return np.array(data), np.array(labels)
