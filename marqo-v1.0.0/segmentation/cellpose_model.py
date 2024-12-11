import os
import json
import yaml
import numpy as np
import skimage.exposure as sk_exposure
from skimage import color

from utils.image import ImageHandler
from cellpose import models

class CellposeSegmentation:
    """
    Class to instantiate StarDist2D model
    """

    def __init__(self):
        self.model = models.Cellpose(model_type='nuclei')


    def set_parameters(self, config_file):
        """
        
        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

            self.technology = config_data['sample']['technology']
            self.flow = config_data['parameters']['segmentation']['flow']
            self.name = config_data['parameters']['segmentation']['model']
            self.cell_diameter = config_data['parameters']['segmentation']['cell_diameter']


    def load_model(self):
        self.model = models.Cellpose(model_type='nuclei')


    def transform_image(self, roi):
        if self.technology in ['if', 'cycif']:
            nbins=256
            clip_limit=0.01
            img_transformed = sk_exposure.equalize_adapthist(roi, nbins=nbins, clip_limit=clip_limit)

            return img_transformed

        else:
            hed = color.rgb2hed(roi)

            # Extract the hematoxylin channel (H)
            hematoxylin_channel = hed[:, :, 0]  # The first channel is hematoxylin
            hematoxylin_channel = (hematoxylin_channel * 255).astype(np.uint8)  # Convert to integers

            nbins=256
            clip_limit=0.01
            img_transformed = sk_exposure.equalize_adapthist(hematoxylin_channel, nbins=nbins, clip_limit=clip_limit)

            return img_transformed

