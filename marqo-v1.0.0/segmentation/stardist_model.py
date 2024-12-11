import os
import json
import yaml
import skimage.exposure as sk_exposure
from skimage import color

from utils.image import ImageHandler
from stardist.models import StarDist2D

class StardistSegmentation:
    """
    Class to instantiate StarDist2D model
    """

    def __init__(self):
        self.threshold_file = 'thresholds.json'


    def set_parameters(self, config_file):
        """
        
        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

            self.technology = config_data['sample']['technology']
            self.nms = config_data['parameters']['segmentation']['nms']
            self.name = config_data['parameters']['segmentation']['model']
            self.prob = config_data['parameters']['segmentation']['prob']
            self.pixel_size = config_data['parameters']['segmentation']['pixel_size']
            self.stardist_model_folder = config_data['directories']['stardist_models']

        if self.technology in ['if', 'cycif']:
            self.model_folder = 'models/2D_versatile_fluo/'

        else:
            self.model_folder = 'models/model-main/'


    def write_thresholds(self):
        '''
        Function to write at run time new thresholds. But right now we're keeping the thresholds static in the thresholds.json file
        within the models' directories
        '''
        model_settings = {}
        model_settings['prob'] = self.prob
        model_settings['nms'] = self.nms

        threshold_file_path = os.path.join(self.stardist_model_folder, self.model_folder, self.threshold_file)

        with open(threshold_file_path, 'w+') as thres:
            json.dump(model_settings, thres)


    def load_model(self):
        model2D = StarDist2D(None, name = self.model_folder, basedir = self.stardist_model_folder)
        
        self.model = model2D


    def transform_image(self, roi):
        if self.technology in ['if', 'cycif']:
            nbins=256
            clip_limit=0.01
            roi = sk_exposure.equalize_adapthist(roi, nbins=nbins, clip_limit=clip_limit)

            img_transformed = ImageHandler.normalize_percentile(roi, 1, 99.8)

            return img_transformed

        else:
            img_transformed = ImageHandler.transform_image(roi, add_hemo=False)

            return img_transformed
