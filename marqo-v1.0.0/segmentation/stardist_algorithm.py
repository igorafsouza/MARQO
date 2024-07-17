import os
import json
import yaml

from stardist.models import StarDist2D

class StardistSegmentation:
    """
    Class to instantiate StarDist2D model
    """

    def __init__(self, model = 'HE'):
        self.model_folder = self.get_model(model)
        self.threshold_file = 'thresholds.json'


    def get_model(self, model):
        if model == 'IF':
            model_folder = 'models/2D_versatile_fluo/'

        elif model == 'HE':
            model_folder = 'models/model-main/'

        return model_folder
    

    def set_parameters(self, config_file):
        """
        
        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)


        try:
            self.nms = config_data['parameters']['segmentation']['nms']
            self.prob = config_data['parameters']['segmentation']['prob']
            self.stardist_model_folder = config_data['directories']['stardist_models']

        except:
            self.nms = config_data['parameters']['nms']
            self.prob = config_data['parameters']['prob']
            self.stardist_model_folder = '/sc/arion/projects/GnjaticLab/MICSSS/MARQO_v1/segmentation'


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
        
        return model2D

