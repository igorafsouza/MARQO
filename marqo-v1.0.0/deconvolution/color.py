import os
import yaml
from tqdm import tqdm
import numpy as np
import histomicstk as ts

from utils.image import ImageHandler

class ColorMatrix:
    
    def __init__(self):
        pass


    def set_parameters(self, config_file):
        """
        Read and fetch the paramaters and names needed to execute the masking.

        Paramters:
            - 

        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        self.sample_name = config_data['sample']['name']

        self.marker_name_list = config_data['sample']['markers']
        self.marco_output_path = config_data['directories']['marco_output']        
        self.raw_images_path = config_data['sample']['raw_images_full_path']
        self.mag_to_analyze_for_matrix = config_data['parameters']['image']['mag_to_analyze_for_matrix']


    def global_evaluation(self, registered_masks):

        for marker_index, image_path in enumerate(tqdm(self.raw_images_path)):

            decon_matrix_path = os.path.join(self.marco_output_path, f'{self.sample_name}_{self.marker_name_list[marker_index]}')

            SF_change = 20 / self.mag_to_analyze_for_matrix
            x_min = int(registered_masks[marker_index][0]/SF_change)
            y_min = int(registered_masks[marker_index][2]/SF_change)
            height = int((registered_masks[marker_index][3] - registered_masks[marker_index][2])/SF_change)
            width = int((registered_masks[marker_index][1] - registered_masks[marker_index][0])/SF_change)
            still_ROI = ImageHandler.acquire_image(image_path, y_min, x_min, height, width, self.mag_to_analyze_for_matrix)

            #init variables for deconv
            stain_color_map = ts.preprocessing.color_deconvolution.stain_color_map
            stain_color_map['aec'] = [0.27, 0.68, 0.68] #pre-dynamic values
            stains = ['hematoxylin',  # nuclei stain
                    'eosin',        # cytoplasm stain
                    'null',
                    'aec']         # set to null if input contains only two stains

            #determine background staining from down-res image background
            I_0 = 242

            #determine dynamic matrix
            w_est = ts.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(still_ROI, I_0)

            # Perform color deconvolution
            aec_channel_index = ts.preprocessing.color_deconvolution.find_stain_index(stain_color_map[stains[3]], w_est)
            hema_channel_index = ts.preprocessing.color_deconvolution.find_stain_index(stain_color_map[stains[0]], w_est)
            if hema_channel_index == aec_channel_index:
                color_deconv_matrix_temp = color_deconv_matrix_temp = np.array([stain_color_map['hematoxylin'], stain_color_map['aec'], stain_color_map['null']]).T
            
            else:
                residual_index = [0,1,2]
                residual_index.remove(hema_channel_index)
                residual_index.remove(aec_channel_index)
                residual_index = residual_index[0]
                color_deconv_matrix_temp = np.array([w_est.T[hema_channel_index],w_est.T[aec_channel_index],w_est.T[residual_index]]).T

            # decon_matrices_dict[slide].append(color_deconv_matrix_temp)
            np.save(decon_matrix_path, color_deconv_matrix_temp)


    @staticmethod
    def roi_evaluation(img):
        """
        
        Input:
            -> img: can be full image or image patch
        
        Output:
            <- colour deconvolution matrix (np.array) respective to the input image
        """

        #init variables for deconv
        stain_color_map = ts.preprocessing.color_deconvolution.stain_color_map
        stain_color_map['aec'] = [0.27, 0.68, 0.68] #pre-dynamic values
        stains = ['hematoxylin',  # nuclei stain
                'eosin',        # cytoplasm stain
                'null',
                'aec']         # set to null if input contains only two stains

        #determine background staining from down-res image background
        I_0 = 242

        #determine dynamic matrix
        w_est = ts.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img, I_0)

        # Perform color deconvolution
        aec_channel_index = ts.preprocessing.color_deconvolution.find_stain_index(stain_color_map[stains[3]], w_est)
        hema_channel_index = ts.preprocessing.color_deconvolution.find_stain_index(stain_color_map[stains[0]], w_est)
        
        if hema_channel_index == aec_channel_index:
            color_deconv_matrix_temp = np.array([stain_color_map['hematoxylin'], stain_color_map['aec'], stain_color_map['null']]).T
        
        else:
            residual_index = [0,1,2]
            residual_index.remove(hema_channel_index)
            residual_index.remove(aec_channel_index)
            residual_index = residual_index[0]
            color_deconv_matrix = np.array([w_est.T[hema_channel_index],w_est.T[aec_channel_index],w_est.T[residual_index]]).T
            
        return color_deconv_matrix
