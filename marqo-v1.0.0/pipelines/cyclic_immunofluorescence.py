
# whole namespaces
import yaml
import numpy as np
import pandas as pd

# specific functions form namespaces
from typing import List

# pipeline classes and functions
from utils.utils import Utils
from utils.file import FileHandler
from masking.binary_mask import Masking
from registration.rigid import RigidRegistration
from registration.elastic import ElasticRegistration
from segmentation.composite import CompositeSegmentation
from segmentation.stardist_algorithm import StardistSegmentation
from quantification.stain_signal import Signal
from tiling.map import TileMap
from tiling.extract_tiles import TileExtraction
from segmentation.single import SimpleSegmentation
from utils.spatial import Spatial



class CyIF:
    """
    Class to define the pipeline designed to process MICSSS images (original MARCO pipeline).
    In this class it's possible to find the main steps performed by it and the methods it imports and uses.
    """

    def __init__(self, config_file = None):
        self.technology = 'cyif'
        self.config_file = config_file


    def initialization(self, images_names, sample_name: str, n_images: str, output_resolution: int, raw_images_path: str, output_path: str):

        images_list = []
        cycles_metadata = {}
        for i in range(len(images_names)):
            # 1. Format all names to eliminate special characters
            cycle = images_names[i]
            cycles_metadata[i] = {'marker_name_list': Utils.remove_special_characters(cycle['marker_name_list']),
                                 'dna_marker': Utils.remove_special_characters(cycle['dna_marker']), 
                                 'cyto_distances': cycle['cyto_distances']
            }

            images_list.append(cycle['image_name'])

        self.images_names = images_list

        # 2. Stage the config file
        config_file = Utils.stage_config_file(sample_name, raw_images_path, output_path, 
                                output_resolution = output_resolution, 
                                marker_name_list = [f'cycle_{i}' for i in range(int(n_images))], 
                                technology = self.technology)
        
        pixel_sizes_per_image = Utils.get_pixelsize(images_list, raw_images_path, config_file)

        # Setting config file as class attribute
        self.config_file = config_file

        # 3. Stage the directories and update the config file
        Utils.stage_directories(config_file)

        # 4. Grab raw images to process and write it to the config file
        Utils.get_slides_full_path(images_list, config_file)

        Utils.add_cycles_metadata(cycles_metadata, config_file)       


    def tissue_masking(self, image_index: int):
        """
        
        """
        masking = Masking()
        masking.set_parameters(self.config_file, image_index)
        mask = masking.get_mask(image_index, image_type='IF')

        return mask


    def prelim_registration(self, image_index: int, tissue_masks: List):
        """
        """
        registration = RigidRegistration()
        registration.set_parameters(self.config_file)
        transformation_map = registration.get_transformation_map(image_index, tissue_masks)

        return transformation_map


    def tiling(self, registered_masks: List, tissue_masks: List):
        """
        
        """
        tile_map = TileMap()
        tile_map.set_parameters(self.config_file)
        return tile_map.get_tiling_map(registered_masks, tissue_masks)


    def analysis(self, paramaters_array):
        '''
        This function is called by a different entry-point (from the command-line). It will perform all steps 
        for tile analysis and return the Tile object. 

        1. Tile-based registration
        2. Segmentation
        3. Quantification
        '''
        roi_index = paramaters_array[0]
        registered_masks = paramaters_array[1]
        mother_coordinates = paramaters_array[2]
        
        # 1. Elastic registration is performed on the different dapi's channels
        registration = ElasticRegistration()
        registration.set_parameters(self.config_file)
        registered_rois, linear_shifts, visualization_figs, registered_dapis = registration.get_registered_dapis(mother_coordinates, registered_masks, roi_index)
        
        # 2. Segmentation only for DAPI channel
        stardist = StardistSegmentation(model='IF')
        stardist.set_parameters(self.config_file)
        stardist_model = stardist.load_model()

        composite = CompositeSegmentation()
        composite.set_parameters(self.config_file)
        results = composite.predict_dapi(registered_dapis, roi_index, visualization_figs, stardist_model)

        # 3. Quantification
        tile_tissue_percentage = paramaters_array[3]

        signal = Signal()
        signal.set_parameters(self.config_file)
        
        # I need to iterate over all channels, and the quantification will not deconvolute - different functions used for MICSSS and Singleplex
        results = signal.quantify_if_channels(roi_index, registered_rois, linear_shifts, 
                                  mother_coordinates, registered_masks, 
                                  tile_tissue_percentage, visualization_figs, 
                                  **results)


        # 4. Write the output files
        file_handler = FileHandler()
        file_handler.set_parameters(self.config_file)
        file_handler._stage_tmp_dir()

        rawcellmetrics_df, _, ROI_pixel_metrics = results['data_for_tile']
        allmarkers_geojson_list = results['geojson_data']

        clipped_rawcellmetrics_df, allmarkers_geojson_updated_list = self.denoising(roi_index, rawcellmetrics_df, allmarkers_geojson_list)

        # 5. Check if exsits cores map, and if it is, add info to rawcellmetrics
        if Utils.exist_cores(self.config_file):
            core_label = Spatial.get_core_label(mother_coordinates, self.config_file)
            file_handler.write_marco_output(roi_index, clipped_rawcellmetrics_df, ROI_pixel_metrics, mother_coordinates, core_label=core_label)

        else:
            file_handler.write_marco_output(roi_index, clipped_rawcellmetrics_df, ROI_pixel_metrics, mother_coordinates)

        file_handler.write_geojsons(roi_index, allmarkers_geojson_updated_list)


    def denoising(self, _file, roi_index):
        df_tmp = pd.read_csv(_file)

        return df_tmp

    def _fetch_parameters(self):
        return Utils.fetch_parameters(self.config_file)


    def _add_parameters(self, step_name, index, **kwargs):
        return Utils.add_parameters(self.config_file, step_name, index, **kwargs)
