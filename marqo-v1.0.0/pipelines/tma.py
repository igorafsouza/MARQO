
# whole namespaces
import numpy as np

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
from deconvolution.color import ColorMatrix
from utils.spatial import Spatial
from denoising.cell_pruning import Denoising


class TMA:
    """
    Class to define the pipeline designed to process MICSSS images (original MARCO pipeline).
    In this class it's possible to find the main steps performed by it and the methods it imports and uses.
    """

    def __init__(self, parent_class = None):
        self.technology = 'tma'
        self.parent_class = parent_class


    def __getattr__(self, name):
        return self.parent_class.__getattribute__(name)


    def initialization(self, images_names: List[str], sample_name: str, marker_name_list: str, output_resolution: int, 
                       raw_images_path: str, output_path: str, cyto_distances: List[int]):
        
        # 1. Format all names to eliminate special characters
        marker_name_list = Utils.remove_special_characters(marker_name_list)

        # 2. Stage the config file
        config_file = Utils.stage_config_file(sample_name, raw_images_path, output_path, 
                                output_resolution = output_resolution, 
                                cyto_distances = cyto_distances, 
                                technology = self.technology,
                                marker_name_list = marker_name_list)
        
        # Setting config file as class attribute
        self.config_file = config_file

        # 3. Stage the directories and update the config file
        Utils.stage_directories(config_file)
        
        # 4. Grab raw images to process and write it to the config file
        Utils.get_slides_full_path(images_names, config_file)

        # 5. Define Scale Factor based on output resolution and image format
        # bot informations are inside the config file
        Utils.get_scale_factor(config_file)


    def tissue_masking(self, image_index: int):
        """
        
        """

        # 1. Set dinamically parameters for Masking Class
        masking = Masking()
        masking.set_parameters(self.config_file, image_index)
        mask = masking.get_mask(image_index)

        return mask

    
    def prelim_registration(self, image_index: int, tissue_masks: List):
        """
        Rigid (global) registration

        in:
            <-

        out:
            ->
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


    def deconvolution(self, registered_masks: List):
        """
        
        """
        color_matrix = ColorMatrix()
        color_matrix.set_parameters(self.config_file)
        color_matrix.global_evaluation(registered_masks)


    def analysis(self, paramaters_array):
        '''
        This function is called by a different entry-point (from the command-line). It will perform all steps 
        for tile analysis and return the Tile object. 

        1. Tile-based registration
        2. Segmentation
        3. Quantification
        '''                
        # 1. Tile-based registration: this function will receive a list of the tiles (list of 1 or N)
        roi_index = paramaters_array[0]
        registered_masks = paramaters_array[1]
        mother_coordinates = paramaters_array[2]

        color_deconv_matrices = Utils.get_decon_matrix(self.config_file)
        
        registration = ElasticRegistration()
        registration.set_parameters(self.config_file)
        registered_rois, linear_shifts, visualization_figs = registration.get_transformed_images(mother_coordinates, registered_masks, color_deconv_matrices, roi_index)

        # 2. Segmentation of the registred rois
        stardist = StardistSegmentation()
        stardist.set_parameters(self.config_file)
        stardist_model = stardist.load_model()

        composite = CompositeSegmentation()
        composite.set_parameters(self.config_file)
        
        results = composite.predict(registered_rois, roi_index, visualization_figs, stardist_model)

        # 3. Quantification over registred_rois and coordinates from segmentation
        tile_tissue_percentage = paramaters_array[3]

        signal = Signal()
        signal.set_parameters(self.config_file)
        results = signal.quantify(roi_index, registered_rois, linear_shifts, 
                                  mother_coordinates, registered_masks, color_deconv_matrices,
                                  tile_tissue_percentage, visualization_figs, 
                                  **results)

        # data_for_tile = rawcellmetrics_fortile_df, visualization_figs, ROI_pixel_metrics, updated_json_list_for_marker

        # 4. Write the output files
        file_handler = FileHandler()
        file_handler.set_parameters(self.config_file)

        # 5. Adding new column to associate each  cell to a specific TMA core
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


        # 2. Denoise the object Tile
        # - need to put some thought on this. I don't like the idea to prune permentaly the cells from the rawcellmetrics. What if 
        # the user chose an edge thickness larger than it should? What if the user notice afterwards that cells of interest lays on 
        # the removed edge? 
        

    def denoising(self, roi_index, rawcellmetrics_df, allmarkers_geojson_list):
        '''
        
        '''
        denos = Denoising()
        denos.set_parameters(self.config_file)

        tissue_mask = self.tissue_masking(denos.marker_index_maskforclipping)

        clipped_rawcellmetrics_df, allmarkers_geojson_updated_list = denos.denoise_tile_cellmetrics(roi_index, rawcellmetrics_df, allmarkers_geojson_list, tissue_mask)

        return clipped_rawcellmetrics_df, allmarkers_geojson_updated_list


    def _fetch_parameters(self):
        return Utils.fetch_parameters(self.config_file)


    def _add_parameters(self, step_name, index, **kwargs):
        return Utils.add_parameters(self.config_file, step_name, index, **kwargs)
