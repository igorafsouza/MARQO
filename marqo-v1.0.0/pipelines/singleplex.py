# whole namespaces
import os
import numpy as np

# specific functions form namespaces
from typing import Any, List

# pipeline classes and functions
from utils.utils import Utils
from utils.file import FileHandler
from masking.binary_mask import Masking
from registration.rigid import RigidRegistration
from segmentation.single import SimpleSegmentation
from segmentation.stardist_model import StardistSegmentation
from segmentation.cellpose_model import CellposeSegmentation
from quantification.stain_signal import Signal
from tiling.map import TileMap
from tiling.extract_tiles import TileExtraction
from deconvolution.color import ColorMatrix
from denoising.cell_pruning import Denoising
from utils.spatial import Spatial
from contextlib import redirect_stdout, redirect_stderr

class SinglePlex:
    '''
    Class to define the pipeline designed to process SinglePlex images (derived from MARCO pipeline).
    In this class it's possible to find the main steps performed by it and the methods it imports and uses.
    '''

    def __init__(self, parent_class = None):
        self.technology = 'singleplex'
        self.parent_class = parent_class

    def __getattr__(self, name):
        return self.parent_class.__getattribute__(name)
    
 
    def initialization(self, images_names: List[str], sample_name: str, marker_name_list: str, output_resolution: int, 
                       raw_images_path: str, output_path: str, cyto_distances: List[int], segmentation_model: dict):
        
        # 1. Format all names to eliminate special characters
        marker_name_list = Utils.remove_special_characters(marker_name_list)

        # 2. Stage the config file
        config_file = Utils.stage_config_file(sample_name, raw_images_path, output_path, 
                                output_resolution=output_resolution, 
                                cyto_distances=cyto_distances, 
                                marker_name_list=marker_name_list,
                                technology=self.technology,
                                segmentation_model=segmentation_model
                                )
        
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
        mask = masking.get_mask(image_index, image_type='HE')

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


    def denoising(self):
        '''
        
        '''
        # Loop over
        # Atomic functions
        pass


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
        segmentation_model = paramaters_array[4]

        color_deconv_matrices = Utils.get_decon_matrix(self.config_file)
 
        extraction = TileExtraction()
        extraction.set_parameters(self.config_file)
        # I need to store ROIs for all channels in registered_rois, although only DAPI is used for Segmentation
        registered_rois, linear_shifts, visualization_figs = extraction.roi_extraction(mother_coordinates, roi_index)

        # 2. Segmentation of ROIs
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):

                if segmentation_model == 'stardist':
                    model = StardistSegmentation()
                    model.set_parameters(self.config_file)
                    model.write_thresholds()
                    model.load_model()

                elif segmentation_model == 'cellpose':
                    model = CellposeSegmentation()
                    model.set_parameters(self.config_file)
                    model.load_model()

        single = SimpleSegmentation()
        single.set_parameters(self.config_file)
        
        results = single.predict(registered_rois, roi_index, visualization_figs, model)

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


    def clustering(self):
        '''
        '''
        # Atomic function, the entry point will handle with workload distribution
        pass


    def _fetch_parameters(self):
        return Utils.fetch_parameters(self.config_file)


    def _add_parameters(self, step_name, index, **kwargs):
        return Utils.add_parameters(self.config_file, step_name, index, **kwargs)
