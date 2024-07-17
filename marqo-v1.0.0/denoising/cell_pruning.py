import os
import yaml
import json
import numpy as np
import pandas as pd


from utils.image import ImageHandler
from utils.file import FileHandler

class Denoising:

    def __init__(self):
        pass

    def set_parameters(self, config_file):
        """
        Read and fetch the paramaters and names needed to execute the masking.

        Paramters:
            - mag_to_analyze_for_matrix:
            - marco_output_path:
            - sample_name:
            - markers_name_list:

        """
        with open(config_file) as stream:
            config_data = yaml.safe_load(stream)

        
        try:
            self.marker_index_maskforclipping = config_data['parameters']['tiling']['marker_index_maskforclipping']
            self.erosion_buffer_microns = config_data['parameters']['tiling']['erosion_buffer_microns']
            self.markers_name_list = config_data['sample']['markers']
            self.output_resolution = config_data['parameters']['image']['output_resolution']
            self.scale_factor = config_data['parameters']['image']['scale_factor']

            self.geojson_path = config_data['directories']['geojson']

            self.marco_output_path = config_data['directories']['marco_output']
            self.sample_name = config_data['sample']['name']

        except:
            self.marker_index_maskforclipping = config_data['tiling_parameters']['marker_index_maskforclipping']
            self.erosion_buffer_microns = config_data['tiling_parameters']['erosion_buffer_microns']
            self.markers_name_list = config_data['sample']['markers']
            self.output_resolution = config_data['parameters']['output_resolution']
            self.scale_factor = config_data['parameters']['SF']

            self.geojson_path = config_data['directories']['product_directories']['geojson']

            self.marco_output_path = config_data['directories']['sample_directory']
            self.sample_name = config_data['sample']['sample_name']




    def denoise_geojson2(self, cell_indices_tokeep, rawcellmetrics_df, allmarkers_geojson_list, roi_index):
        allmarkers_geojson_updated_list = []

        for i, geojson in enumerate(allmarkers_geojson_list):
            marker_name = self.markers_name_list[i]
            geojson_filename = f'{self.sample_name}_ROI#{roi_index}_{marker_name}.json'
            save_path = os.path.join(self.geojson_path, geojson_filename)

            updated_json_data = list(np.array(geojson)[cell_indices_tokeep])

            allmarkers_geojson_updated_list.append(updated_json_data)

        return allmarkers_geojson_updated_list


    def denoise_geojson(self, cell_indices_tokeep, rawcellmetrics_df, roi_index):
        keep_cells_list_forROI = cell_indices_tokeep[rawcellmetrics_df['Tile index'] == roi_index]
        allmarkers_geojson_updated_list = []

        for marker_name in self.markers_name_list:
            all_jsons_path = os.path.join(self.marco_output_path, 'geoJSONS_perROI')
            json_path = os.path.join(all_jsons_path, self.sample_name)
            json_to_update_path = all_jsons_path + '/' + self.sample_name + '_ROI#' + str(roi_index) + '_' + marker_name + '.json'

            with open(json_to_update_path) as json_file:
                json_data = json.load(json_file)

            #update json file by only keeping cells within tissue masking bounds
            updated_json_data = list(np.array(json_data)[keep_cells_list_forROI])
            
            allmarkers_geojson_updated_list.append(updated_json_data)
        
        return allmarkers_geojson_updated_list


    def denoise_tile_cellmetrics(self, roi_index, rawcellmetrics_df, allmarkers_geojson_list, tissue_mask):
    #def denoise_tile_cellmetrics(self, roi_index, _file, tissue_mask):

        eroded_thumbnail_mask_forclipping = ImageHandler.get_eroded_thumbnail(tissue_mask, self.erosion_buffer_microns, self.output_resolution, self.scale_factor)

        #rawcellmetrics_df = pd.read_csv(_file, sep=',', header=0)
        thumbnail_ys = ((rawcellmetrics_df.loc[:, (str(self.markers_name_list[self.marker_index_maskforclipping]) + ' y_coor (pixels)')] / (self.output_resolution / 1.25)).round()).astype(int)
        thumbnail_xs = ((rawcellmetrics_df.loc[:, (str(self.markers_name_list[self.marker_index_maskforclipping]) + ' x_coor (pixels)')] / (self.output_resolution / 1.25)).round()).astype(int)

        # For some cases where the mask/tissue is very near the edge we need to make sure the coordinates are falling within it
        thumbnail_ys[np.where(thumbnail_ys >= eroded_thumbnail_mask_forclipping.shape[0])] = eroded_thumbnail_mask_forclipping.shape[0] - 1
        thumbnail_xs[np.where(thumbnail_xs >= eroded_thumbnail_mask_forclipping.shape[1])] = eroded_thumbnail_mask_forclipping.shape[1] - 1

        cell_indices_tokeep = eroded_thumbnail_mask_forclipping[(thumbnail_ys, thumbnail_xs)]
        clipped_rawcellmetrics_df = rawcellmetrics_df[cell_indices_tokeep]

        #allmarkers_geojson_updated_list = self.denoise_geojson(cell_indices_tokeep, rawcellmetrics_df, roi_index)
        allmarkers_geojson_updated_list = self.denoise_geojson2(cell_indices_tokeep, rawcellmetrics_df, allmarkers_geojson_list, roi_index)

        #reset cell indices and export dataframe
        clipped_rawcellmetrics_df = clipped_rawcellmetrics_df.reset_index()
        clipped_rawcellmetrics_df = clipped_rawcellmetrics_df.drop(columns=['index']) #remove original indices column
        clipped_rawcellmetrics_df.index.rename('Cell index', inplace=True); #name new indices column

        return clipped_rawcellmetrics_df, allmarkers_geojson_updated_list
 
